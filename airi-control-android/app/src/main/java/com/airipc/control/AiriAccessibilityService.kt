package com.airipc.control

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.app.KeyguardManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Path
import android.os.Build
import android.os.Bundle
import android.view.Display
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.io.ByteArrayOutputStream
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class AiriAccessibilityService : AccessibilityService() {
    companion object {
        private const val CHANNEL_ID = "airi_control_active_v2"
        private const val NOTIFICATION_ID = 4302

        @Volatile var instance: AiriAccessibilityService? = null
            private set

        private val _active = MutableStateFlow(false)
        val active: StateFlow<Boolean> = _active

        private val _controllerConnected = MutableStateFlow(false)
        val controllerConnected: StateFlow<Boolean> = _controllerConnected

        fun connected(): Boolean = instance != null

        fun startSession(): Boolean {
            val service = instance ?: return false
            service.startSessionInternal()
            return true
        }

        fun stopSession(): Boolean {
            val service = instance ?: return false
            service.stopSessionInternal()
            return true
        }

        fun disableFromApp() {
            instance?.stopSessionInternal()
            runCatching { instance?.disableSelf() }
        }
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private var client: ControlClient? = null

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
        ensureNotificationChannel()
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit
    override fun onInterrupt() = Unit

    override fun onUnbind(intent: Intent?): Boolean {
        stopSessionInternal()
        if (instance === this) instance = null
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        stopSessionInternal()
        if (instance === this) instance = null
        scope.cancel()
        super.onDestroy()
    }

    private fun startSessionInternal() {
        if (_active.value) return
        val c = ControlClient(
            context = this,
            scope = scope,
            onCommand = ::handleCommand,
            onControllerState = { connected -> _controllerConnected.value = connected },
        )
        client = c
        _active.value = true
        _controllerConnected.value = false
        showActiveNotification()
        c.start()
    }

    private fun stopSessionInternal() {
        runCatching { client?.stop() }
        client = null
        _controllerConnected.value = false
        _active.value = false
        getSystemService(NotificationManager::class.java).cancel(NOTIFICATION_ID)
    }

    private suspend fun handleCommand(command: PhoneCommand) {
        val locked = getSystemService(KeyguardManager::class.java).isDeviceLocked
        val metrics = resources.displayMetrics
        val result = JSONObject()
            .put("op", command.op)
            .put("locked", locked)
            .put("width", metrics.widthPixels)
            .put("height", metrics.heightPixels)

        if (command.op !in setOf("status", "stop") && locked) {
            result.put("ok", false).put("error", "Device is locked")
            client?.publishResult(command, result)
            return
        }

        try {
            when (command.op) {
                "status" -> {
                    val observed = observe()
                    result
                        .put("ok", true)
                        .put("active", _active.value)
                        .put("controller_connected", _controllerConnected.value)
                        .put("accessibility", true)
                        .put("screen_capture", Build.VERSION.SDK_INT >= 30)
                        .put("package", observed.first)
                }
                "observe" -> {
                    val observed = observe()
                    result
                        .put("ok", true)
                        .put("package", observed.first)
                        .put("text", observed.second)
                }
                "tap" -> {
                    result.put(
                        "ok",
                        tap(
                            command.args.getDouble("x").toFloat(),
                            command.args.getDouble("y").toFloat(),
                        ),
                    )
                }
                "swipe" -> {
                    result.put(
                        "ok",
                        swipe(
                            command.args.getDouble("x1").toFloat(),
                            command.args.getDouble("y1").toFloat(),
                            command.args.getDouble("x2").toFloat(),
                            command.args.getDouble("y2").toFloat(),
                            command.args.optLong("duration_ms", 450),
                        ),
                    )
                }
                "back" -> result.put("ok", performGlobalAction(GLOBAL_ACTION_BACK))
                "home" -> result.put("ok", performGlobalAction(GLOBAL_ACTION_HOME))
                "recents" -> result.put("ok", performGlobalAction(GLOBAL_ACTION_RECENTS))
                "text" -> {
                    val (ok, message) = setText(command.args.optString("text"))
                    result.put("ok", ok).put("message", message)
                }
                "screenshot" -> {
                    val jpeg = captureJpeg()
                    val frame = client?.uploadEncryptedFrame(command, jpeg)
                        ?: error("Control client unavailable")
                    result.put("ok", true)
                    frame.keys().forEach { key -> result.put(key, frame.get(key)) }
                }
                "stop" -> {
                    result.put("ok", true).put("message", "Remote session stopped")
                }
                else -> result.put("ok", false).put("error", "Unsupported command")
            }
        } catch (e: Exception) {
            result.put("ok", false).put("error", e.message ?: e.javaClass.simpleName)
        }

        client?.publishResult(command, result)
        if (command.op == "stop") {
            delay(200)
            stopSessionInternal()
        }
    }

    fun tap(x: Float, y: Float): Boolean {
        val path = Path().apply { moveTo(x, y) }
        val stroke = GestureDescription.StrokeDescription(path, 0, 80)
        return dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
    }

    fun swipe(x1: Float, y1: Float, x2: Float, y2: Float, durationMs: Long): Boolean {
        val path = Path().apply {
            moveTo(x1, y1)
            lineTo(x2, y2)
        }
        val stroke = GestureDescription.StrokeDescription(path, 0, durationMs.coerceIn(80, 2500))
        return dispatchGesture(GestureDescription.Builder().addStroke(stroke).build(), null, null)
    }

    fun setText(text: String): Pair<Boolean, String> {
        val root = rootInActiveWindow ?: return false to "No active window"
        val node = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            ?: findEditable(root)
            ?: return false to "No editable field focused"
        if (node.isPassword) return false to "Password fields are blocked"
        val args = Bundle().apply {
            putCharSequence(AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE, text)
        }
        return if (node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) {
            true to "Text inserted"
        } else {
            false to "Focused field rejected ACTION_SET_TEXT"
        }
    }

    fun observe(): Pair<String, String> {
        val root = rootInActiveWindow ?: return "" to ""
        val pkg = root.packageName?.toString().orEmpty()
        val out = StringBuilder()
        collectText(root, out, 0)
        return pkg to out.toString().take(1800)
    }

    private suspend fun captureJpeg(): ByteArray {
        if (Build.VERSION.SDK_INT < 30) {
            error("Screenshots require Android 11 or newer")
        }
        return suspendCoroutine { continuation ->
            takeScreenshot(
                Display.DEFAULT_DISPLAY,
                mainExecutor,
                object : TakeScreenshotCallback {
                    override fun onSuccess(screenshot: ScreenshotResult) {
                        try {
                            val hardware = screenshot.hardwareBuffer
                            val bitmap = Bitmap.wrapHardwareBuffer(
                                hardware,
                                screenshot.colorSpace,
                            )?.copy(Bitmap.Config.ARGB_8888, false)
                                ?: error("Unable to map screenshot buffer")
                            hardware.close()
                            val out = ByteArrayOutputStream()
                            bitmap.compress(Bitmap.CompressFormat.JPEG, 70, out)
                            bitmap.recycle()
                            continuation.resume(out.toByteArray())
                        } catch (e: Exception) {
                            continuation.resumeWithException(e)
                        }
                    }

                    override fun onFailure(errorCode: Int) {
                        continuation.resumeWithException(
                            IllegalStateException("Accessibility screenshot failed: $errorCode")
                        )
                    }
                },
            )
        }
    }

    private fun findEditable(node: AccessibilityNodeInfo?): AccessibilityNodeInfo? {
        if (node == null) return null
        if (node.isEditable) return node
        for (i in 0 until node.childCount) {
            findEditable(node.getChild(i))?.let { return it }
        }
        return null
    }

    private fun collectText(node: AccessibilityNodeInfo?, out: StringBuilder, depth: Int) {
        if (node == null || out.length >= 1800 || depth > 30) return
        if (!node.isPassword) {
            val text = node.text?.toString()?.trim().orEmpty()
            val desc = node.contentDescription?.toString()?.trim().orEmpty()
            if (text.isNotBlank()) out.append(text).append('\n')
            if (desc.isNotBlank() && desc != text) out.append(desc).append('\n')
        }
        for (i in 0 until node.childCount) collectText(node.getChild(i), out, depth + 1)
    }

    private fun ensureNotificationChannel() {
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Airi Control",
                NotificationManager.IMPORTANCE_HIGH,
            ).apply {
                description = "Mostra quando una sessione Airi Control e attiva"
            }
        )
    }

    private fun showActiveNotification() {
        val open = PendingIntent.getActivity(
            this,
            1,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getBroadcast(
            this,
            2,
            Intent(this, StopControlReceiver::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(R.drawable.ic_airi_control)
            .setContentTitle("Airi Control attivo")
            .setContentText("Airi-PC puo interagire finche non premi STOP.")
            .setContentIntent(open)
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .addAction(0, "STOP", stop)
            .build()
        getSystemService(NotificationManager::class.java).notify(NOTIFICATION_ID, notification)
    }
}
