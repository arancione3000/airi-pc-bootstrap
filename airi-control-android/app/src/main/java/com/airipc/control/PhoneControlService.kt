package com.airipc.control

import android.app.Activity
import android.app.KeyguardManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat
import androidx.core.app.ServiceCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject

class PhoneControlService : Service() {
    companion object {
        private const val ACTION_START = "com.airipc.control.START"
        private const val ACTION_STOP = "com.airipc.control.STOP"
        private const val EXTRA_RESULT_CODE = "projection_result_code"
        private const val EXTRA_DATA = "projection_data"
        private const val CHANNEL_ID = "airi_control_active"
        private const val NOTIFICATION_ID = 4301

        private val _active = MutableStateFlow(false)
        val active: StateFlow<Boolean> = _active

        fun startIntent(context: Context, resultCode: Int, data: Intent): Intent =
            Intent(context, PhoneControlService::class.java)
                .setAction(ACTION_START)
                .putExtra(EXTRA_RESULT_CODE, resultCode)
                .putExtra(EXTRA_DATA, data)

        fun stopIntent(context: Context): Intent =
            Intent(context, PhoneControlService::class.java).setAction(ACTION_STOP)
    }

    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private lateinit var capture: ScreenCaptureSession
    private var client: ControlClient? = null
    private var stopping = false

    override fun onCreate() {
        super.onCreate()
        capture = ScreenCaptureSession(this)
        ensureChannel()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            ACTION_STOP -> {
                stopAll(disableAccessibility = true)
                return START_NOT_STICKY
            }
            ACTION_START -> {
                startVisibleForeground()
                if (!_active.value) {
                    val resultCode = intent.getIntExtra(EXTRA_RESULT_CODE, Activity.RESULT_CANCELED)
                    val data = projectionIntent(intent)
                    if (resultCode != Activity.RESULT_OK || data == null) {
                        stopAll(disableAccessibility = false)
                        return START_NOT_STICKY
                    }
                    scope.launch {
                        runCatching { startControl(resultCode, data) }
                            .onFailure { stopAll(disableAccessibility = false) }
                    }
                }
            }
        }
        return START_NOT_STICKY
    }

    private fun startVisibleForeground() {
        val open = PendingIntent.getActivity(
            this,
            1,
            Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getService(
            this,
            2,
            stopIntent(this),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(this, CHANNEL_ID)
            .setSmallIcon(com.airipc.control.R.drawable.ic_airi_control)
            .setContentTitle("Airi Control attivo")
            .setContentText("Airi può vedere e controllare lo schermo. Tocca STOP per interrompere.")
            .setContentIntent(open)
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .addAction(0, "STOP", stop)
            .build()
        val type = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
        } else 0
        ServiceCompat.startForeground(this, NOTIFICATION_ID, notification, type)
    }

    private fun startControl(resultCode: Int, data: Intent) {
        if (!AiriAccessibilityService.connected()) {
            error("Accessibility service is not enabled")
        }
        capture.start(resultCode, data)
        val c = ControlClient(this, scope, ::handleCommand)
        client = c
        c.start()
        _active.value = true
    }

    private suspend fun handleCommand(command: PhoneCommand) {
        val access = AiriAccessibilityService.instance
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
                    val observed = access?.observe()
                    result
                        .put("ok", true)
                        .put("active", _active.value)
                        .put("accessibility", access != null)
                        .put("screen_capture", capture.active())
                        .put("package", observed?.first.orEmpty())
                }
                "observe" -> {
                    val observed = access?.observe() ?: ("" to "")
                    result
                        .put("ok", access != null)
                        .put("package", observed.first)
                        .put("text", observed.second)
                }
                "tap" -> {
                    val ok = access?.tap(
                        command.args.getDouble("x").toFloat(),
                        command.args.getDouble("y").toFloat(),
                    ) ?: false
                    result.put("ok", ok)
                }
                "swipe" -> {
                    val ok = access?.swipe(
                        command.args.getDouble("x1").toFloat(),
                        command.args.getDouble("y1").toFloat(),
                        command.args.getDouble("x2").toFloat(),
                        command.args.getDouble("y2").toFloat(),
                        command.args.optLong("duration_ms", 450),
                    ) ?: false
                    result.put("ok", ok)
                }
                "back" -> result.put("ok", access?.back() ?: false)
                "home" -> result.put("ok", access?.home() ?: false)
                "recents" -> result.put("ok", access?.recents() ?: false)
                "text" -> {
                    val (ok, message) = access?.setText(command.args.optString("text"))
                        ?: (false to "Accessibility unavailable")
                    result.put("ok", ok).put("message", message)
                }
                "screenshot" -> {
                    val jpeg = capture.captureJpeg()
                    val frame = client?.uploadEncryptedFrame(command, jpeg)
                        ?: error("Control client unavailable")
                    result.put("ok", true)
                    frame.keys().forEach { key -> result.put(key, frame.get(key)) }
                }
                "stop" -> result.put("ok", true).put("message", "Control stopping")
                else -> result.put("ok", false).put("error", "Unsupported command")
            }
        } catch (e: Exception) {
            result.put("ok", false).put("error", e.message ?: e.javaClass.simpleName)
        }

        client?.publishResult(command, result)
        if (command.op == "stop") {
            delay(250)
            stopAll(disableAccessibility = true)
        }
    }

    private fun stopAll(disableAccessibility: Boolean) {
        if (stopping) return
        stopping = true
        _active.value = false
        runCatching { client?.stop() }
        client = null
        runCatching { capture.stop() }
        if (disableAccessibility) AiriAccessibilityService.disableFromApp()
        ServiceCompat.stopForeground(this, ServiceCompat.STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        _active.value = false
        runCatching { client?.stop() }
        runCatching { capture.stop() }
        scope.cancel()
        super.onDestroy()
    }

    private fun ensureChannel() {
        val manager = getSystemService(NotificationManager::class.java)
        manager.createNotificationChannel(
            NotificationChannel(
                CHANNEL_ID,
                "Airi Control",
                NotificationManager.IMPORTANCE_HIGH,
            ).apply {
                description = "Mostra sempre quando Airi Control è attivo"
            }
        )
    }

    @Suppress("DEPRECATION")
    private fun projectionIntent(source: Intent): Intent? =
        if (Build.VERSION.SDK_INT >= 33) {
            source.getParcelableExtra(EXTRA_DATA, Intent::class.java)
        } else {
            source.getParcelableExtra(EXTRA_DATA)
        }
}
