package com.airipc.control

import android.app.KeyguardManager
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch
import org.json.JSONObject

object ControlSessionManager {
    private const val CHANNEL_ID = "airi_control_active_v2"
    private const val NOTIFICATION_ID = 4302

    private val scope = CoroutineScope(
        SupervisorJob() + Dispatchers.Main.immediate
    )

    private val _active = MutableStateFlow(false)
    val active: StateFlow<Boolean> = _active

    private val _controllerConnected = MutableStateFlow(false)
    val controllerConnected: StateFlow<Boolean> = _controllerConnected

    private var client: ControlClient? = null
    private var appContext: Context? = null

    fun start(context: Context) {
        if (_active.value) return
        val app = context.applicationContext
        appContext = app
        ensureNotificationChannel(app)
        val c = ControlClient(
            context = app,
            scope = scope,
            onCommand = { command -> handleCommand(app, command) },
            onControllerState = { connected ->
                _controllerConnected.value = connected
            },
        )
        client = c
        _active.value = true
        _controllerConnected.value = false
        showActiveNotification(app)
        c.start()
    }

    fun stop(context: Context? = null) {
        runCatching { client?.stop() }
        client = null
        _controllerConnected.value = false
        _active.value = false
        val app = context?.applicationContext ?: appContext
        app?.getSystemService(NotificationManager::class.java)
            ?.cancel(NOTIFICATION_ID)
    }

    private suspend fun handleCommand(
        context: Context,
        command: PhoneCommand,
    ) {
        val locked = context.getSystemService(
            KeyguardManager::class.java
        ).isDeviceLocked
        val metrics = context.resources.displayMetrics
        val access = awaitAccessibility()
        val result = JSONObject()
            .put("op", command.op)
            .put("locked", locked)
            .put("width", metrics.widthPixels)
            .put("height", metrics.heightPixels)

        if (
            command.op !in setOf("status", "stop") &&
            locked
        ) {
            result
                .put("ok", false)
                .put("error", "Device is locked")
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
                        .put(
                            "controller_connected",
                            _controllerConnected.value,
                        )
                        .put("accessibility", access != null)
                        .put(
                            "screen_capture",
                            access != null && Build.VERSION.SDK_INT >= 30,
                        )
                        .put("package", observed?.first.orEmpty())
                }

                "observe" -> {
                    val service = access
                        ?: error("Accessibility unavailable")
                    val observed = service.observe()
                    result
                        .put("ok", true)
                        .put("package", observed.first)
                        .put("text", observed.second)
                }

                "tap" -> {
                    val service = access
                        ?: error("Accessibility unavailable")
                    result.put(
                        "ok",
                        service.tap(
                            command.args.getDouble("x").toFloat(),
                            command.args.getDouble("y").toFloat(),
                        ),
                    )
                }

                "swipe" -> {
                    val service = access
                        ?: error("Accessibility unavailable")
                    result.put(
                        "ok",
                        service.swipe(
                            command.args.getDouble("x1").toFloat(),
                            command.args.getDouble("y1").toFloat(),
                            command.args.getDouble("x2").toFloat(),
                            command.args.getDouble("y2").toFloat(),
                            command.args.optLong("duration_ms", 450),
                        ),
                    )
                }

                "back" -> {
                    val service = access
                        ?: error("Accessibility unavailable")
                    result.put("ok", service.back())
                }

                "home" -> {
                    val service = access
                        ?: error("Accessibility unavailable")
                    result.put("ok", service.home())
                }

                "recents" -> {
                    val service = access
                        ?: error("Accessibility unavailable")
                    result.put("ok", service.recents())
                }

                "text" -> {
                    val service = access
                        ?: error("Accessibility unavailable")
                    val (ok, message) = service.setText(
                        command.args.optString("text")
                    )
                    result.put("ok", ok).put("message", message)
                }

                "screenshot" -> {
                    val service = access
                        ?: error("Accessibility unavailable")
                    val jpeg = service.captureJpeg()
                    val frame = client?.uploadEncryptedFrame(
                        command,
                        jpeg,
                    ) ?: error("Control client unavailable")
                    result.put("ok", true)
                    frame.keys().forEach { key ->
                        result.put(key, frame.get(key))
                    }
                }

                "stop" -> {
                    result
                        .put("ok", true)
                        .put("message", "Remote session stopped")
                }

                else -> {
                    result
                        .put("ok", false)
                        .put("error", "Unsupported command")
                }
            }
        } catch (e: Exception) {
            result
                .put("ok", false)
                .put(
                    "error",
                    e.message ?: e.javaClass.simpleName,
                )
        }

        client?.publishResult(command, result)
        if (command.op == "stop") {
            delay(200)
            stop(context)
        }
    }

    private suspend fun awaitAccessibility(): AiriAccessibilityService? {
        repeat(20) {
            AiriAccessibilityService.instance?.let { return it }
            delay(100)
        }
        return AiriAccessibilityService.instance
    }

    private fun ensureNotificationChannel(context: Context) {
        context.getSystemService(NotificationManager::class.java)
            .createNotificationChannel(
                NotificationChannel(
                    CHANNEL_ID,
                    "Airi Control",
                    NotificationManager.IMPORTANCE_HIGH,
                ).apply {
                    description =
                        "Mostra quando una sessione Airi Control e attiva"
                }
            )
    }

    private fun showActiveNotification(context: Context) {
        val open = PendingIntent.getActivity(
            context,
            1,
            Intent(context, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or
                PendingIntent.FLAG_IMMUTABLE,
        )
        val stop = PendingIntent.getBroadcast(
            context,
            2,
            Intent(context, StopControlReceiver::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or
                PendingIntent.FLAG_IMMUTABLE,
        )
        val notification = NotificationCompat.Builder(
            context,
            CHANNEL_ID,
        )
            .setSmallIcon(R.drawable.ic_airi_control)
            .setContentTitle("Airi Control attivo")
            .setContentText(
                "Airi-PC puo interagire finche non premi STOP."
            )
            .setContentIntent(open)
            .setOngoing(true)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .addAction(0, "STOP", stop)
            .build()
        context.getSystemService(NotificationManager::class.java)
            .notify(NOTIFICATION_ID, notification)
    }
}
