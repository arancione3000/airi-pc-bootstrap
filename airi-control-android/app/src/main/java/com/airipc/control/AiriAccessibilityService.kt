package com.airipc.control

import android.accessibilityservice.AccessibilityService
import android.accessibilityservice.GestureDescription
import android.graphics.Bitmap
import android.graphics.Path
import android.os.Build
import android.os.Bundle
import android.view.Display
import android.view.accessibility.AccessibilityEvent
import android.view.accessibility.AccessibilityNodeInfo
import java.io.ByteArrayOutputStream
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCoroutine

class AiriAccessibilityService : AccessibilityService() {
    companion object {
        @Volatile var instance: AiriAccessibilityService? = null
            private set

        fun connected(): Boolean = instance != null

        fun disableFromApp() {
            runCatching { instance?.disableSelf() }
        }
    }

    override fun onServiceConnected() {
        super.onServiceConnected()
        instance = this
    }

    override fun onAccessibilityEvent(event: AccessibilityEvent?) = Unit
    override fun onInterrupt() = Unit

    override fun onUnbind(intent: android.content.Intent?): Boolean {
        if (instance === this) instance = null
        return super.onUnbind(intent)
    }

    override fun onDestroy() {
        if (instance === this) instance = null
        super.onDestroy()
    }

    fun tap(x: Float, y: Float): Boolean {
        val path = Path().apply { moveTo(x, y) }
        val stroke = GestureDescription.StrokeDescription(path, 0, 80)
        return dispatchGesture(
            GestureDescription.Builder().addStroke(stroke).build(),
            null,
            null,
        )
    }

    fun swipe(
        x1: Float,
        y1: Float,
        x2: Float,
        y2: Float,
        durationMs: Long,
    ): Boolean {
        val path = Path().apply {
            moveTo(x1, y1)
            lineTo(x2, y2)
        }
        val stroke = GestureDescription.StrokeDescription(
            path,
            0,
            durationMs.coerceIn(80, 2500),
        )
        return dispatchGesture(
            GestureDescription.Builder().addStroke(stroke).build(),
            null,
            null,
        )
    }

    fun back(): Boolean = performGlobalAction(GLOBAL_ACTION_BACK)
    fun home(): Boolean = performGlobalAction(GLOBAL_ACTION_HOME)
    fun recents(): Boolean = performGlobalAction(GLOBAL_ACTION_RECENTS)

    fun setText(text: String): Pair<Boolean, String> {
        val root = rootInActiveWindow ?: return false to "No active window"
        val node = root.findFocus(AccessibilityNodeInfo.FOCUS_INPUT)
            ?: findEditable(root)
            ?: return false to "No editable field focused"
        if (node.isPassword) return false to "Password fields are blocked"
        val args = Bundle().apply {
            putCharSequence(
                AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                text,
            )
        }
        return if (
            node.performAction(
                AccessibilityNodeInfo.ACTION_SET_TEXT,
                args,
            )
        ) {
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

    suspend fun captureJpeg(): ByteArray {
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
                            bitmap.compress(
                                Bitmap.CompressFormat.JPEG,
                                70,
                                out,
                            )
                            bitmap.recycle()
                            continuation.resume(out.toByteArray())
                        } catch (e: Exception) {
                            continuation.resumeWithException(e)
                        }
                    }

                    override fun onFailure(errorCode: Int) {
                        continuation.resumeWithException(
                            IllegalStateException(
                                "Accessibility screenshot failed: $errorCode"
                            )
                        )
                    }
                },
            )
        }
    }

    private fun findEditable(
        node: AccessibilityNodeInfo?,
    ): AccessibilityNodeInfo? {
        if (node == null) return null
        if (node.isEditable) return node
        for (i in 0 until node.childCount) {
            findEditable(node.getChild(i))?.let { return it }
        }
        return null
    }

    private fun collectText(
        node: AccessibilityNodeInfo?,
        out: StringBuilder,
        depth: Int,
    ) {
        if (node == null || out.length >= 1800 || depth > 30) return
        if (!node.isPassword) {
            val text = node.text?.toString()?.trim().orEmpty()
            val desc = node.contentDescription?.toString()?.trim().orEmpty()
            if (text.isNotBlank()) out.append(text).append('\n')
            if (desc.isNotBlank() && desc != text) {
                out.append(desc).append('\n')
            }
        }
        for (i in 0 until node.childCount) {
            collectText(node.getChild(i), out, depth + 1)
        }
    }
}
