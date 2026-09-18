package com.airipc.control

import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.PixelFormat
import android.hardware.display.DisplayManager
import android.hardware.display.VirtualDisplay
import android.media.ImageReader
import android.media.projection.MediaProjection
import android.media.projection.MediaProjectionManager
import android.os.Handler
import android.os.Looper
import java.io.ByteArrayOutputStream

class ScreenCaptureSession(private val context: Context) {
    private var projection: MediaProjection? = null
    private var reader: ImageReader? = null
    private var display: VirtualDisplay? = null
    @Volatile private var stopped = true

    fun start(resultCode: Int, data: Intent) {
        stop()
        val manager = context.getSystemService(MediaProjectionManager::class.java)
        val metrics = context.resources.displayMetrics
        val width = metrics.widthPixels.coerceAtLeast(1)
        val height = metrics.heightPixels.coerceAtLeast(1)
        val density = metrics.densityDpi.coerceAtLeast(1)
        val mp = manager.getMediaProjection(resultCode, data)
            ?: error("MediaProjection permission unavailable")
        projection = mp
        reader = ImageReader.newInstance(width, height, PixelFormat.RGBA_8888, 3)
        mp.registerCallback(object : MediaProjection.Callback() {
            override fun onStop() {
                stopped = true
            }
        }, Handler(Looper.getMainLooper()))
        display = mp.createVirtualDisplay(
            "AiriControlPhone",
            width,
            height,
            density,
            DisplayManager.VIRTUAL_DISPLAY_FLAG_AUTO_MIRROR,
            reader!!.surface,
            null,
            null,
        )
        stopped = false
    }

    fun active(): Boolean = !stopped && projection != null && reader != null

    fun captureJpeg(quality: Int = 68): ByteArray {
        val imageReader = reader ?: error("Screen capture is not active")
        var image = imageReader.acquireLatestImage()
        var tries = 0
        while (image == null && tries < 25) {
            Thread.sleep(80)
            image = imageReader.acquireLatestImage()
            tries++
        }
        val frame = image ?: error("No screen frame available")
        frame.use {
            val plane = it.planes.first()
            val buffer = plane.buffer
            val pixelStride = plane.pixelStride
            val rowStride = plane.rowStride
            val rowPadding = rowStride - pixelStride * it.width
            val paddedWidth = it.width + rowPadding / pixelStride
            val padded = Bitmap.createBitmap(paddedWidth, it.height, Bitmap.Config.ARGB_8888)
            padded.copyPixelsFromBuffer(buffer)
            val cropped = Bitmap.createBitmap(padded, 0, 0, it.width, it.height)
            val out = ByteArrayOutputStream()
            cropped.compress(Bitmap.CompressFormat.JPEG, quality.coerceIn(35, 90), out)
            cropped.recycle()
            padded.recycle()
            return out.toByteArray()
        }
    }

    fun stop() {
        stopped = true
        runCatching { display?.release() }
        runCatching { reader?.close() }
        runCatching { projection?.stop() }
        display = null
        reader = null
        projection = null
    }
}
