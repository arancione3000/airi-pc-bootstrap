package com.airi.randomreelswiper;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.GestureDescription;
import android.graphics.Color;
import android.graphics.Path;
import android.graphics.PixelFormat;
import android.graphics.Point;
import android.graphics.drawable.GradientDrawable;
import android.os.Handler;
import android.os.Looper;
import android.view.Display;
import android.view.Gravity;
import android.view.WindowManager;
import android.view.accessibility.AccessibilityEvent;
import android.widget.Button;

import java.util.Random;

public class AutoSwipeAccessibilityService extends AccessibilityService {
    private static AutoSwipeAccessibilityService instance;

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final Random random = new Random();
    private boolean running = false;
    private Button stopOverlay;
    private WindowManager windowManager;

    private final Runnable swipeRunnable = new Runnable() {
        @Override
        public void run() {
            if (!running) return;
            performRandomSwipe();
            scheduleNextSwipe();
        }
    };

    public static AutoSwipeAccessibilityService getInstance() {
        return instance;
    }

    public boolean isRunning() {
        return running;
    }

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        instance = this;
        windowManager = (WindowManager) getSystemService(WINDOW_SERVICE);
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        // Deliberately unused: the app does not inspect other apps' screen contents.
    }

    @Override
    public void onInterrupt() {
        stopAutomation();
    }

    @Override
    public void onDestroy() {
        stopAutomation();
        instance = null;
        super.onDestroy();
    }

    public void startAutomation() {
        if (running) return;
        running = true;
        showStopOverlay();

        // The first swipe follows the same requested random 10–60 second interval.
        scheduleNextSwipe();
    }

    public void stopAutomation() {
        running = false;
        handler.removeCallbacks(swipeRunnable);
        hideStopOverlay();
    }

    private void scheduleNextSwipe() {
        if (!running) return;
        long delayMs = 10_000L + random.nextInt(50_001); // 10–60 s
        handler.postDelayed(swipeRunnable, delayMs);
    }

    private void performRandomSwipe() {
        if (windowManager == null) return;

        Point size = new Point();
        Display display = windowManager.getDefaultDisplay();
        display.getRealSize(size);
        int w = size.x;
        int h = size.y;

        // Central safe corridor: no top controls/status shade, no bottom nav area,
        // and no contact with the STOP overlay.
        float startX = randomRange(w * 0.30f, w * 0.70f);
        float startY = randomRange(h * 0.64f, h * 0.80f);

        // Upward swipe with random diagonal drift.
        float driftX = randomRange(-w * 0.18f, w * 0.18f);
        float endX = clamp(startX + driftX, w * 0.22f, w * 0.78f);
        float endY = randomRange(h * 0.28f, h * 0.45f);

        long duration = 280L + random.nextInt(521); // 280–800 ms

        Path path = new Path();
        path.moveTo(startX, startY);

        // Slight curve makes consecutive gestures less mechanically identical.
        float midX = clamp(
                (startX + endX) / 2f + randomRange(-w * 0.06f, w * 0.06f),
                w * 0.20f,
                w * 0.80f
        );
        float midY = (startY + endY) / 2f;
        path.quadTo(midX, midY, endX, endY);

        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(new GestureDescription.StrokeDescription(path, 0, duration))
                .build();

        dispatchGesture(gesture, null, null);
    }

    private void showStopOverlay() {
        if (stopOverlay != null || windowManager == null) return;

        stopOverlay = new Button(this);
        stopOverlay.setText("STOP");
        stopOverlay.setTextColor(Color.WHITE);
        stopOverlay.setTextSize(13);
        stopOverlay.setAllCaps(true);
        stopOverlay.setPadding(dp(10), 0, dp(10), 0);

        GradientDrawable bg = new GradientDrawable();
        bg.setColor(Color.rgb(190, 35, 35));
        bg.setCornerRadius(dp(18));
        stopOverlay.setBackground(bg);
        stopOverlay.setOnClickListener(v -> stopAutomation());

        WindowManager.LayoutParams params = new WindowManager.LayoutParams(
                dp(88),
                dp(48),
                WindowManager.LayoutParams.TYPE_ACCESSIBILITY_OVERLAY,
                WindowManager.LayoutParams.FLAG_NOT_FOCUSABLE |
                        WindowManager.LayoutParams.FLAG_LAYOUT_IN_SCREEN,
                PixelFormat.TRANSLUCENT
        );
        params.gravity = Gravity.TOP | Gravity.END;
        params.x = dp(12);
        params.y = dp(56);

        windowManager.addView(stopOverlay, params);
    }

    private void hideStopOverlay() {
        if (stopOverlay != null && windowManager != null) {
            try {
                windowManager.removeView(stopOverlay);
            } catch (Exception ignored) {
            }
            stopOverlay = null;
        }
    }

    private float randomRange(float min, float max) {
        return min + random.nextFloat() * (max - min);
    }

    private float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
