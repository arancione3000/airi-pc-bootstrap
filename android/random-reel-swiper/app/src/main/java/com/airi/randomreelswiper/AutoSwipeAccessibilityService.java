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

        // Safe central area: avoids status/quick-settings region, bottom navigation,
        // and the upper-right STOP overlay.
        final float minX = w * 0.18f;
        final float maxX = w * 0.82f;
        final float minY = h * 0.22f;
        final float maxY = h * 0.80f;

        float startX = randomRange(w * 0.28f, w * 0.72f);
        float startY = randomRange(h * 0.34f, h * 0.70f);

        // Fully random direction: 0..360 degrees, so gestures may go up, down,
        // left, right or diagonally. X/Y amplitudes are scaled to screen shape.
        double angle = random.nextDouble() * Math.PI * 2.0;
        float horizontalReach = randomRange(w * 0.18f, w * 0.34f);
        float verticalReach = randomRange(h * 0.16f, h * 0.30f);

        float endX = clamp(startX + (float) Math.cos(angle) * horizontalReach, minX, maxX);
        float endY = clamp(startY + (float) Math.sin(angle) * verticalReach, minY, maxY);

        // If clamping made the gesture too short, push it in the opposite random direction.
        float dx = endX - startX;
        float dy = endY - startY;
        float minDistance = Math.min(w, h) * 0.14f;
        if (Math.hypot(dx, dy) < minDistance) {
            endX = clamp(startX - (float) Math.cos(angle) * horizontalReach, minX, maxX);
            endY = clamp(startY - (float) Math.sin(angle) * verticalReach, minY, maxY);
        }

        long duration = 2_000L + random.nextInt(2_001); // 2–4 seconds

        Path path = new Path();
        path.moveTo(startX, startY);

        // Curved control point adds small natural variation while staying safe.
        float midX = clamp(
                (startX + endX) / 2f + randomRange(-w * 0.08f, w * 0.08f),
                minX,
                maxX
        );
        float midY = clamp(
                (startY + endY) / 2f + randomRange(-h * 0.05f, h * 0.05f),
                minY,
                maxY
        );
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
