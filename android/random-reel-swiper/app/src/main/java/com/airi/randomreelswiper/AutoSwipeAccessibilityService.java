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
import java.util.Set;

public class AutoSwipeAccessibilityService extends AccessibilityService {
    private static AutoSwipeAccessibilityService instance;

    private static final Set<String> ALLOWED_PACKAGES = Set.of(
            "com.instagram.android",
            "com.instagram.lite"
    );

    private final Handler handler = new Handler(Looper.getMainLooper());
    private final Random random = new Random();
    private boolean running = false;
    private Button stopOverlay;
    private WindowManager windowManager;
    private volatile String foregroundPackage = "";

    private final Runnable swipeRunnable = new Runnable() {
        @Override
        public void run() {
            if (!running) return;
            if (isAllowedForegroundPackage()) {
                performRandomSwipe();
            }
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
        // We only keep the foreground package name. Screen contents are never read.
        CharSequence packageName = event.getPackageName();
        if (packageName != null) {
            foregroundPackage = packageName.toString();
        }
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

    static boolean isAllowedPackage(String packageName) {
        return packageName != null && ALLOWED_PACKAGES.contains(packageName);
    }

    private boolean isAllowedForegroundPackage() {
        return isAllowedPackage(foregroundPackage);
    }

    private void performRandomSwipe() {
        if (windowManager == null || !isAllowedForegroundPackage()) return;

        Point size = new Point();
        Display display = windowManager.getDefaultDisplay();
        display.getRealSize(size);

        GesturePlan plan = GesturePlan.random(random, size.x, size.y);

        Path path = new Path();
        path.moveTo(plan.startX, plan.startY);
        path.quadTo(plan.midX, plan.midY, plan.endX, plan.endY);

        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(new GestureDescription.StrokeDescription(path, 0, plan.durationMs))
                .build();

        // This dispatches one continuous drag stroke; it never dispatches a tap/click action.
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
