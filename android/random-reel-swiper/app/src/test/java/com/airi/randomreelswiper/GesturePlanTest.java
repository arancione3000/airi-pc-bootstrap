package com.airi.randomreelswiper;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

import java.util.Random;

public class GesturePlanTest {

    @Test
    public void generatedGesturesStayInsideSafeAreaAndRemainLongDrags() {
        int[][] sizes = {
                {720, 1600},
                {1080, 1920},
                {1080, 2400},
                {1440, 3200}
        };

        for (int[] size : sizes) {
            int width = size[0];
            int height = size[1];
            float minX = width * 0.12f;
            float maxX = width * 0.88f;
            float minY = height * 0.24f;
            float maxY = height * 0.80f;
            float minDistance = Math.min(width, height) * 0.34f;

            Random random = new Random(width * 10000L + height);

            for (int i = 0; i < 25000; i++) {
                GesturePlan plan = GesturePlan.random(random, width, height);

                assertTrue(plan.startX >= minX && plan.startX <= maxX);
                assertTrue(plan.startY >= minY && plan.startY <= maxY);
                assertTrue(plan.midX >= minX && plan.midX <= maxX);
                assertTrue(plan.midY >= minY && plan.midY <= maxY);
                assertTrue(plan.endX >= minX && plan.endX <= maxX);
                assertTrue(plan.endY >= minY && plan.endY <= maxY);

                assertTrue(plan.durationMs >= 2000L);
                assertTrue(plan.durationMs <= 4000L);
                assertTrue("Gesture became too short: " + plan.distance(),
                        plan.distance() >= minDistance);
            }
        }
    }

    @Test
    public void randomGeneratorActuallyCoversAllDirections() {
        int[] sectors = new int[8];
        Random random = new Random(20260924L);

        for (int i = 0; i < 80000; i++) {
            GesturePlan plan = GesturePlan.random(random, 1080, 2400);
            double angle = Math.atan2(plan.endY - plan.startY, plan.endX - plan.startX);
            if (angle < 0) angle += Math.PI * 2.0;
            int sector = Math.min(7, (int) (angle / (Math.PI / 4.0)));
            sectors[sector]++;
        }

        for (int count : sectors) {
            assertTrue("One direction is effectively missing: " + count, count > 3000);
        }
    }

    @Test
    public void foregroundPackageGuardIsStrict() {
        assertTrue(AutoSwipeAccessibilityService.isAllowedPackage("com.instagram.android"));
        assertTrue(AutoSwipeAccessibilityService.isAllowedPackage("com.instagram.lite"));

        assertFalse(AutoSwipeAccessibilityService.isAllowedPackage(null));
        assertFalse(AutoSwipeAccessibilityService.isAllowedPackage(""));
        assertFalse(AutoSwipeAccessibilityService.isAllowedPackage("com.android.settings"));
        assertFalse(AutoSwipeAccessibilityService.isAllowedPackage("com.google.android.apps.nexuslauncher"));
        assertFalse(AutoSwipeAccessibilityService.isAllowedPackage("com.airi.randomreelswiper"));
        assertFalse(AutoSwipeAccessibilityService.isAllowedPackage("com.example.otherapp"));
    }
}
