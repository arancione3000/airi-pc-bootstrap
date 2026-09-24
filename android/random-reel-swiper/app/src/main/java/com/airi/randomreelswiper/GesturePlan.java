package com.airi.randomreelswiper;

import java.util.Random;

final class GesturePlan {
    final float startX;
    final float startY;
    final float midX;
    final float midY;
    final float endX;
    final float endY;
    final long durationMs;

    GesturePlan(float startX, float startY, float midX, float midY,
                float endX, float endY, long durationMs) {
        this.startX = startX;
        this.startY = startY;
        this.midX = midX;
        this.midY = midY;
        this.endX = endX;
        this.endY = endY;
        this.durationMs = durationMs;
    }

    static GesturePlan random(Random random, int width, int height) {
        float minX = width * 0.12f;
        float maxX = width * 0.88f;
        float minY = height * 0.24f;
        float maxY = height * 0.80f;

        float startX = range(random, width * 0.30f, width * 0.70f);
        float startY = range(random, height * 0.36f, height * 0.68f);

        float minDimension = Math.min(width, height);
        float endX = startX;
        float endY = startY;
        double chosenAngle = 0.0;
        boolean found = false;

        // Rejection sampling keeps the full stroke inside the safe rectangle
        // without shortening it into a near-tap.
        for (int attempt = 0; attempt < 64; attempt++) {
            double angle = random.nextDouble() * Math.PI * 2.0;
            float distance = range(random, minDimension * 0.35f, minDimension * 0.52f);
            float candidateX = startX + (float) Math.cos(angle) * distance;
            float candidateY = startY + (float) Math.sin(angle) * distance;

            if (candidateX >= minX && candidateX <= maxX &&
                    candidateY >= minY && candidateY <= maxY) {
                endX = candidateX;
                endY = candidateY;
                chosenAngle = angle;
                found = true;
                break;
            }
        }

        if (!found) {
            // Extremely unlikely fallback: choose a long cardinal drag, never a tap.
            int direction = random.nextInt(4);
            float distance = minDimension * 0.35f;
            switch (direction) {
                case 0:
                    endX = clamp(startX + distance, minX, maxX);
                    chosenAngle = 0.0;
                    break;
                case 1:
                    endY = clamp(startY + distance, minY, maxY);
                    chosenAngle = Math.PI / 2.0;
                    break;
                case 2:
                    endX = clamp(startX - distance, minX, maxX);
                    chosenAngle = Math.PI;
                    break;
                default:
                    endY = clamp(startY - distance, minY, maxY);
                    chosenAngle = Math.PI * 1.5;
                    break;
            }
        }

        // Bow the line slightly perpendicular to its main direction.
        float bend = range(random, -minDimension * 0.06f, minDimension * 0.06f);
        float perpendicularX = (float) -Math.sin(chosenAngle);
        float perpendicularY = (float) Math.cos(chosenAngle);

        float midX = clamp((startX + endX) / 2f + perpendicularX * bend, minX, maxX);
        float midY = clamp((startY + endY) / 2f + perpendicularY * bend, minY, maxY);

        long durationMs = 2_000L + random.nextInt(2_001);
        return new GesturePlan(startX, startY, midX, midY, endX, endY, durationMs);
    }

    float distance() {
        return (float) Math.hypot(endX - startX, endY - startY);
    }

    private static float range(Random random, float min, float max) {
        return min + random.nextFloat() * (max - min);
    }

    private static float clamp(float value, float min, float max) {
        return Math.max(min, Math.min(max, value));
    }
}
