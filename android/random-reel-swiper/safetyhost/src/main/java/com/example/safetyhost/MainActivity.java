package com.example.safetyhost;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.widget.Button;

public class MainActivity extends Activity {
    private static final String TAG = "SafetyHost";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Button dangerSurface = new Button(this);
        dangerSurface.setText("FAKE INSTAGRAM\nCLICK / LONG-CLICK SAFETY SURFACE");
        dangerSurface.setTextSize(20);
        dangerSurface.setTextColor(Color.WHITE);
        dangerSurface.setBackgroundColor(Color.rgb(35, 35, 35));

        dangerSurface.setOnClickListener(v -> Log.e(TAG, "CLICK_TRIGGERED"));
        dangerSurface.setOnLongClickListener(v -> {
            Log.e(TAG, "LONG_CLICK_TRIGGERED");
            return true;
        });

        setContentView(dangerSurface);
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent event) {
        String action;
        switch (event.getActionMasked()) {
            case MotionEvent.ACTION_DOWN:
                action = "DOWN";
                break;
            case MotionEvent.ACTION_UP:
                action = "UP";
                break;
            case MotionEvent.ACTION_MOVE:
                action = "MOVE";
                break;
            case MotionEvent.ACTION_CANCEL:
                action = "CANCEL";
                break;
            default:
                action = "OTHER";
        }
        Log.i(TAG, "TOUCH_" + action + " x=" + event.getX() + " y=" + event.getY());
        return super.dispatchTouchEvent(event);
    }
}
