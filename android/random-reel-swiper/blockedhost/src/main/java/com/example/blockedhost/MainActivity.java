package com.example.blockedhost;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.util.Log;
import android.view.MotionEvent;
import android.widget.Button;

public class MainActivity extends Activity {
    private static final String TAG = "BlockedHost";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        Button dangerSurface = new Button(this);
        dangerSurface.setText("NON-INSTAGRAM APP\nMUST RECEIVE ZERO AUTOMATED TOUCHES");
        dangerSurface.setTextSize(20);
        dangerSurface.setTextColor(Color.WHITE);
        dangerSurface.setBackgroundColor(Color.rgb(80, 20, 20));

        dangerSurface.setOnClickListener(v -> Log.e(TAG, "CLICK_TRIGGERED"));
        dangerSurface.setOnLongClickListener(v -> {
            Log.e(TAG, "LONG_CLICK_TRIGGERED");
            return true;
        });

        setContentView(dangerSurface);
    }

    @Override
    public boolean dispatchTouchEvent(MotionEvent event) {
        Log.e(TAG, "UNEXPECTED_TOUCH action=" + event.getActionMasked()
                + " x=" + event.getX() + " y=" + event.getY());
        return super.dispatchTouchEvent(event);
    }
}
