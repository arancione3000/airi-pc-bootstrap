package com.airi.randomreelswiper;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.os.Bundle;
import android.provider.Settings;
import android.view.Gravity;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.TextView;

public class MainActivity extends Activity {
    private TextView status;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setGravity(Gravity.CENTER_HORIZONTAL);
        root.setPadding(dp(24), dp(48), dp(24), dp(24));
        root.setBackgroundColor(Color.rgb(20, 20, 20));

        TextView title = new TextView(this);
        title.setText("Random Reel Swiper");
        title.setTextColor(Color.WHITE);
        title.setTextSize(28);
        title.setGravity(Gravity.CENTER);
        root.addView(title, new LinearLayout.LayoutParams(-1, -2));

        TextView info = new TextView(this);
        info.setText("Swipe casuali ogni 10–60 secondi, concentrati nella zona centrale dello schermo.\n\n1) Abilita il servizio Accessibilità\n2) Torna qui e premi START\n3) Apri Instagram/Reels\n4) Usa il pulsante STOP flottante per fermare tutto");
        info.setTextColor(Color.LTGRAY);
        info.setTextSize(16);
        info.setPadding(0, dp(24), 0, dp(24));
        root.addView(info, new LinearLayout.LayoutParams(-1, -2));

        status = new TextView(this);
        status.setTextColor(Color.WHITE);
        status.setTextSize(16);
        status.setGravity(Gravity.CENTER);
        root.addView(status, new LinearLayout.LayoutParams(-1, -2));

        Button enable = new Button(this);
        enable.setText("ABILITA ACCESSIBILITÀ");
        enable.setOnClickListener(v -> startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)));
        LinearLayout.LayoutParams buttonParams = new LinearLayout.LayoutParams(-1, dp(58));
        buttonParams.setMargins(0, dp(18), 0, dp(8));
        root.addView(enable, buttonParams);

        Button start = new Button(this);
        start.setText("START");
        start.setOnClickListener(v -> {
            AutoSwipeAccessibilityService service = AutoSwipeAccessibilityService.getInstance();
            if (service == null) {
                status.setText("Servizio non attivo: abilitalo nelle impostazioni Accessibilità.");
                return;
            }
            service.startAutomation();
            status.setText("ATTIVO ✓  Apri Reels. STOP resta visibile in alto a destra.");
        });
        root.addView(start, buttonParams);

        Button stop = new Button(this);
        stop.setText("STOP");
        stop.setOnClickListener(v -> {
            AutoSwipeAccessibilityService service = AutoSwipeAccessibilityService.getInstance();
            if (service != null) service.stopAutomation();
            status.setText("Fermo.");
        });
        root.addView(stop, buttonParams);

        setContentView(root);
    }

    @Override
    protected void onResume() {
        super.onResume();
        AutoSwipeAccessibilityService service = AutoSwipeAccessibilityService.getInstance();
        status.setText(service == null ? "Accessibilità: non attiva" : (service.isRunning() ? "ATTIVO ✓" : "Accessibilità: attiva, automazione ferma"));
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
