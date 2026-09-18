package com.airipc.control

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

class StopControlReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent?) {
        ControlSessionManager.stop(context)
    }
}
