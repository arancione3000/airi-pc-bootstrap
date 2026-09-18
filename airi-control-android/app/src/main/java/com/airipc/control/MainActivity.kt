package com.airipc.control

import android.Manifest
import android.app.Activity
import android.content.ClipboardManager
import android.content.ClipData
import android.content.Intent
import android.content.pm.PackageManager
import android.media.projection.MediaProjectionManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat

class MainActivity : ComponentActivity() {
    private var accessibilityEnabled by mutableStateOf(false)

    private val projectionLauncher =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK && result.data != null) {
                ContextCompat.startForegroundService(
                    this,
                    PhoneControlService.startIntent(this, result.resultCode, result.data!!),
                )
            } else {
                Toast.makeText(this, "Condivisione schermo annullata", Toast.LENGTH_SHORT).show()
            }
        }

    private val notificationPermission =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        PairingSecret.text(this)
        maybeRequestNotifications()
        setContent {
            val active by PhoneControlService.active.collectAsState()
            MaterialTheme {
                Surface(color = Color(0xFF0C0C0F), modifier = Modifier.fillMaxSize()) {
                    ControlScreen(
                        active = active,
                        accessibilityEnabled = accessibilityEnabled,
                        pairingCode = PairingSecret.text(this),
                        onEnableAccessibility = { startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)) },
                        onStart = { beginControl() },
                        onStop = { startService(PhoneControlService.stopIntent(this)) },
                        onCopyPairing = { copyPairingCode() },
                    )
                }
            }
        }
    }

    override fun onResume() {
        super.onResume()
        accessibilityEnabled = AiriAccessibilityService.connected() || isAccessibilityEnabled()
    }

    private fun beginControl() {
        accessibilityEnabled = AiriAccessibilityService.connected() || isAccessibilityEnabled()
        if (!accessibilityEnabled) {
            Toast.makeText(
                this,
                "Prima abilita Airi Control nelle impostazioni Accessibilità",
                Toast.LENGTH_LONG,
            ).show()
            startActivity(Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS))
            return
        }
        projectionLauncher.launch(
            getSystemService(MediaProjectionManager::class.java).createScreenCaptureIntent()
        )
    }

    private fun maybeRequestNotifications() {
        if (Build.VERSION.SDK_INT >= 33 &&
            checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED
        ) {
            notificationPermission.launch(Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    private fun copyPairingCode() {
        getSystemService(ClipboardManager::class.java).setPrimaryClip(
            ClipData.newPlainText("Airi Control pairing", PairingSecret.text(this))
        )
        Toast.makeText(this, "Codice copiato", Toast.LENGTH_SHORT).show()
    }

    private fun isAccessibilityEnabled(): Boolean {
        val enabled = Settings.Secure.getString(
            contentResolver,
            Settings.Secure.ENABLED_ACCESSIBILITY_SERVICES,
        ).orEmpty()
        val expected = packageName + "/" + AiriAccessibilityService::class.java.name
        val shortExpected = packageName + "/.AiriAccessibilityService"
        return enabled.split(':').any {
            it.equals(expected, ignoreCase = true) || it.equals(shortExpected, ignoreCase = true)
        }
    }
}

@Composable
private fun ControlScreen(
    active: Boolean,
    accessibilityEnabled: Boolean,
    pairingCode: String,
    onEnableAccessibility: () -> Unit,
    onStart: () -> Unit,
    onStop: () -> Unit,
    onCopyPairing: () -> Unit,
) {
    var targetHit by remember { mutableStateOf(false) }
    var testText by remember { mutableStateOf("") }

    Column(
        modifier = Modifier.fillMaxSize().padding(28.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        Text("Airi Control", color = Color.White, fontSize = 34.sp, fontWeight = FontWeight.Bold)
        Text(
            if (active) "CONTROLLO ATTIVO" else "Controllo disattivato",
            color = if (active) Color(0xFF67D69A) else Color(0xFFB7B2BD),
            fontSize = 18.sp,
            fontWeight = FontWeight.SemiBold,
        )

        Surface(
            color = Color(0xFF17171D),
            shape = RoundedCornerShape(22.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("ACCESSIBILITÀ", color = Color(0xFFFF8A00), fontWeight = FontWeight.Bold)
                Text(if (accessibilityEnabled) "Abilitata ✓" else "Da abilitare", color = Color.White)
                if (!accessibilityEnabled) {
                    Button(onClick = onEnableAccessibility) { Text("APRI IMPOSTAZIONI ACCESSIBILITÀ") }
                }
            }
        }

        Surface(
            color = Color(0xFF17171D),
            shape = RoundedCornerShape(22.dp),
            modifier = Modifier.fillMaxWidth(),
        ) {
            Column(modifier = Modifier.padding(20.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
                Text("CODICE DI ABBINAMENTO", color = Color(0xFFFF8A00), fontWeight = FontWeight.Bold)
                Text(pairingCode, color = Color.White, fontSize = 14.sp)
                Text("Tienilo privato. Serve una sola volta per autorizzare Airi-PC.", color = Color(0xFFAAA5B0), fontSize = 13.sp)
                Button(onClick = onCopyPairing) { Text("COPIA CODICE") }
            }
        }

        if (!active) {
            Button(
                onClick = onStart,
                enabled = accessibilityEnabled,
                modifier = Modifier.fillMaxWidth().height(58.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFF8A00)),
            ) {
                Text("AVVIA CONTROLLO", color = Color.Black, fontWeight = FontWeight.Bold)
            }
        } else {
            Button(
                onClick = onStop,
                modifier = Modifier.fillMaxWidth().height(68.dp),
                colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFD83A3A)),
            ) {
                Text("STOP IMMEDIATO", color = Color.White, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            }
        }

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(120.dp)
                .background(
                    if (targetHit) Color(0xFF275F42) else Color(0xFF17171D),
                    RoundedCornerShape(22.dp),
                )
                .clickable { targetHit = !targetHit },
            contentAlignment = Alignment.Center,
        ) {
            Text(
                if (targetHit) "TOCCO RICEVUTO" else "Area test controllo",
                color = if (targetHit) Color(0xFF67D69A) else Color(0xFFB7B2BD),
                fontSize = 18.sp,
                fontWeight = if (targetHit) FontWeight.Bold else FontWeight.Normal,
            )
        }

        OutlinedTextField(
            value = testText,
            onValueChange = { testText = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Campo test testo") },
            singleLine = true,
            textStyle = androidx.compose.ui.text.TextStyle(color = Color.White, fontSize = 16.sp),
            colors = OutlinedTextFieldDefaults.colors(
                focusedTextColor = Color.White,
                unfocusedTextColor = Color.White,
                focusedBorderColor = Color(0xFFFF8A00),
                unfocusedBorderColor = Color(0xFF6F6975),
                focusedLabelColor = Color(0xFFFF8A00),
                unfocusedLabelColor = Color(0xFFAAA5B0),
                cursorColor = Color(0xFFFF8A00),
            ),
        )

        Text(
            "STOP chiude controllo remoto e condivisione schermo. " +
                "L'app non può sbloccare il telefono né scrivere nei campi password.",
            color = Color(0xFF8F8A95),
            fontSize = 13.sp,
        )
    }
}
