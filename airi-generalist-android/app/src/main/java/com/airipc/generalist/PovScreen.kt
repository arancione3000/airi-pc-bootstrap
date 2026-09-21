package com.airipc.generalist

import android.annotation.SuppressLint
import android.graphics.Color as AndroidColor
import android.webkit.WebSettings
import android.webkit.WebView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView

@SuppressLint("SetJavaScriptEnabled")
@Composable
fun PovScreen(
    state: PovState,
    modifier: Modifier = Modifier,
) {
    Column(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text(
                    "Airi-PC POV live",
                    style = MaterialTheme.typography.titleMedium,
                )
                Text(
                    when {
                        state.screenUrl.isNotBlank() ->
                            "Sessione ${state.sessionId.takeLast(12)} · stream HTTPS cifrato in rendezvous"
                        state.connected ->
                            "Relay collegato · attendo una sessione Airi-PC con desktop attivo"
                        else ->
                            "Collegamento al relay…"
                    },
                    style = MaterialTheme.typography.bodySmall,
                )
                state.error?.let {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        it,
                        color = MaterialTheme.colorScheme.error,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }

        if (state.screenUrl.isBlank()) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
                    .background(Color.Black),
                contentAlignment = Alignment.Center,
            ) {
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    CircularProgressIndicator()
                    Spacer(Modifier.height(12.dp))
                    Text(
                        "In attesa del POV Airi-PC…",
                        color = Color.White,
                    )
                    Text(
                        "Quando Airi-PC apre il desktop virtuale, il video compare qui automaticamente.",
                        color = Color.LightGray,
                        style = MaterialTheme.typography.bodySmall,
                        modifier = Modifier.padding(16.dp),
                    )
                }
            }
        } else {
            key(state.screenUrl) {
                AndroidView(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f),
                    factory = { context ->
                        WebView(context).apply {
                            setBackgroundColor(AndroidColor.BLACK)
                            settings.javaScriptEnabled = true
                            settings.cacheMode = WebSettings.LOAD_NO_CACHE
                            settings.domStorageEnabled = false
                            settings.allowFileAccess = false
                            settings.allowContentAccess = false
                            settings.mixedContentMode =
                                WebSettings.MIXED_CONTENT_NEVER_ALLOW
                            loadUrl(state.screenUrl)
                        }
                    },
                    update = { web ->
                        if (web.url != state.screenUrl) {
                            web.loadUrl(state.screenUrl)
                        }
                    },
                )
            }
        }
    }
}
