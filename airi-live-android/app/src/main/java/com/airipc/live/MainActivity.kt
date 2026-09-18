package com.airipc.live

import android.os.Bundle
import android.graphics.BitmapFactory
import android.os.SystemClock
import android.net.Uri
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.foundation.background
import androidx.compose.foundation.Image
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.semantics.contentDescription
import androidx.compose.ui.semantics.semantics
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.*
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.MessageDigest
import java.security.spec.MGF1ParameterSpec
import javax.crypto.Cipher
import javax.crypto.spec.OAEPParameterSpec
import javax.crypto.spec.PSource
import java.util.*
import java.util.concurrent.TimeUnit

private const val REPO = "arancione3000/airi-pc-bootstrap"
private const val GITHUB_BRANCH = "https://api.github.com/repos/$REPO/branches/main"
private const val FALLBACK_RELAY = "https://ntfy.sh"
private const val FALLBACK_TOPIC = "airi-live-09d926b9a4207c660a5f3efa5f50505ef5ff46e8463b3beb"
private const val LIVE_CONFIG_URL = "https://raw.githubusercontent.com/$REPO/main/.ai/airi_live.json"

data class LiveConfig(
    val relayBase: String = FALLBACK_RELAY,
    val topic: String = FALLBACK_TOPIC,
    val history: String = "24h",
)

data class LiveEvent(
    val id: String,
    val ts: Long,
    val kind: String,
    val title: String,
    val detail: String,
    val status: String,
    val taskId: String? = null,
    val nodeId: String? = null,
    val sourceSha: String? = null,
    val sessionId: String? = null,
)

data class UiState(
    val connecting: Boolean = true,
    val connected: Boolean = false,
    val sourceSha: String = "",
    val topic: String = "",
    val events: List<LiveEvent> = emptyList(),
    val error: String? = null,
    val lastSeenMs: Long = 0,
    val sessionId: String = "",
    val sessionWatermarkMs: Long = 0,
    val screenUrl: String = "",
    val screenSessionId: String = "",
    val screenOfferTs: Long = 0,
)

data class ViewerIdentity(val keyId: String, val publicKeyB64: String)

private const val VIEWER_KEY_ALIAS = "airi_live_screen_viewer_v1"

private fun ensureViewerIdentity(): ViewerIdentity {
    val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
    if (!store.containsAlias(VIEWER_KEY_ALIAS)) {
        val generator = KeyPairGenerator.getInstance(KeyProperties.KEY_ALGORITHM_RSA, "AndroidKeyStore")
        generator.initialize(
            KeyGenParameterSpec.Builder(VIEWER_KEY_ALIAS, KeyProperties.PURPOSE_DECRYPT)
                .setKeySize(3072)
                .setDigests(KeyProperties.DIGEST_SHA256, KeyProperties.DIGEST_SHA512)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_OAEP)
                .build()
        )
        generator.generateKeyPair()
    }
    val publicBytes = store.getCertificate(VIEWER_KEY_ALIAS).publicKey.encoded
    val digest = MessageDigest.getInstance("SHA-256").digest(publicBytes)
    val keyId = digest.take(12).joinToString("") { "%02x".format(it) }
    return ViewerIdentity(keyId, Base64.encodeToString(publicBytes, Base64.NO_WRAP))
}

private fun decryptScreenOffer(ciphertextB64: String): JSONObject {
    val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
    val privateKey = store.getKey(VIEWER_KEY_ALIAS, null)
    val cipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
    val spec = OAEPParameterSpec("SHA-256", "MGF1", MGF1ParameterSpec.SHA1, PSource.PSpecified.DEFAULT)
    cipher.init(Cipher.DECRYPT_MODE, privateKey, spec)
    val plain = cipher.doFinal(Base64.decode(ciphertextB64, Base64.NO_WRAP))
    return JSONObject(String(plain, Charsets.UTF_8))
}

class AiriLiveClient(private val configOverride: LiveConfig? = null) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    private var streamCall: Call? = null
    private var scope: CoroutineScope? = null
    private var viewerKeyJob: Job? = null
    private val viewerIdentity by lazy { ensureViewerIdentity() }

    fun start(onState: (UiState) -> Unit) {
        stop()
        scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
        scope!!.launch {
            var state = UiState()
            onState(state)
            while (isActive) {
                try {
                    val config = fetchLiveConfig()
                    Log.i(
                        "AiriLivePOV",
                        "client config override=${configOverride != null} relayHost=${runCatching { Uri.parse(config.relayBase).host }.getOrNull()} topic=${config.topic.take(24)}",
                    )
                    val sha = runCatching { fetchMainSha() }.getOrDefault("")
                    publishViewerKey(config)
                    if (viewerKeyJob?.isActive != true) {
                        viewerKeyJob = scope?.launch {
                            while (isActive) {
                                delay(30 * 60 * 1000L)
                                runCatching { publishViewerKey(config) }
                            }
                        }
                    }
                    state = state.copy(sourceSha = sha, topic = config.topic, error = null)
                    onState(state)
                    state = stream(config, state, onState)
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    state = state.copy(connecting = false, connected = false, error = e.message ?: "Connessione interrotta")
                    onState(state)
                    delay(4_000)
                }
            }
        }
    }

    fun stop() {
        streamCall?.cancel()
        streamCall = null
        viewerKeyJob?.cancel()
        viewerKeyJob = null
        scope?.cancel()
        scope = null
    }

    private fun fetchMainSha(): String {
        val req = Request.Builder()
            .url(GITHUB_BRANCH)
            .header("Accept", "application/vnd.github+json")
            .header("User-Agent", "Airi-Live-Android/1.4")
            .build()
        client.newCall(req).execute().use { r ->
            if (!r.isSuccessful) error("GitHub ${r.code}")
            val obj = JSONObject(r.body?.string().orEmpty())
            return obj.getJSONObject("commit").getString("sha")
        }
    }

    private fun fetchLiveConfig(): LiveConfig {
        configOverride?.let { return it }
        return runCatching {
            val req = Request.Builder()
                .url(LIVE_CONFIG_URL)
                .header("User-Agent", "Airi-Live-Android/1.4")
                .build()
            client.newCall(req).execute().use { r ->
                if (!r.isSuccessful) error("Airi Live config ${r.code}")
                val obj = JSONObject(r.body?.string().orEmpty())
                LiveConfig(
                    relayBase = obj.optString("relay_base", FALLBACK_RELAY).trimEnd('/'),
                    topic = obj.optString("topic", FALLBACK_TOPIC),
                    history = obj.optString("history", "24h"),
                )
            }
        }.getOrElse { LiveConfig() }
    }

    private fun publishViewerKey(config: LiveConfig) {
        val payload = JSONObject()
            .put("kind", "viewer_key")
            .put("key_id", viewerIdentity.keyId)
            .put("public_key_b64", viewerIdentity.publicKeyB64)
            .put("ts", System.currentTimeMillis() / 1000)
            .put("protocol", 3)
            .toString()
        val req = Request.Builder()
            .url("${config.relayBase}/${config.topic}")
            .header("User-Agent", "Airi-Live-Android/1.4")
            .post(payload.toRequestBody("text/plain; charset=utf-8".toMediaType()))
            .build()
        client.newCall(req).execute().use { r ->
            if (!r.isSuccessful) error("viewer key publish ${r.code}")
            Log.i("AiriLivePOV", "viewer key published")
        }
    }

    private fun stream(config: LiveConfig, initial: UiState, onState: (UiState) -> Unit): UiState {
        val topic = config.topic
        var state = initial.copy(connecting = true, connected = false)
        onState(state)
        val req = Request.Builder()
            .url("${config.relayBase}/$topic/json?since=${config.history}")
            .header("User-Agent", "Airi-Live-Android/1.4")
            .build()
        val call = client.newCall(req)
        streamCall = call
        call.execute().use { response ->
            if (!response.isSuccessful) error("ntfy ${response.code}")
            state = state.copy(connecting = false, connected = true, error = null)
            onState(state)
            val source = response.body?.source() ?: error("Stream vuoto")
            val seen = state.events.mapTo(mutableSetOf()) { it.id }
            while (true) {
                val line = source.readUtf8Line() ?: break
                if (line.isBlank()) continue
                val outer = runCatching { JSONObject(line) }.getOrNull() ?: continue
                val relayEvent = outer.optString("event")
                if (relayEvent != "message") {
                    Log.d("AiriLivePOV", "relay event=$relayEvent")
                    continue
                }
                val ntfyId = outer.optString("id", UUID.randomUUID().toString())
                val payload = outer.optString("message")
                val control = runCatching { JSONObject(payload) }.getOrNull()
                Log.d("AiriLivePOV", "relay message kind=${control?.optString("kind").orEmpty()}")
                if (control?.optString("kind") == "viewer_key") continue
                if (control?.optString("kind") == "screen_offer") {
                    if (control.optString("key_id") == viewerIdentity.keyId) {
                        Log.i("AiriLivePOV", "screen offer received")
                        val offerTs = control.optLong("ts", 0L) * 1000
                        if (offerTs >= state.screenOfferTs) {
                            runCatching {
                                val descriptor = decryptScreenOffer(control.getString("ciphertext_b64"))
                                state = state.copy(
                                    screenUrl = descriptor.getString("u"),
                                    screenSessionId = descriptor.optString("s"),
                                    screenOfferTs = offerTs,
                                    lastSeenMs = System.currentTimeMillis(),
                                )
                                Log.i("AiriLivePOV", "screen offer decrypted")
                                onState(state)
                            }.onFailure {
                                Log.e("AiriLivePOV", "screen offer decrypt failed", it)
                            }
                        }
                    }
                    continue
                }
                val parsed = parsePayload(payload, ntfyId)
                if (parsed.isEmpty()) continue
                val unique = parsed.filter { seen.add(it.id) }
                if (unique.isEmpty()) continue
                val candidate = unique
                    .filter { !it.sessionId.isNullOrBlank() }
                    .maxByOrNull { it.ts }
                var activeSession = state.sessionId
                var watermark = state.sessionWatermarkMs
                if (candidate != null && (activeSession.isBlank() || candidate.ts >= watermark)) {
                    activeSession = candidate.sessionId.orEmpty()
                    watermark = candidate.ts
                }
                val merged = (unique + state.events).sortedByDescending { it.ts }.take(500)
                val visible = if (activeSession.isNotBlank()) {
                    merged.filter { it.sessionId == activeSession }.take(250)
                } else {
                    merged.take(250)
                }
                val newestSha = visible.firstNotNullOfOrNull { it.sourceSha }
                val keepScreen = state.screenSessionId.isBlank() || activeSession.isBlank() || state.screenSessionId == activeSession
                state = state.copy(
                    events = visible,
                    sourceSha = newestSha ?: state.sourceSha,
                    sessionId = activeSession,
                    sessionWatermarkMs = watermark,
                    screenUrl = if (keepScreen) state.screenUrl else "",
                    screenSessionId = if (keepScreen) state.screenSessionId else "",
                    lastSeenMs = System.currentTimeMillis(),
                    connected = true,
                    error = null,
                )
                onState(state)
            }
        }
        Log.w("AiriLivePOV", "relay stream ended; reconnecting")
        return state.copy(connected = false)
    }

    private fun parsePayload(raw: String, ntfyId: String): List<LiveEvent> {
        val trimmed = raw.trim()
        return try {
            if (trimmed.startsWith("[")) {
                val a = JSONArray(trimmed)
                buildList { for (i in 0 until a.length()) parseEvent(a.getJSONObject(i), "$ntfyId-$i")?.let(::add) }
            } else {
                listOfNotNull(parseEvent(JSONObject(trimmed), ntfyId))
            }
        } catch (_: Exception) { emptyList() }
    }

    private fun parseEvent(o: JSONObject, fallbackId: String): LiveEvent? {
        if (o.optInt("v", 0) != 1) return null
        return LiveEvent(
            id = o.optString("id", fallbackId),
            ts = o.optLong("ts", System.currentTimeMillis() / 1000) * 1000,
            kind = o.optString("kind", "event"),
            title = o.optString("title", "Airi-PC"),
            detail = o.optString("detail", ""),
            status = o.optString("status", "info"),
            taskId = o.optString("task_id").ifBlank { null },
            nodeId = o.optString("node_id").ifBlank { null },
            sourceSha = o.optString("source_sha").ifBlank { null },
            sessionId = o.optString("session_id").ifBlank { null },
        )
    }
}

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val relay = intent.getStringExtra("airi_relay")?.trim()?.trimEnd('/').orEmpty()
        val topic = intent.getStringExtra("airi_topic")?.trim().orEmpty()
        val override = if (relay.isNotBlank() && topic.isNotBlank()) {
            LiveConfig(relayBase = relay, topic = topic, history = "24h")
        } else null
        Log.i(
            "AiriLivePOV",
            "activity config override=${override != null} relayHost=${runCatching { Uri.parse(override?.relayBase ?: FALLBACK_RELAY).host }.getOrNull()} topic=${(override?.topic ?: FALLBACK_TOPIC).take(24)}",
        )
        setContent { AiriLiveApp(override) }
    }
}

private val Orange = Color(0xFFFF8A00)
private val Bg = Color(0xFF0C0C0F)
private val Card = Color(0xFF17171C)
private val Muted = Color(0xFFA8A8B3)
private val Good = Color(0xFF5ED390)
private val Bad = Color(0xFFFF5D6C)

@Composable
fun AiriLiveApp(configOverride: LiveConfig? = null) {
    val client = remember(configOverride) { AiriLiveClient(configOverride) }
    var state by remember { mutableStateOf(UiState()) }
    var filter by remember { mutableStateOf("all") }
    DisposableEffect(Unit) {
        client.start { s -> state = s }
        onDispose { client.stop() }
    }

    MaterialTheme(
        colorScheme = darkColorScheme(primary = Orange, background = Bg, surface = Card, onBackground = Color.White, onSurface = Color.White)
    ) {
        var fullScreen by remember { mutableStateOf(false) }
        Surface(modifier = Modifier.fillMaxSize(), color = Bg) {
            if (fullScreen && state.screenUrl.isNotBlank()) {
                FullScreenPov(state.screenUrl) { fullScreen = false }
            } else {
                Column(Modifier.fillMaxSize().padding(horizontal = 18.dp)) {
                    Spacer(Modifier.height(20.dp))
                    Header(state)
                    Spacer(Modifier.height(18.dp))
                    if (state.screenUrl.isNotBlank()) {
                        LivePovCard(state) { fullScreen = true }
                    } else {
                        CurrentCard(state)
                    }
                    Spacer(Modifier.height(14.dp))
                    Metrics(state)
                    Spacer(Modifier.height(16.dp))
                    FilterBar(filter) { filter = it }
                    Spacer(Modifier.height(10.dp))
                    val shown = state.events.filter { filter == "all" || it.kind == filter }
                    if (shown.isEmpty()) EmptyState(state, Modifier.weight(1f)) else Timeline(shown, Modifier.weight(1f))
                    Spacer(Modifier.height(10.dp))
                    Footer(state)
                    Spacer(Modifier.height(12.dp))
                }
            }
        }
    }
}

private fun povFrameUrl(viewUrl: String): String {
    val uri = Uri.parse(viewUrl)
    val token = uri.lastPathSegment?.takeIf { it.isNotBlank() }
        ?: error("Invalid Airi POV viewer URL")
    return uri.buildUpon()
        .path("/frame/" + token + ".jpg")
        .clearQuery()
        .fragment(null)
        .build()
        .toString()
}

@Composable
private fun PovFrameView(url: String, modifier: Modifier = Modifier) {
    val frameUrl = remember(url) { povFrameUrl(url) }
    val frameClient = remember {
        OkHttpClient.Builder()
            .connectTimeout(12, TimeUnit.SECONDS)
            .readTimeout(12, TimeUnit.SECONDS)
            .callTimeout(15, TimeUnit.SECONDS)
            .build()
    }
    var image by remember(url) { mutableStateOf<androidx.compose.ui.graphics.ImageBitmap?>(null) }
    var fps by remember(url) { mutableStateOf(0.0) }
    var status by remember(url) { mutableStateOf("connessione…") }

    LaunchedEffect(frameUrl) {
        var frames = 0
        var windowStart = SystemClock.elapsedRealtime()
        while (isActive) {
            try {
                val bitmap = withContext(Dispatchers.IO) {
                    val request = Request.Builder()
                        .url(frameUrl + "?t=" + System.nanoTime())
                        .header("Cache-Control", "no-cache, no-store")
                        .header("Pragma", "no-cache")
                        .header("User-Agent", "Airi-Live-Android/1.4-native")
                        .build()
                    frameClient.newCall(request).execute().use { response ->
                        if (!response.isSuccessful) error("POV frame HTTP " + response.code)
                        val bytes = response.body?.bytes() ?: error("Empty POV frame")
                        BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                            ?: error("Invalid POV JPEG")
                    }
                }
                image = bitmap.asImageBitmap()
                frames += 1
                status = "live"
                val now = SystemClock.elapsedRealtime()
                val elapsed = now - windowStart
                if (elapsed >= 1000) {
                    fps = frames * 1000.0 / elapsed
                    frames = 0
                    windowStart = now
                }
                delay(8)
            } catch (e: CancellationException) {
                throw e
            } catch (e: Exception) {
                Log.w("AiriLivePOV", "frame fetch failed: " + e.javaClass.simpleName + ": " + (e.message ?: ""), e)
                status = "riconnessione…"
                delay(180)
            }
        }
    }

    Box(modifier = modifier.background(Color.Black), contentAlignment = Alignment.Center) {
        val current = image
        if (current != null) {
            Image(
                bitmap = current,
                contentDescription = "Airi-PC live desktop",
                modifier = Modifier.fillMaxSize(),
                contentScale = ContentScale.Fit,
            )
        } else {
            CircularProgressIndicator(color = Good, strokeWidth = 2.dp)
        }
        Surface(
            color = Color(0xCC000000),
            shape = RoundedCornerShape(10.dp),
            modifier = Modifier.align(Alignment.TopStart).padding(8.dp),
        ) {
            val label = if (fps > 0.0) {
                "AIRI-PC · LIVE POV · " + "%.1f".format(Locale.US, fps) + " FPS"
            } else {
                "AIRI-PC · POV · " + status
            }
            Text(
                label,
                color = Good,
                fontWeight = FontWeight.Bold,
                fontSize = 11.sp,
                modifier = Modifier.padding(horizontal = 8.dp, vertical = 5.dp),
            )
        }
    }
}

@Composable
private fun LivePovCard(state: UiState, onFullScreen: () -> Unit) {
    Card(colors = CardDefaults.cardColors(containerColor = Card), shape = RoundedCornerShape(22.dp), modifier = Modifier.fillMaxWidth()) {
        Column {
            Row(Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 11.dp), verticalAlignment = Alignment.CenterVertically) {
                Box(Modifier.size(8.dp).background(Good, CircleShape))
                Spacer(Modifier.width(8.dp))
                Text("POV Airi-PC · LIVE", color = Good, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                Spacer(Modifier.weight(1f))
                Text(state.screenSessionId.takeLast(10), color = Muted, fontSize = 11.sp)
            }
            PovFrameView(
                state.screenUrl,
                Modifier.fillMaxWidth().aspectRatio(1.6f).background(Color.Black)
            )
            TextButton(
                onClick = onFullScreen,
                modifier = Modifier
                    .align(Alignment.End)
                    .padding(end = 8.dp)
                    .semantics { contentDescription = "Airi POV fullscreen" },
            ) {
                Text("SCHERMO INTERO")
            }
        }
    }
}

@Composable
private fun FullScreenPov(url: String, onClose: () -> Unit) {
    Box(Modifier.fillMaxSize().background(Color.Black)) {
        PovFrameView(url, Modifier.fillMaxSize())
        FilledTonalButton(
            onClick = onClose,
            modifier = Modifier.align(Alignment.TopEnd).padding(16.dp)
        ) { Text("×  CHIUDI") }
    }
}

@Composable
private fun Header(state: UiState) {
    Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.fillMaxWidth()) {
        Box(Modifier.size(42.dp).background(Orange, RoundedCornerShape(14.dp)), contentAlignment = Alignment.Center) {
            Text("A", color = Bg, fontWeight = FontWeight.Black, fontSize = 24.sp)
        }
        Spacer(Modifier.width(12.dp))
        Column(Modifier.weight(1f)) {
            Text("Airi Live", fontSize = 26.sp, fontWeight = FontWeight.Black)
            Text("Airi-PC activity monitor", color = Muted, fontSize = 13.sp)
        }
        val c = if (state.connected) Good else if (state.connecting) Orange else Bad
        Box(Modifier.size(10.dp).background(c, CircleShape))
    }
}

@Composable
private fun CurrentCard(state: UiState) {
    val current = state.events.firstOrNull { it.status in setOf("running", "retrying", "verifying") }
        ?: state.events.firstOrNull()
    Card(colors = CardDefaults.cardColors(containerColor = Card), shape = RoundedCornerShape(22.dp), modifier = Modifier.fillMaxWidth()) {
        Column(Modifier.padding(18.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text("ADESSO", color = Orange, fontWeight = FontWeight.Bold, fontSize = 12.sp)
                Spacer(Modifier.weight(1f))
                val marker = when {
                    state.sessionId.isNotBlank() -> state.sessionId.takeLast(12)
                    state.sourceSha.isNotBlank() -> state.sourceSha.take(8)
                    else -> "—"
                }
                Text(marker, color = Muted, fontSize = 12.sp)
            }
            Spacer(Modifier.height(10.dp))
            Text(
                current?.title ?: if (state.connected) "Canale pronto · attendo una sessione Airi-PC…" else "Connessione al relay…",
                fontWeight = FontWeight.Bold,
                fontSize = 20.sp,
            )
            if (!current?.detail.isNullOrBlank()) {
                Spacer(Modifier.height(5.dp))
                Text(current!!.detail, color = Muted, maxLines = 2, overflow = TextOverflow.Ellipsis)
            }
            Spacer(Modifier.height(14.dp))
            LinearProgressIndicator(
                progress = { if (current?.status == "running") 0.58f else if (current?.status == "completed") 1f else 0.16f },
                modifier = Modifier.fillMaxWidth().height(7.dp),
                color = Orange,
                trackColor = Color(0xFF2B2B32),
            )
        }
    }
}

@Composable
private fun Metrics(state: UiState) {
    val errors = state.events.count { it.status in setOf("failed", "error", "blocked") }
    val tasks = state.events.count { it.kind == "task" }
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        Metric("EVENTI", state.events.size.toString(), Modifier.weight(1f))
        Metric("TASK", tasks.toString(), Modifier.weight(1f))
        Metric("ERRORI", errors.toString(), Modifier.weight(1f), if (errors > 0) Bad else Good)
    }
}

@Composable
private fun Metric(label: String, value: String, modifier: Modifier, accent: Color = Color.White) {
    Card(colors = CardDefaults.cardColors(containerColor = Card), shape = RoundedCornerShape(18.dp), modifier = modifier) {
        Column(Modifier.padding(13.dp)) {
            Text(label, color = Muted, fontSize = 10.sp, fontWeight = FontWeight.Bold)
            Text(value, color = accent, fontSize = 22.sp, fontWeight = FontWeight.Black)
        }
    }
}

@Composable
private fun FilterBar(selected: String, onChange: (String) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        listOf("all" to "Tutto", "task" to "Task", "job" to "Job", "runtime" to "Runtime").forEach { (k, label) ->
            FilterChip(selected = selected == k, onClick = { onChange(k) }, label = { Text(label) })
        }
    }
}

@Composable
private fun Timeline(events: List<LiveEvent>, modifier: Modifier = Modifier) {
    LazyColumn(modifier, verticalArrangement = Arrangement.spacedBy(8.dp), contentPadding = PaddingValues(bottom = 8.dp)) {
        items(events, key = { it.id }) { e -> EventRow(e) }
    }
}

@Composable
private fun EventRow(e: LiveEvent) {
    val accent = when (e.status) {
        "completed", "success" -> Good
        "failed", "error", "blocked" -> Bad
        "running", "retrying", "verifying" -> Orange
        else -> Muted
    }
    Card(colors = CardDefaults.cardColors(containerColor = Card), shape = RoundedCornerShape(16.dp), modifier = Modifier.fillMaxWidth()) {
        Row(Modifier.padding(14.dp), verticalAlignment = Alignment.Top) {
            Box(Modifier.padding(top = 5.dp).size(9.dp).background(accent, CircleShape))
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Row {
                    Text(e.title, fontWeight = FontWeight.SemiBold, modifier = Modifier.weight(1f))
                    Text(time(e.ts), color = Muted, fontSize = 11.sp)
                }
                AnimatedVisibility(e.detail.isNotBlank()) {
                    Text(e.detail, color = Muted, fontSize = 13.sp, maxLines = 3, overflow = TextOverflow.Ellipsis)
                }
                if (!e.nodeId.isNullOrBlank()) Text("step ${e.nodeId}", color = accent, fontSize = 11.sp)
            }
        }
    }
}

@Composable
private fun EmptyState(state: UiState, modifier: Modifier = Modifier) {
    Box(modifier.fillMaxWidth(), contentAlignment = Alignment.Center) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            Text("◉", color = Orange, fontSize = 42.sp)
            Spacer(Modifier.height(8.dp))
            Text(if (state.connected) "Canale Airi Live collegato" else "Collegamento al relay…", fontWeight = FontWeight.Bold)
            Text(
                state.error ?: if (state.connected) "Ogni ricostruzione di Airi-PC viene riconosciuta automaticamente." else "Riconnessione automatica.",
                color = Muted,
                fontSize = 13.sp,
            )
        }
    }
}

@Composable
private fun Footer(state: UiState) {
    val text = when {
        state.screenUrl.isNotBlank() -> "POV LIVE · sessione ${state.screenSessionId.takeLast(10)}"
        state.connected && state.lastSeenMs > 0 && state.sessionId.isNotBlank() -> "LIVE · sessione ${state.sessionId.takeLast(10)}"
        state.connected && state.lastSeenMs > 0 -> "LIVE · eventi Airi-PC ricevuti"
        state.connected -> "RELAY ONLINE · in attesa di una sessione Airi-PC"
        state.error != null -> "Riconnessione automatica"
        else -> "Connessione automatica"
    }
    Text(text, color = if (state.connected) Good else Muted, fontSize = 11.sp, modifier = Modifier.fillMaxWidth())
}

private fun time(ms: Long): String = SimpleDateFormat("HH:mm:ss", Locale.getDefault()).format(Date(ms))
