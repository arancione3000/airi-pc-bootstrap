package com.airipc.control

import android.content.Context
import android.util.Base64
import android.util.Log
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import okhttp3.Call
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.security.SecureRandom
import java.util.UUID
import java.util.concurrent.TimeUnit

private const val CONTROL_REPO = "arancione3000/airi-pc-bootstrap"
private const val CONTROL_CONFIG_URL =
    "https://raw.githubusercontent.com/$CONTROL_REPO/main/.ai/airi_control.json"
private const val FALLBACK_RELAY = "https://ntfy.sh"
private const val FALLBACK_BOOTSTRAP =
    "airi-control-bootstrap-2e6a6f6c97314f2ba8d6c41e2fa1e4d2"

data class ControlConfig(
    val relayBase: String = FALLBACK_RELAY,
    val bootstrapTopic: String = FALLBACK_BOOTSTRAP,
)

data class SessionController(
    val keyId: String,
    val publicKeyB64: String,
    val pairedAtMs: Long,
)

data class PhoneCommand(
    val id: String,
    val op: String,
    val args: JSONObject,
    val controller: SessionController,
)

class ControlClient(
    private val context: Context,
    private val scope: CoroutineScope,
    private val onCommand: suspend (PhoneCommand) -> Unit,
    private val onControllerState: (Boolean) -> Unit,
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    private val identity by lazy { CryptoBox.ensureIdentity() }
    private val seenCommands = LinkedHashMap<String, Long>()
    private var streamCall: Call? = null
    private var streamJob: Job? = null
    private var announceJob: Job? = null
    @Volatile private var config = ControlConfig()
    @Volatile private var controller: SessionController? = null
    @Volatile private var lastControllerCommandAt = 0L

    val sessionId: String = UUID.randomUUID().toString()
    val sessionTopic: String = "airi-control-session-" + randomToken(24)

    fun start() {
        stop()
        config = fetchConfig()
        streamJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                try {
                    stream()
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.w("AiriControl", "control stream reconnecting", e)
                    delay(1200)
                }
            }
        }
        announceJob = scope.launch(Dispatchers.IO) {
            while (isActive) {
                runCatching { publishReady() }
                delay(4000)
            }
        }
    }

    fun stop() {
        streamCall?.cancel()
        streamCall = null
        streamJob?.cancel()
        announceJob?.cancel()
        streamJob = null
        announceJob = null
        controller = null
        onControllerState(false)
    }

    fun connected(): Boolean = controller != null

    private fun fetchConfig(): ControlConfig {
        val relayOverride = BuildConfig.CONTROL_RELAY_OVERRIDE.trim()
        val topicOverride = BuildConfig.CONTROL_TOPIC_OVERRIDE.trim()
        if (relayOverride.isNotBlank() && topicOverride.isNotBlank()) {
            return ControlConfig(relayOverride.trimEnd('/'), topicOverride)
        }
        return runCatching {
            val request = Request.Builder()
                .url(CONTROL_CONFIG_URL)
                .header("User-Agent", "Airi-Control-Android/0.2")
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) error("config HTTP ${response.code}")
                val obj = JSONObject(response.body?.string().orEmpty())
                ControlConfig(
                    relayBase = obj.optString("relay_base", FALLBACK_RELAY).trimEnd('/'),
                    bootstrapTopic = obj.optString(
                        "bootstrap_topic",
                        obj.optString("topic", FALLBACK_BOOTSTRAP),
                    ),
                )
            }
        }.getOrElse { ControlConfig() }
    }

    private fun publishReady() {
        val now = System.currentTimeMillis() / 1000
        val expires = now + 20
        val canonical = canonical(
            "phone_ready",
            sessionId,
            identity.keyId,
            identity.encryptionPublicKeyB64,
            identity.signingPublicKeyB64,
            sessionTopic,
            now.toString(),
            expires.toString(),
        )
        val payload = JSONObject()
            .put("kind", "phone_ready")
            .put("session_id", sessionId)
            .put("device_key_id", identity.keyId)
            .put("encryption_public_key_b64", identity.encryptionPublicKeyB64)
            .put("signing_public_key_b64", identity.signingPublicKeyB64)
            .put("session_topic", sessionTopic)
            .put("ts", now)
            .put("expires_at", expires)
            .put("device_signature_b64", CryptoBox.signDevice(canonical))
            .put("protocol", 2)
        publishJson(config.bootstrapTopic, payload)
    }

    private fun stream() {
        val request = Request.Builder()
            .url("${config.relayBase}/$sessionTopic/json?since=2m")
            .header("User-Agent", "Airi-Control-Android/0.2")
            .build()
        val call = client.newCall(request)
        streamCall = call
        call.execute().use { response ->
            if (!response.isSuccessful) error("relay HTTP ${response.code}")
            val source = response.body?.source() ?: error("empty relay stream")
            while (true) {
                val line = source.readUtf8Line() ?: break
                if (line.isBlank()) continue
                val outer = runCatching { JSONObject(line) }.getOrNull() ?: continue
                if (outer.optString("event") != "message") continue
                val packet = runCatching { JSONObject(outer.optString("message")) }.getOrNull() ?: continue
                when (packet.optString("kind")) {
                    "phone_pair_request" -> acceptPairRequest(packet)
                    "phone_cmd" -> acceptCommand(packet)
                }
            }
        }
    }

    private fun acceptPairRequest(packet: JSONObject) {
        if (packet.optString("session_id") != sessionId) return
        if (packet.optString("device_key_id") != identity.keyId) return
        val ts = packet.optLong("ts", 0L)
        if (kotlin.math.abs(System.currentTimeMillis() / 1000 - ts) > 60) return
        val publicKey = packet.optString("controller_public_key_b64")
        val keyId = packet.optString("controller_key_id")
        if (publicKey.isBlank() || keyId.isBlank()) return
        if (CryptoBox.publicKeyId(publicKey) != keyId) return

        val existing = controller
        val idleMs = System.currentTimeMillis() - lastControllerCommandAt
        if (existing != null && existing.keyId != keyId && idleMs < 20_000L) return

        val accepted = SessionController(keyId, publicKey, System.currentTimeMillis())
        controller = accepted
        lastControllerCommandAt = System.currentTimeMillis()
        onControllerState(true)

        val plain = JSONObject()
            .put("ok", true)
            .put("session_id", sessionId)
            .put("device_key_id", identity.keyId)
            .put("controller_key_id", keyId)
            .put("ts", System.currentTimeMillis() / 1000)
            .toString()
            .toByteArray(Charsets.UTF_8)
        val envelope = CryptoBox.encryptEnvelope(publicKey, plain)
        val now = System.currentTimeMillis() / 1000
        val signed = canonical(
            "phone_pair_ack",
            sessionId,
            identity.keyId,
            keyId,
            envelope.getString("wrapped_key_b64"),
            envelope.getString("nonce_b64"),
            envelope.getString("ciphertext_b64"),
            now.toString(),
        )
        publishJson(
            sessionTopic,
            JSONObject()
                .put("kind", "phone_pair_ack")
                .put("session_id", sessionId)
                .put("device_key_id", identity.keyId)
                .put("controller_key_id", keyId)
                .put("wrapped_key_b64", envelope.getString("wrapped_key_b64"))
                .put("nonce_b64", envelope.getString("nonce_b64"))
                .put("ciphertext_b64", envelope.getString("ciphertext_b64"))
                .put("ts", now)
                .put("device_signature_b64", CryptoBox.signDevice(signed))
                .put("protocol", 2)
        )
    }

    private fun acceptCommand(packet: JSONObject) {
        if (packet.optString("session_id") != sessionId) return
        if (packet.optString("device_key_id") != identity.keyId) return
        val current = controller ?: return
        if (packet.optString("controller_key_id") != current.keyId) return
        val ts = packet.optLong("ts", 0L)
        if (kotlin.math.abs(System.currentTimeMillis() / 1000 - ts) > 120) return

        val wrapped = packet.optString("wrapped_key_b64")
        val nonce = packet.optString("nonce_b64")
        val ciphertext = packet.optString("ciphertext_b64")
        val signature = packet.optString("controller_signature_b64")
        val signed = canonical(
            "phone_cmd",
            sessionId,
            identity.keyId,
            current.keyId,
            wrapped,
            nonce,
            ciphertext,
            ts.toString(),
        )
        if (!CryptoBox.verifySignature(current.publicKeyB64, signed, signature)) return

        val plain = runCatching { CryptoBox.decryptEnvelope(packet) }.getOrNull() ?: return
        val cmd = runCatching { JSONObject(String(plain, Charsets.UTF_8)) }.getOrNull() ?: return
        val id = cmd.optString("id")
        val op = cmd.optString("op")
        val now = System.currentTimeMillis() / 1000
        val issued = cmd.optLong("issued_at", 0L)
        val expires = cmd.optLong("expires_at", 0L)
        if (id.isBlank() || op.isBlank() || issued > now + 30 || expires < now || expires - issued > 180) return

        synchronized(seenCommands) {
            if (seenCommands.containsKey(id)) return
            seenCommands[id] = System.currentTimeMillis()
            while (seenCommands.size > 200) {
                val first = seenCommands.entries.firstOrNull()?.key ?: break
                seenCommands.remove(first)
            }
        }
        lastControllerCommandAt = System.currentTimeMillis()
        val args = cmd.optJSONObject("args") ?: JSONObject()
        scope.launch(Dispatchers.IO) {
            onCommand(PhoneCommand(id, op, args, current))
        }
    }

    fun publishResult(command: PhoneCommand, result: JSONObject) {
        val plain = JSONObject(result.toString())
            .put("id", command.id)
            .put("session_id", sessionId)
            .put("device_key_id", identity.keyId)
            .put("ts", System.currentTimeMillis() / 1000)
            .toString()
            .toByteArray(Charsets.UTF_8)
        val envelope = CryptoBox.encryptEnvelope(command.controller.publicKeyB64, plain)
        val now = System.currentTimeMillis() / 1000
        val signed = canonical(
            "phone_result",
            sessionId,
            identity.keyId,
            command.controller.keyId,
            command.id,
            envelope.getString("wrapped_key_b64"),
            envelope.getString("nonce_b64"),
            envelope.getString("ciphertext_b64"),
            now.toString(),
        )
        publishJson(
            sessionTopic,
            JSONObject()
                .put("kind", "phone_result")
                .put("session_id", sessionId)
                .put("device_key_id", identity.keyId)
                .put("controller_key_id", command.controller.keyId)
                .put("command_id", command.id)
                .put("wrapped_key_b64", envelope.getString("wrapped_key_b64"))
                .put("nonce_b64", envelope.getString("nonce_b64"))
                .put("ciphertext_b64", envelope.getString("ciphertext_b64"))
                .put("ts", now)
                .put("device_signature_b64", CryptoBox.signDevice(signed))
                .put("protocol", 2)
        )
    }

    fun uploadEncryptedFrame(command: PhoneCommand, jpeg: ByteArray): JSONObject {
        val blob = CryptoBox.encryptBinary(jpeg)
        val request = Request.Builder()
            .url("${config.relayBase}/$sessionTopic")
            .header("User-Agent", "Airi-Control-Android/0.2")
            .header("Filename", "airi-phone-${command.id}.bin")
            .put(blob.ciphertext.toRequestBody("application/octet-stream".toMediaType()))
            .build()
        val attachmentUrl = client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) error("frame upload HTTP ${response.code}")
            val obj = JSONObject(response.body?.string().orEmpty())
            obj.optJSONObject("attachment")?.optString("url").orEmpty()
        }
        if (attachmentUrl.isBlank()) error("relay did not return attachment URL")
        return JSONObject()
            .put("frame_url", attachmentUrl)
            .put("frame_key_b64", blob.keyB64)
            .put("frame_nonce_b64", blob.nonceB64)
            .put("frame_type", "image/jpeg")
            .put("frame_bytes", jpeg.size)
    }

    private fun publishJson(topic: String, payload: JSONObject) {
        val request = Request.Builder()
            .url("${config.relayBase}/$topic")
            .header("User-Agent", "Airi-Control-Android/0.2")
            .post(payload.toString().toRequestBody("text/plain; charset=utf-8".toMediaType()))
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) error("relay publish HTTP ${response.code}")
        }
    }

    private fun canonical(vararg fields: String): ByteArray =
        fields.joinToString("\n").toByteArray(Charsets.UTF_8)

    private fun randomToken(bytes: Int): String {
        val raw = ByteArray(bytes).also { SecureRandom().nextBytes(it) }
        return Base64.encodeToString(raw, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
    }
}
