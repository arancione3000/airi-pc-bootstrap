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
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit

private const val CONTROL_REPO = "arancione3000/airi-pc-bootstrap"
private const val CONTROL_CONFIG_URL =
    "https://raw.githubusercontent.com/$CONTROL_REPO/main/.ai/airi_control.json"
private const val FALLBACK_RELAY = "https://ntfy.sh"
private const val FALLBACK_TOPIC =
    "airi-control-6f4ea987afbe4cc9b71c11f0421e4d88555ee508c97a4102"

data class ControlConfig(
    val relayBase: String = FALLBACK_RELAY,
    val topic: String = FALLBACK_TOPIC,
    val history: String = "30m",
)

data class ControllerKey(
    val keyId: String,
    val publicKeyB64: String,
    val seenAt: Long,
)

data class PhoneCommand(
    val id: String,
    val op: String,
    val args: JSONObject,
    val controller: ControllerKey,
)

class ControlClient(
    private val context: Context,
    private val scope: CoroutineScope,
    private val onCommand: suspend (PhoneCommand) -> Unit,
) {
    private val client = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS)
        .build()
    private val identity by lazy { CryptoBox.ensureIdentity() }
    private val controllerKeys = ConcurrentHashMap<String, ControllerKey>()
    private val seenCommands = LinkedHashMap<String, Long>()
    private var streamCall: Call? = null
    private var job: Job? = null
    @Volatile private var config: ControlConfig = ControlConfig()

    fun start() {
        stop()
        job = scope.launch(Dispatchers.IO) {
            while (isActive) {
                try {
                    config = fetchConfig()
                    publishDeviceKey()
                    stream()
                } catch (e: CancellationException) {
                    throw e
                } catch (e: Exception) {
                    Log.w("AiriControl", "control stream reconnecting", e)
                    delay(3000)
                }
            }
        }
    }

    fun stop() {
        streamCall?.cancel()
        streamCall = null
        job?.cancel()
        job = null
    }

    fun deviceKeyId(): String = identity.keyId

    private fun fetchConfig(): ControlConfig {
        val relayOverride = BuildConfig.CONTROL_RELAY_OVERRIDE.trim()
        val topicOverride = BuildConfig.CONTROL_TOPIC_OVERRIDE.trim()
        if (relayOverride.isNotBlank() && topicOverride.isNotBlank()) {
            return ControlConfig(
                relayBase = relayOverride.trimEnd('/'),
                topic = topicOverride,
                history = "30m",
            )
        }
        return runCatching {
        val request = Request.Builder()
            .url(CONTROL_CONFIG_URL)
            .header("User-Agent", "Airi-Control-Android/0.1")
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) error("config HTTP ${response.code}")
            val obj = JSONObject(response.body?.string().orEmpty())
            ControlConfig(
                relayBase = obj.optString("relay_base", FALLBACK_RELAY).trimEnd('/'),
                topic = obj.optString("topic", FALLBACK_TOPIC),
                history = obj.optString("history", "30m"),
            )
        }
        }.getOrElse { ControlConfig() }
    }

    private fun publishDeviceKey() {
        val payload = JSONObject()
            .put("kind", "phone_device_key")
            .put("device_key_id", identity.keyId)
            .put("public_key_b64", identity.publicKeyB64)
            .put("auth_id", PairingSecret.authId(context))
            .put("ts", System.currentTimeMillis() / 1000)
            .put("protocol", 1)
        publishJson(payload)
    }

    private fun stream() {
        val request = Request.Builder()
            .url("${config.relayBase}/${config.topic}/json?since=${config.history}")
            .header("User-Agent", "Airi-Control-Android/0.1")
            .build()
        val call = client.newCall(request)
        streamCall = call
        call.execute().use { response ->
            if (!response.isSuccessful) error("relay HTTP ${response.code}")
            val source = response.body?.source() ?: error("empty relay stream")
            var lastDeviceKey = System.currentTimeMillis()
            while (true) {
                val line = source.readUtf8Line() ?: break
                if (line.isBlank()) continue
                val outer = runCatching { JSONObject(line) }.getOrNull() ?: continue
                if (outer.optString("event") != "message") continue
                val packet = runCatching { JSONObject(outer.optString("message")) }.getOrNull() ?: continue
                when (packet.optString("kind")) {
                    "phone_controller_key" -> acceptControllerKey(packet)
                    "phone_cmd" -> acceptCommand(packet)
                }
                if (System.currentTimeMillis() - lastDeviceKey > 10 * 60 * 1000L) {
                    publishDeviceKey()
                    lastDeviceKey = System.currentTimeMillis()
                }
            }
        }
    }

    private fun acceptControllerKey(packet: JSONObject) {
        val keyId = packet.optString("controller_key_id")
        val publicKey = packet.optString("public_key_b64")
        val ts = packet.optLong("ts", 0L)
        val authId = packet.optString("auth_id")
        val tag = packet.optString("auth_tag")
        if (keyId.isBlank() || publicKey.isBlank() || authId != PairingSecret.authId(context)) return
        if (kotlin.math.abs(System.currentTimeMillis() / 1000 - ts) > 300) return
        val ok = PairingSecret.verify(
            context,
            tag,
            "phone_controller_key", keyId, publicKey, ts.toString(), authId,
        )
        if (!ok) return
        controllerKeys[keyId] = ControllerKey(keyId, publicKey, System.currentTimeMillis())
        controllerKeys.entries.removeIf { System.currentTimeMillis() - it.value.seenAt > 20 * 60 * 1000L }
    }

    private fun acceptCommand(packet: JSONObject) {
        if (packet.optString("device_key_id") != identity.keyId) return
        val controllerId = packet.optString("controller_key_id")
        val controller = controllerKeys[controllerId] ?: return
        val ts = packet.optLong("ts", 0L)
        val authId = packet.optString("auth_id")
        if (authId != PairingSecret.authId(context)) return
        if (kotlin.math.abs(System.currentTimeMillis() / 1000 - ts) > 120) return

        val wrapped = packet.optString("wrapped_key_b64")
        val nonce = packet.optString("nonce_b64")
        val ciphertext = packet.optString("ciphertext_b64")
        val tag = packet.optString("auth_tag")
        val verified = PairingSecret.verify(
            context,
            tag,
            "phone_cmd",
            identity.keyId,
            controllerId,
            wrapped,
            nonce,
            ciphertext,
            ts.toString(),
            authId,
        )
        if (!verified) return

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
        val args = cmd.optJSONObject("args") ?: JSONObject()
        scope.launch(Dispatchers.IO) { onCommand(PhoneCommand(id, op, args, controller)) }
    }

    fun publishResult(command: PhoneCommand, result: JSONObject) {
        val plain = JSONObject(result.toString())
            .put("id", command.id)
            .put("device_key_id", identity.keyId)
            .put("ts", System.currentTimeMillis() / 1000)
            .toString()
            .toByteArray(Charsets.UTF_8)
        val envelope = CryptoBox.encryptEnvelope(command.controller.publicKeyB64, plain)
        val ts = System.currentTimeMillis() / 1000
        val authId = PairingSecret.authId(context)
        val wrapped = envelope.getString("wrapped_key_b64")
        val nonce = envelope.getString("nonce_b64")
        val ciphertext = envelope.getString("ciphertext_b64")
        val tag = PairingSecret.tag(
            context,
            "phone_result",
            identity.keyId,
            command.controller.keyId,
            command.id,
            wrapped,
            nonce,
            ciphertext,
            ts.toString(),
            authId,
        )
        publishJson(
            JSONObject()
                .put("kind", "phone_result")
                .put("device_key_id", identity.keyId)
                .put("controller_key_id", command.controller.keyId)
                .put("command_id", command.id)
                .put("wrapped_key_b64", wrapped)
                .put("nonce_b64", nonce)
                .put("ciphertext_b64", ciphertext)
                .put("ts", ts)
                .put("auth_id", authId)
                .put("auth_tag", tag)
                .put("protocol", 1)
        )
    }

    fun uploadEncryptedFrame(command: PhoneCommand, jpeg: ByteArray): JSONObject {
        val blob = CryptoBox.encryptBinary(jpeg)
        val request = Request.Builder()
            .url("${config.relayBase}/${config.topic}")
            .header("User-Agent", "Airi-Control-Android/0.1")
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

    private fun publishJson(payload: JSONObject) {
        val request = Request.Builder()
            .url("${config.relayBase}/${config.topic}")
            .header("User-Agent", "Airi-Control-Android/0.1")
            .post(payload.toString().toRequestBody("text/plain; charset=utf-8".toMediaType()))
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) error("relay publish HTTP ${response.code}")
        }
    }
}
