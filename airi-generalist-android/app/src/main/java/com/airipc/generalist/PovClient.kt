package com.airipc.generalist

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.MessageDigest
import java.security.spec.MGF1ParameterSpec
import java.util.concurrent.TimeUnit
import javax.crypto.Cipher
import javax.crypto.spec.OAEPParameterSpec
import javax.crypto.spec.PSource

private const val LIVE_CONFIG_URL =
    "https://raw.githubusercontent.com/arancione3000/airi-pc-bootstrap/main/.ai/airi_live.json"
private const val FALLBACK_RELAY = "https://ntfy.sh"
private const val FALLBACK_TOPIC =
    "airi-live-09d926b9a4207c660a5f3efa5f50505ef5ff46e8463b3beb"
private const val POV_KEY_ALIAS = "airi_generalist_pov_viewer_v1"

data class PovState(
    val connected: Boolean = false,
    val waiting: Boolean = true,
    val screenUrl: String = "",
    val sessionId: String = "",
    val lastOfferTs: Long = 0L,
    val error: String? = null,
)

private data class PovConfig(
    val relayBase: String,
    val topic: String,
)

private data class PovIdentity(
    val keyId: String,
    val publicKeyB64: String,
)

class AiriPovClient {
    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .callTimeout(45, TimeUnit.SECONDS)
        .build()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)
    private val identity by lazy { ensureIdentity() }
    private val _state = MutableStateFlow(PovState())
    val state: StateFlow<PovState> = _state
    private var job: Job? = null

    fun start() {
        if (job?.isActive == true) return
        job = scope.launch {
            while (isActive) {
                try {
                    val config = fetchConfig()
                    publishViewerKey(config)
                    _state.value = _state.value.copy(
                        connected = true,
                        waiting = _state.value.screenUrl.isBlank(),
                        error = null,
                    )
                    pollOffers(config)
                } catch (exc: CancellationException) {
                    throw exc
                } catch (exc: Exception) {
                    _state.value = _state.value.copy(
                        connected = false,
                        waiting = _state.value.screenUrl.isBlank(),
                        error = exc.message ?: exc.javaClass.simpleName,
                    )
                }
                delay(2500L)
            }
        }
    }

    fun stop() {
        job?.cancel()
        job = null
        scope.cancel()
    }

    private fun fetchConfig(): PovConfig {
        return runCatching {
            val request = Request.Builder()
                .url(LIVE_CONFIG_URL)
                .header("User-Agent", "AIRI-Generalist-Lab/1.2")
                .build()
            client.newCall(request).execute().use { response ->
                if (!response.isSuccessful) error("Live config HTTP ${response.code}")
                val root = JSONObject(response.body?.string().orEmpty())
                PovConfig(
                    relayBase = root.optString("relay_base", FALLBACK_RELAY).trimEnd('/'),
                    topic = root.optString("topic", FALLBACK_TOPIC),
                )
            }
        }.getOrElse {
            PovConfig(FALLBACK_RELAY, FALLBACK_TOPIC)
        }
    }

    private fun publishViewerKey(config: PovConfig) {
        val body = JSONObject()
            .put("kind", "viewer_key")
            .put("key_id", identity.keyId)
            .put("public_key_b64", identity.publicKeyB64)
            .put("ts", System.currentTimeMillis() / 1000)
            .put("protocol", 3)
            .toString()
            .toRequestBody("text/plain; charset=utf-8".toMediaType())
        val request = Request.Builder()
            .url("${config.relayBase}/${config.topic}")
            .header("User-Agent", "AIRI-Generalist-Lab/1.2")
            .post(body)
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) error("Viewer key HTTP ${response.code}")
        }
    }

    private fun pollOffers(config: PovConfig) {
        val request = Request.Builder()
            .url("${config.relayBase}/${config.topic}/json?poll=1&since=10m")
            .header("User-Agent", "AIRI-Generalist-Lab/1.2")
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) error("POV relay HTTP ${response.code}")
            var bestTs = _state.value.lastOfferTs
            var bestUrl = _state.value.screenUrl
            var bestSession = _state.value.sessionId
            for (line in response.body?.string().orEmpty().lineSequence()) {
                val outer = runCatching { JSONObject(line) }.getOrNull() ?: continue
                if (outer.optString("event") != "message") continue
                val control = runCatching {
                    JSONObject(outer.optString("message"))
                }.getOrNull() ?: continue
                if (control.optString("kind") != "screen_offer") continue
                if (control.optString("key_id") != identity.keyId) continue
                val ts = control.optLong("ts", 0L) * 1000L
                if (ts < bestTs) continue
                val descriptor = runCatching {
                    decryptOffer(control.getString("ciphertext_b64"))
                }.getOrNull() ?: continue
                val url = descriptor.optString("u")
                if (!url.startsWith("https://") || "/view/" !in url) continue
                bestTs = ts
                bestUrl = url
                bestSession = descriptor.optString("s")
            }
            _state.value = PovState(
                connected = true,
                waiting = bestUrl.isBlank(),
                screenUrl = bestUrl,
                sessionId = bestSession,
                lastOfferTs = bestTs,
                error = null,
            )
        }
    }

    private fun ensureIdentity(): PovIdentity {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        if (!store.containsAlias(POV_KEY_ALIAS)) {
            val generator = KeyPairGenerator.getInstance(
                KeyProperties.KEY_ALGORITHM_RSA,
                "AndroidKeyStore",
            )
            generator.initialize(
                KeyGenParameterSpec.Builder(
                    POV_KEY_ALIAS,
                    KeyProperties.PURPOSE_DECRYPT,
                )
                    .setKeySize(3072)
                    .setDigests(
                        KeyProperties.DIGEST_SHA256,
                        KeyProperties.DIGEST_SHA512,
                    )
                    .setEncryptionPaddings(
                        KeyProperties.ENCRYPTION_PADDING_RSA_OAEP,
                    )
                    .build()
            )
            generator.generateKeyPair()
        }
        val publicBytes = store.getCertificate(POV_KEY_ALIAS).publicKey.encoded
        val digest = MessageDigest.getInstance("SHA-256").digest(publicBytes)
        val keyId = digest.take(12).joinToString("") { "%02x".format(it) }
        return PovIdentity(
            keyId = keyId,
            publicKeyB64 = Base64.encodeToString(
                publicBytes,
                Base64.NO_WRAP,
            ),
        )
    }

    private fun decryptOffer(ciphertextB64: String): JSONObject {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val privateKey = store.getKey(POV_KEY_ALIAS, null)
        val cipher = Cipher.getInstance(
            "RSA/ECB/OAEPWithSHA-256AndMGF1Padding",
        )
        val spec = OAEPParameterSpec(
            "SHA-256",
            "MGF1",
            MGF1ParameterSpec.SHA1,
            PSource.PSpecified.DEFAULT,
        )
        cipher.init(Cipher.DECRYPT_MODE, privateKey, spec)
        val plain = cipher.doFinal(
            Base64.decode(ciphertextB64, Base64.NO_WRAP),
        )
        return JSONObject(String(plain, Charsets.UTF_8))
    }
}
