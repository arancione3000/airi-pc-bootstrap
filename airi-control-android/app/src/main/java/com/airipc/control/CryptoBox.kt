package com.airipc.control

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import org.json.JSONObject
import java.security.KeyFactory
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.MessageDigest
import java.security.SecureRandom
import java.security.spec.MGF1ParameterSpec
import java.security.spec.X509EncodedKeySpec
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.Mac
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.OAEPParameterSpec
import javax.crypto.spec.PSource
import javax.crypto.spec.SecretKeySpec

data class DeviceIdentity(val keyId: String, val publicKeyB64: String)
data class EncryptedBlob(val ciphertext: ByteArray, val keyB64: String, val nonceB64: String)

object PairingSecret {
    private const val PREFS = "airi_control_pairing"
    private const val KEY = "pairing_secret_v1"

    fun text(context: Context): String {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        prefs.getString(KEY, null)?.let { return it }
        val raw = ByteArray(32).also { SecureRandom().nextBytes(it) }
        val encoded = Base64.encodeToString(raw, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
        check(prefs.edit().putString(KEY, encoded).commit()) { "Unable to persist pairing secret" }
        return encoded
    }

    fun bytes(context: Context): ByteArray =
        Base64.decode(text(context), Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)

    fun authId(context: Context): String =
        MessageDigest.getInstance("SHA-256").digest(bytes(context))
            .take(8).joinToString("") { "%02x".format(it) }

    fun tag(context: Context, vararg fields: String): String {
        val mac = Mac.getInstance("HmacSHA256")
        mac.init(SecretKeySpec(bytes(context), "HmacSHA256"))
        val out = mac.doFinal(fields.joinToString("\n").toByteArray(Charsets.UTF_8))
        return Base64.encodeToString(out, Base64.URL_SAFE or Base64.NO_WRAP or Base64.NO_PADDING)
    }

    fun verify(context: Context, expected: String, vararg fields: String): Boolean {
        val actual = tag(context, *fields)
        return MessageDigest.isEqual(actual.toByteArray(), expected.toByteArray())
    }
}

object CryptoBox {
    private const val DEVICE_KEY_ALIAS = "airi_control_device_rsa_v1"

    fun ensureIdentity(): DeviceIdentity {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        if (!store.containsAlias(DEVICE_KEY_ALIAS)) {
            val generator = KeyPairGenerator.getInstance(KeyProperties.KEY_ALGORITHM_RSA, "AndroidKeyStore")
            generator.initialize(
                KeyGenParameterSpec.Builder(
                    DEVICE_KEY_ALIAS,
                    KeyProperties.PURPOSE_DECRYPT or KeyProperties.PURPOSE_ENCRYPT,
                )
                    .setKeySize(3072)
                    .setDigests(KeyProperties.DIGEST_SHA256, KeyProperties.DIGEST_SHA512)
                    .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_OAEP)
                    .build()
            )
            generator.generateKeyPair()
        }
        val publicBytes = store.getCertificate(DEVICE_KEY_ALIAS).publicKey.encoded
        val keyId = MessageDigest.getInstance("SHA-256").digest(publicBytes)
            .take(12).joinToString("") { "%02x".format(it) }
        return DeviceIdentity(keyId, Base64.encodeToString(publicBytes, Base64.NO_WRAP))
    }

    private fun oaep(): OAEPParameterSpec =
        OAEPParameterSpec("SHA-256", "MGF1", MGF1ParameterSpec.SHA1, PSource.PSpecified.DEFAULT)

    private fun unwrapForDevice(wrappedB64: String): ByteArray {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val privateKey = store.getKey(DEVICE_KEY_ALIAS, null)
        val cipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
        cipher.init(Cipher.DECRYPT_MODE, privateKey, oaep())
        return cipher.doFinal(Base64.decode(wrappedB64, Base64.NO_WRAP))
    }

    private fun wrapTo(publicKeyB64: String, raw: ByteArray): String {
        val pub = KeyFactory.getInstance("RSA").generatePublic(
            X509EncodedKeySpec(Base64.decode(publicKeyB64, Base64.NO_WRAP))
        )
        val cipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
        cipher.init(Cipher.ENCRYPT_MODE, pub, oaep())
        return Base64.encodeToString(cipher.doFinal(raw), Base64.NO_WRAP)
    }

    fun decryptEnvelope(packet: JSONObject): ByteArray {
        val key = unwrapForDevice(packet.getString("wrapped_key_b64"))
        val nonce = Base64.decode(packet.getString("nonce_b64"), Base64.NO_WRAP)
        val ciphertext = Base64.decode(packet.getString("ciphertext_b64"), Base64.NO_WRAP)
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.DECRYPT_MODE, SecretKeySpec(key, "AES"), GCMParameterSpec(128, nonce))
        return cipher.doFinal(ciphertext)
    }

    fun encryptEnvelope(publicKeyB64: String, plain: ByteArray): JSONObject {
        val key = ByteArray(32).also { SecureRandom().nextBytes(it) }
        val nonce = ByteArray(12).also { SecureRandom().nextBytes(it) }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(key, "AES"), GCMParameterSpec(128, nonce))
        val ciphertext = cipher.doFinal(plain)
        return JSONObject()
            .put("wrapped_key_b64", wrapTo(publicKeyB64, key))
            .put("nonce_b64", Base64.encodeToString(nonce, Base64.NO_WRAP))
            .put("ciphertext_b64", Base64.encodeToString(ciphertext, Base64.NO_WRAP))
    }

    fun encryptBinary(plain: ByteArray): EncryptedBlob {
        val key = ByteArray(32).also { SecureRandom().nextBytes(it) }
        val nonce = ByteArray(12).also { SecureRandom().nextBytes(it) }
        val cipher = Cipher.getInstance("AES/GCM/NoPadding")
        cipher.init(Cipher.ENCRYPT_MODE, SecretKeySpec(key, "AES"), GCMParameterSpec(128, nonce))
        return EncryptedBlob(
            ciphertext = cipher.doFinal(plain),
            keyB64 = Base64.encodeToString(key, Base64.NO_WRAP),
            nonceB64 = Base64.encodeToString(nonce, Base64.NO_WRAP),
        )
    }
}
