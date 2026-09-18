package com.airipc.control

import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import org.json.JSONObject
import java.security.KeyFactory
import java.security.KeyPairGenerator
import java.security.KeyStore
import java.security.MessageDigest
import java.security.SecureRandom
import java.security.Signature
import java.security.spec.MGF1ParameterSpec
import java.security.spec.X509EncodedKeySpec
import javax.crypto.Cipher
import javax.crypto.spec.GCMParameterSpec
import javax.crypto.spec.OAEPParameterSpec
import javax.crypto.spec.PSource
import javax.crypto.spec.SecretKeySpec

data class DeviceIdentity(
    val keyId: String,
    val encryptionPublicKeyB64: String,
    val signingPublicKeyB64: String,
)

data class EncryptedBlob(
    val ciphertext: ByteArray,
    val keyB64: String,
    val nonceB64: String,
)

object CryptoBox {
    private const val ENCRYPTION_ALIAS = "airi_control_device_rsa_v1"
    private const val SIGNING_ALIAS = "airi_control_device_sign_rsa_v2"

    fun ensureIdentity(): DeviceIdentity {
        ensureEncryptionKey()
        ensureSigningKey()
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val encryptionPublic = store.getCertificate(ENCRYPTION_ALIAS).publicKey.encoded
        val signingPublic = store.getCertificate(SIGNING_ALIAS).publicKey.encoded
        val keyId = MessageDigest.getInstance("SHA-256").digest(encryptionPublic)
            .take(12).joinToString("") { "%02x".format(it) }
        return DeviceIdentity(
            keyId = keyId,
            encryptionPublicKeyB64 = Base64.encodeToString(encryptionPublic, Base64.NO_WRAP),
            signingPublicKeyB64 = Base64.encodeToString(signingPublic, Base64.NO_WRAP),
        )
    }

    private fun ensureEncryptionKey() {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        if (store.containsAlias(ENCRYPTION_ALIAS)) return
        val generator = KeyPairGenerator.getInstance(KeyProperties.KEY_ALGORITHM_RSA, "AndroidKeyStore")
        generator.initialize(
            KeyGenParameterSpec.Builder(
                ENCRYPTION_ALIAS,
                KeyProperties.PURPOSE_DECRYPT or KeyProperties.PURPOSE_ENCRYPT,
            )
                .setKeySize(3072)
                .setDigests(KeyProperties.DIGEST_SHA256, KeyProperties.DIGEST_SHA512)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_RSA_OAEP)
                .build()
        )
        generator.generateKeyPair()
    }

    private fun ensureSigningKey() {
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        if (store.containsAlias(SIGNING_ALIAS)) return
        val generator = KeyPairGenerator.getInstance(KeyProperties.KEY_ALGORITHM_RSA, "AndroidKeyStore")
        generator.initialize(
            KeyGenParameterSpec.Builder(
                SIGNING_ALIAS,
                KeyProperties.PURPOSE_SIGN or KeyProperties.PURPOSE_VERIFY,
            )
                .setKeySize(3072)
                .setDigests(KeyProperties.DIGEST_SHA256)
                .setSignaturePaddings(KeyProperties.SIGNATURE_PADDING_RSA_PKCS1)
                .build()
        )
        generator.generateKeyPair()
    }

    fun publicKeyId(publicKeyB64: String): String {
        val bytes = Base64.decode(publicKeyB64, Base64.NO_WRAP)
        return MessageDigest.getInstance("SHA-256").digest(bytes)
            .take(12).joinToString("") { "%02x".format(it) }
    }

    fun signDevice(data: ByteArray): String {
        ensureSigningKey()
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val privateKey = store.getKey(SIGNING_ALIAS, null)
        val signature = Signature.getInstance("SHA256withRSA")
        signature.initSign(privateKey as java.security.PrivateKey)
        signature.update(data)
        return Base64.encodeToString(signature.sign(), Base64.NO_WRAP)
    }

    fun verifySignature(publicKeyB64: String, data: ByteArray, signatureB64: String): Boolean =
        runCatching {
            val publicKey = KeyFactory.getInstance("RSA").generatePublic(
                X509EncodedKeySpec(Base64.decode(publicKeyB64, Base64.NO_WRAP))
            )
            val signature = Signature.getInstance("SHA256withRSA")
            signature.initVerify(publicKey)
            signature.update(data)
            signature.verify(Base64.decode(signatureB64, Base64.NO_WRAP))
        }.getOrDefault(false)

    private fun oaep(): OAEPParameterSpec =
        OAEPParameterSpec("SHA-256", "MGF1", MGF1ParameterSpec.SHA1, PSource.PSpecified.DEFAULT)

    private fun unwrapForDevice(wrappedB64: String): ByteArray {
        ensureEncryptionKey()
        val store = KeyStore.getInstance("AndroidKeyStore").apply { load(null) }
        val privateKey = store.getKey(ENCRYPTION_ALIAS, null)
        val cipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
        cipher.init(Cipher.DECRYPT_MODE, privateKey, oaep())
        return cipher.doFinal(Base64.decode(wrappedB64, Base64.NO_WRAP))
    }

    private fun wrapTo(publicKeyB64: String, raw: ByteArray): String {
        val publicKey = KeyFactory.getInstance("RSA").generatePublic(
            X509EncodedKeySpec(Base64.decode(publicKeyB64, Base64.NO_WRAP))
        )
        val cipher = Cipher.getInstance("RSA/ECB/OAEPWithSHA-256AndMGF1Padding")
        cipher.init(Cipher.ENCRYPT_MODE, publicKey, oaep())
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
