package com.airipc.generalist

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.concurrent.TimeUnit

private const val REPO = "arancione3000/airi-pc-bootstrap"
private const val MOBILE_BRANCH = "generalist-mobile"

data class BundleFileInfo(
    val sha256: String,
    val bytes: Long,
)

data class ModelSlot(
    val name: String,
    val id: String,
    val path: String,
    val cycle: Int,
    val parameters: Int,
    val score: Double,
    val nllPerByte: Double,
    val generationSimilarity: Double,
    val generationExactAccuracy: Double,
    val generationNonemptyRate: Double,
    val tokenizerVersion: String,
    val tokenizerVocabSize: Int,
    val contextLength: Int,
    val researchOnly: Boolean,
    val allSeedEligible: Boolean,
    val neural: NeuralDiagnostics?,
    val files: Map<String, BundleFileInfo>,
)

internal fun resolveDownloadRevision(
    revision: String,
    fallback: () -> String,
): String = revision.ifBlank(fallback)

internal fun sameModelArtifact(first: ModelSlot?, second: ModelSlot?): Boolean {
    val a = first?.files?.get("model.onnx")?.sha256.orEmpty()
    val b = second?.files?.get("model.onnx")?.sha256.orEmpty()
    return a.isNotBlank() && a == b
}

data class MobileManifest(
    val schema: Int,
    val mobileRevision: String,
    val stateSha: String,
    val cycle: Int,
    val generatedAt: Long,
    val generalistVersion: String,
    val promotedThisCycle: Boolean,
    val promotionReason: String,
    val evolution: EvolutionSummary,
    val airiPcLab: AiriPcLabReport,
    val slots: Map<String, ModelSlot>,
)

data class ManifestFetch(
    val manifest: MobileManifest,
    val freshFromNetwork: Boolean,
    val mobileRevision: String,
)

data class InstalledBundle(
    val slot: ModelSlot,
    val directory: File,
    val model: File,
    val config: File,
    val tokenizer: File,
)

class BundleRepository(context: Context) {
    private val root = File(context.filesDir, "airi-generalist-models").apply { mkdirs() }
    private val cachedManifest = File(root, "manifest.json")
    private val cachedRevision = File(root, "manifest-revision.txt")
    private val client = OkHttpClient.Builder()
        .connectTimeout(20, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .callTimeout(120, TimeUnit.SECONDS)
        .build()
    private val github = GitHubSnapshotClient(client, REPO)

    suspend fun fetchManifest(): ManifestFetch = withContext(Dispatchers.IO) {
        try {
            val revision = github.resolveBranchSha(MOBILE_BRANCH)
            val text = github.getTextAtRevision(revision, "manifest.json")
            val manifest = parseManifest(text, revision)
            val tmp = File(root, "manifest.json.tmp")
            tmp.writeText(text)
            if (cachedManifest.exists()) cachedManifest.delete()
            check(tmp.renameTo(cachedManifest))
            cachedRevision.writeText(revision)
            ManifestFetch(manifest, true, revision)
        } catch (network: Exception) {
            if (!cachedManifest.isFile) throw network
            val revision = cachedRevision.takeIf { it.isFile }
                ?.readText()
                ?.trim()
                .orEmpty()
            ManifestFetch(
                parseManifest(cachedManifest.readText(), revision),
                false,
                revision,
            )
        }
    }

    suspend fun ensureBundle(
        slot: ModelSlot,
        mobileRevision: String,
    ): InstalledBundle = withContext(Dispatchers.IO) {
        val modelHash = slot.files["model.onnx"]?.sha256
            ?: error("Manifest mobile privo di model.onnx")
        val directory = File(root, "${slot.name}-${modelHash.take(16)}").apply { mkdirs() }

        var downloadRevision = mobileRevision
        for ((name, info) in slot.files) {
            val target = File(directory, name)
            if (!target.isFile || sha256(target) != info.sha256) {
                downloadRevision = resolveDownloadRevision(downloadRevision) {
                    github.resolveBranchSha(MOBILE_BRANCH)
                }
                check(downloadRevision.isNotBlank()) { "Revisione mobile mancante" }
                val bytes = github.getBytesAtRevision(
                    downloadRevision,
                    "${slot.path}/$name",
                )
                val tmp = File(directory, "$name.tmp")
                tmp.writeBytes(bytes)
                check(sha256(tmp) == info.sha256) {
                    "SHA-256 non valido per $name"
                }
                if (target.exists()) target.delete()
                check(tmp.renameTo(target)) { "Impossibile installare $name" }
            }
        }

        val model = File(directory, "model.onnx")
        val config = File(directory, "config.json")
        val tokenizer = File(directory, "tokenizer.json")
        check(model.isFile && config.isFile && tokenizer.isFile) {
            "Bundle mobile incompleto"
        }

        root.listFiles()
            ?.filter {
                it.isDirectory &&
                it.name.startsWith("${slot.name}-") &&
                it != directory
            }
            ?.forEach { it.deleteRecursively() }

        InstalledBundle(slot, directory, model, config, tokenizer)
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(64 * 1024)
            while (true) {
                val count = input.read(buffer)
                if (count <= 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }

    private fun parseManifest(
        text: String,
        mobileRevision: String,
    ): MobileManifest {
        val root = JSONObject(text)
        check(root.getInt("schema") == 1)
        val rawSlots = root.getJSONObject("slots")
        val slots = mutableMapOf<String, ModelSlot>()
        for (name in rawSlots.keys()) {
            val raw = rawSlots.getJSONObject(name)
            val rawFiles = raw.getJSONObject("files")
            val files = mutableMapOf<String, BundleFileInfo>()
            for (fileName in rawFiles.keys()) {
                val f = rawFiles.getJSONObject(fileName)
                files[fileName] = BundleFileInfo(
                    sha256 = f.getString("sha256"),
                    bytes = f.getLong("bytes"),
                )
            }
            slots[name] = ModelSlot(
                name = name,
                id = raw.getString("id"),
                path = raw.getString("path"),
                cycle = raw.optInt("cycle", root.optInt("cycle", 0)),
                parameters = raw.optInt("parameters", 0),
                score = raw.optDouble("score", 0.0),
                nllPerByte = raw.optDouble("nll_per_byte", 0.0),
                generationSimilarity = raw.optDouble("generation_similarity", 0.0),
                generationExactAccuracy = raw.optDouble("generation_exact_accuracy", 0.0),
                generationNonemptyRate = raw.optDouble("generation_nonempty_rate", 0.0),
                tokenizerVersion = raw.optString("tokenizer_version"),
                tokenizerVocabSize = raw.optInt("tokenizer_vocab_size", 0),
                contextLength = raw.optInt("context_length", 128),
                researchOnly = raw.optBoolean("research_only", name != "champion"),
                allSeedEligible = raw.optBoolean("all_seed_eligible", name == "champion"),
                neural = parseNeuralDiagnostics(raw.optJSONObject("neural")),
                files = files,
            )
        }
        return MobileManifest(
            schema = root.getInt("schema"),
            mobileRevision = mobileRevision,
            stateSha = root.optString("state_sha"),
            cycle = root.optInt("cycle", 0),
            generatedAt = root.optLong("generated_at", 0L),
            generalistVersion = root.optString("generalist_version"),
            promotedThisCycle = root.optBoolean("promoted_this_cycle", false),
            promotionReason = root.optString("promotion_reason"),
            evolution = parseEvolutionSummary(root.optJSONObject("evolution")),
            airiPcLab = parseAiriPcLabReport(root.optJSONObject("airi_pc_lab")),
            slots = slots,
        )
    }
}
