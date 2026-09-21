package com.airipc.generalist

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import org.json.JSONObject
import java.util.concurrent.TimeUnit

private const val ACTIONS_BASE =
    "https://api.github.com/repos/arancione3000/airi-pc-bootstrap/actions"
private const val CONTINUUM_RUNS =
    "$ACTIONS_BASE/workflows/generalist-continuum.yml/runs?branch=main&per_page=1"
private const val BOOTSTRAP_RUNS =
    "$ACTIONS_BASE/workflows/generalist-bootstrap.yml/runs?branch=main&per_page=1"
private const val STATE_RAW =
    "https://raw.githubusercontent.com/arancione3000/airi-pc-bootstrap/generalist-state/generalist-state"
private const val BOOTSTRAP_PROGRESS =
    "$STATE_RAW/bootstrap-data/progress.json"
private const val BOOTSTRAP_MANIFEST =
    "$STATE_RAW/bootstrap-data/manifest.json"

data class LiveJob(
    val name: String,
    val status: String,
    val conclusion: String?,
)

data class BootstrapRun(
    val runId: Long,
    val runNumber: Int,
    val status: String,
    val conclusion: String?,
    val webUrl: String,
)

data class BootstrapProgress(
    val version: String,
    val targetTokens: Long,
    val tokensProcessed: Long,
    val steps: Long,
    val learningRate: Double?,
    val lastTrainLoss: Double?,
    val lastValidationLoss: Double?,
    val bestValidationLoss: Double?,
    val rungComplete: Boolean,
    val earlyStopped: Boolean,
    val sourceIds: List<String>,
    val englishTokens: Long,
    val italianTokens: Long,
    val domainTokens: Map<String, Long>,
    val tokenizerVersion: String,
    val tokenizerVocabSize: Int,
) {
    val progressFraction: Double
        get() = if (targetTokens > 0L) {
            (tokensProcessed.toDouble() / targetTokens.toDouble()).coerceIn(0.0, 1.0)
        } else {
            0.0
        }
}

data class LiveEvolutionSnapshot(
    val runId: Long,
    val runNumber: Int,
    val status: String,
    val conclusion: String?,
    val event: String,
    val createdAt: String,
    val updatedAt: String,
    val webUrl: String,
    val jobs: List<LiveJob>,
    val bootstrapRun: BootstrapRun?,
    val bootstrap: BootstrapProgress?,
) {
    val runningJobs: List<LiveJob>
        get() = jobs.filter { it.status == "in_progress" }
    val completedJobs: Int
        get() = jobs.count { it.status == "completed" }
    val stageLabel: String
        get() {
            val bootstrapStatus = bootstrapRun?.status.orEmpty()
            if (bootstrapStatus in setOf("queued", "pending", "in_progress", "waiting", "requested")) {
                return "Phase 5 · language bootstrap"
            }
            val names = runningJobs.map { it.name }
            return when {
                names.any { it.startsWith("stage3") } -> "Stage 3 · finalisti"
                names.any { it == "reduce2" } -> "Reducer 2"
                names.any { it.startsWith("stage2") } -> "Stage 2 · survivor"
                names.any { it == "reduce1" } -> "Reducer 1"
                names.any { it.startsWith("stage1") } -> "Stage 1 · swarm"
                names.any { it == "finalize" } -> "Verifier + persistenza"
                names.any { it == "plan" } -> "Planner + autodata"
                status == "queued" || status == "pending" -> "In coda"
                status == "completed" -> "Ciclo completato"
                else -> status
            }
        }
}

class LiveEvolutionRepository {
    private val client = OkHttpClient.Builder()
        .connectTimeout(15, TimeUnit.SECONDS)
        .readTimeout(30, TimeUnit.SECONDS)
        .callTimeout(45, TimeUnit.SECONDS)
        .build()

    suspend fun fetch(): LiveEvolutionSnapshot = withContext(Dispatchers.IO) {
        val runs = getJson(CONTINUUM_RUNS)
        val array = runs.getJSONArray("workflow_runs")
        check(array.length() > 0) { "Nessun Continuum trovato" }
        val run = array.getJSONObject(0)
        val runId = run.getLong("id")
        val jobsRaw = getJson("$ACTIONS_BASE/runs/$runId/jobs?per_page=100")
        val jobsArray = jobsRaw.getJSONArray("jobs")
        val jobs = buildList {
            for (i in 0 until jobsArray.length()) {
                val job = jobsArray.getJSONObject(i)
                add(
                    LiveJob(
                        name = job.optString("name"),
                        status = job.optString("status"),
                        conclusion = job.optString("conclusion")
                            .takeIf { it.isNotBlank() && it != "null" },
                    )
                )
            }
        }

        val bootstrapRun = getJsonOrNull(BOOTSTRAP_RUNS)
            ?.optJSONArray("workflow_runs")
            ?.takeIf { it.length() > 0 }
            ?.getJSONObject(0)
            ?.let { raw ->
                BootstrapRun(
                    runId = raw.optLong("id"),
                    runNumber = raw.optInt("run_number"),
                    status = raw.optString("status"),
                    conclusion = raw.optString("conclusion")
                        .takeIf { it.isNotBlank() && it != "null" },
                    webUrl = raw.optString("html_url"),
                )
            }

        val progressRaw = getJsonOrNull(BOOTSTRAP_PROGRESS)
        val manifestRaw = getJsonOrNull(BOOTSTRAP_MANIFEST)
        val bootstrap = if (progressRaw != null && manifestRaw != null) {
            parseBootstrap(progressRaw, manifestRaw)
        } else {
            null
        }

        LiveEvolutionSnapshot(
            runId = runId,
            runNumber = run.optInt("run_number"),
            status = run.optString("status"),
            conclusion = run.optString("conclusion")
                .takeIf { it.isNotBlank() && it != "null" },
            event = run.optString("event"),
            createdAt = run.optString("created_at"),
            updatedAt = run.optString("updated_at"),
            webUrl = run.optString("html_url"),
            jobs = jobs,
            bootstrapRun = bootstrapRun,
            bootstrap = bootstrap,
        )
    }

    private fun parseBootstrap(
        progress: JSONObject,
        manifest: JSONObject,
    ): BootstrapProgress {
        val languages = manifest.optJSONObject("token_counts_by_language")
        val domains = manifest.optJSONObject("token_counts_by_domain")
        val domainTokens = buildMap {
            if (domains != null) {
                for (key in domains.keys()) {
                    put(key, domains.optLong(key))
                }
            }
        }
        val sources = buildList {
            val array = manifest.optJSONArray("sources") ?: return@buildList
            for (i in 0 until array.length()) {
                val id = array.optJSONObject(i)?.optString("id").orEmpty()
                if (id.isNotBlank()) add(id)
            }
        }
        return BootstrapProgress(
            version = progress.optString("version"),
            targetTokens = progress.optLong("target_tokens"),
            tokensProcessed = progress.optLong("tokens_processed"),
            steps = progress.optLong("steps"),
            learningRate = progress.optDoubleOrNull("learning_rate"),
            lastTrainLoss = progress.optDoubleOrNull("last_train_loss"),
            lastValidationLoss = progress.optDoubleOrNull("last_validation_loss"),
            bestValidationLoss = progress.optDoubleOrNull("best_validation_loss"),
            rungComplete = progress.optBoolean("rung_complete"),
            earlyStopped = progress.optBoolean("early_stopped"),
            sourceIds = sources,
            englishTokens = languages?.optLong("english") ?: 0L,
            italianTokens = languages?.optLong("italian") ?: 0L,
            domainTokens = domainTokens,
            tokenizerVersion = manifest.optString("tokenizer_version"),
            tokenizerVocabSize = manifest.optInt("tokenizer_vocab_size"),
        )
    }

    private fun JSONObject.optDoubleOrNull(key: String): Double? {
        if (!has(key) || isNull(key)) return null
        return optDouble(key).takeIf { it.isFinite() }
    }

    private fun getJsonOrNull(url: String): JSONObject? =
        try {
            getJson(url)
        } catch (_: Exception) {
            null
        }

    private fun getJson(url: String): JSONObject {
        val request = Request.Builder()
            .url(url)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "AIRI-Generalist-Lab/1.3")
            .build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                error("GitHub API HTTP ${response.code}")
            }
            val text = response.body?.string() ?: error("Risposta GitHub vuota")
            return JSONObject(text)
        }
    }
}
