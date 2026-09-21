package com.airipc.generalist

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import org.json.JSONObject
import java.util.concurrent.TimeUnit

private const val ACTIONS_BASE =
    "https://api.github.com/repos/arancione3000/airi-pc-bootstrap/actions"
private const val MAIN_RUNS =
    "$ACTIONS_BASE/runs?branch=main&per_page=30"
private const val CONTINUUM_NAME = "AIRI Generalist Research Continuum"
private const val BOOTSTRAP_NAME = "AIRI Generalist Language Bootstrap"
private const val STATE_BRANCH = "generalist-state"
private const val STATE_ROOT = "generalist-state"

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
    val stateRevision: String,
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
    private val github = GitHubSnapshotClient(client)

    suspend fun fetch(): LiveEvolutionSnapshot = withContext(Dispatchers.IO) {
        val stateRevision = github.resolveBranchSha(STATE_BRANCH)
        val runs = github.getJson(MAIN_RUNS)
        val array = runs.getJSONArray("workflow_runs")

        fun latestRun(name: String): JSONObject? {
            for (index in 0 until array.length()) {
                val candidate = array.getJSONObject(index)
                if (candidate.optString("name") == name) return candidate
            }
            return null
        }

        val run = latestRun(CONTINUUM_NAME)
            ?: error("Nessun Continuum trovato")
        val runId = run.getLong("id")
        val jobsRaw = github.getJson("$ACTIONS_BASE/runs/$runId/jobs?per_page=100")
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

        val bootstrapRun = latestRun(BOOTSTRAP_NAME)
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

        val progressRaw = getStateJsonOrNull(
            stateRevision,
            "$STATE_ROOT/bootstrap-data/progress.json",
        )
        val manifestRaw = getStateJsonOrNull(
            stateRevision,
            "$STATE_ROOT/bootstrap-data/manifest.json",
        )
        val bootstrap = if (progressRaw != null && manifestRaw != null) {
            parseBootstrap(progressRaw, manifestRaw)
        } else {
            null
        }

        LiveEvolutionSnapshot(
            stateRevision = stateRevision,
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

    private fun getStateJsonOrNull(
        revision: String,
        path: String,
    ): JSONObject? =
        try {
            github.getJsonAtRevision(revision, path)
        } catch (_: Exception) {
            null
        }
}
