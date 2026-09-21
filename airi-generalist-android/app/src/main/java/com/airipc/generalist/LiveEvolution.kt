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

data class LiveJob(
    val name: String,
    val status: String,
    val conclusion: String?,
)

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
) {
    val runningJobs: List<LiveJob>
        get() = jobs.filter { it.status == "in_progress" }
    val completedJobs: Int
        get() = jobs.count { it.status == "completed" }
    val stageLabel: String
        get() {
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
        )
    }

    private fun getJson(url: String): JSONObject {
        val request = Request.Builder()
            .url(url)
            .header("Accept", "application/vnd.github+json")
            .header("X-GitHub-Api-Version", "2022-11-28")
            .header("User-Agent", "AIRI-Generalist-Lab/1.1")
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
