package com.airipc.generalist

import org.json.JSONObject

data class NeuralArchitecture(
    val type: String,
    val vocabSize: Int,
    val contextLength: Int,
    val dModel: Int,
    val nLayers: Int,
    val nHeads: Int,
    val headDim: Int,
    val dFf: Int,
    val normType: String,
    val positionEncoding: String,
    val ffVariant: String,
    val tiedEmbeddingLmHead: Boolean,
)

data class NeuralComponent(
    val id: String,
    val label: String,
    val parameters: Int,
    val meanAbsWeight: Double,
    val rmsWeight: Double,
    val maxAbsWeight: Double,
)

data class NeuralHead(
    val layer: Int,
    val head: Int,
    val parameters: Int,
    val meanAbsWeight: Double,
    val rmsWeight: Double,
    val maxAbsWeight: Double,
)

data class NeuralConnection(
    val from: String,
    val to: String,
    val kind: String,
)

data class NeuralDiagnostics(
    val kind: String,
    val note: String,
    val architecture: NeuralArchitecture,
    val components: List<NeuralComponent>,
    val heads: List<NeuralHead>,
    val connections: List<NeuralConnection>,
)

data class EvolutionRepo(
    val repo: String,
    val files: Int,
    val bytes: Long,
    val domains: Map<String, Int>,
)

data class DataGrowthSummary(
    val version: String,
    val queries: List<String>,
    val desiredDomains: List<String>,
    val addedFiles: Int,
    val addedBytes: Long,
    val totalFiles: Int,
    val totalBytes: Long,
    val domainFiles: Map<String, Int>,
    val domainBytes: Map<String, Long>,
    val repositoriesUsed: Int,
    val topRepositories: List<EvolutionRepo>,
    val rejectionCounts: Map<String, Int>,
)

data class EvolutionSummary(
    val signals: List<String>,
    val domainWeights: Map<String, Double>,
    val curriculumStored: Int,
    val curriculumDomains: Map<String, Int>,
    val dataGrowth: DataGrowthSummary,
    val scaleCurrentParameters: Int,
    val scaleTargetParameters: Int?,
    val scaleCandidateGenerated: Boolean,
    val tokenizerCurrentVocab: Int,
    val tokenizerTargetVocab: Int,
    val tokenizerMaxVocab: Int,
    val tokenizerProbeGenerated: Boolean,
    val latestResearchId: String?,
)

data class LabModule(
    val name: String,
    val path: String,
    val symbols: List<String>,
)

data class LabProbe(
    val ok: Boolean,
    val toolCallValid: Boolean,
    val tool: String?,
    val error: String?,
    val modelSummary: String?,
)

data class AiriPcLabReport(
    val version: String,
    val mode: String,
    val trainingRows: Int,
    val trainingDomains: Map<String, Int>,
    val capabilities: List<String>,
    val deniedCapabilities: List<String>,
    val modules: List<LabModule>,
    val champion: LabProbe?,
    val research: LabProbe?,
)

internal fun parseNeuralDiagnostics(raw: JSONObject?): NeuralDiagnostics? {
    if (raw == null) return null
    val archRaw = raw.optJSONObject("architecture") ?: return null
    val architecture = NeuralArchitecture(
        type = archRaw.optString("type"),
        vocabSize = archRaw.optInt("vocab_size"),
        contextLength = archRaw.optInt("context_length"),
        dModel = archRaw.optInt("d_model"),
        nLayers = archRaw.optInt("n_layers"),
        nHeads = archRaw.optInt("n_heads"),
        headDim = archRaw.optInt("head_dim"),
        dFf = archRaw.optInt("d_ff"),
        normType = archRaw.optString("norm_type"),
        positionEncoding = archRaw.optString("position_encoding"),
        ffVariant = archRaw.optString("ff_variant"),
        tiedEmbeddingLmHead = archRaw.optBoolean("tied_embedding_lm_head"),
    )
    val components = buildList {
        val array = raw.optJSONArray("components") ?: return@buildList
        for (i in 0 until array.length()) {
            val row = array.optJSONObject(i) ?: continue
            add(
                NeuralComponent(
                    id = row.optString("id"),
                    label = row.optString("label"),
                    parameters = row.optInt("parameters"),
                    meanAbsWeight = row.optDouble("mean_abs_weight"),
                    rmsWeight = row.optDouble("rms_weight"),
                    maxAbsWeight = row.optDouble("max_abs_weight"),
                )
            )
        }
    }
    val heads = buildList {
        val array = raw.optJSONArray("heads") ?: return@buildList
        for (i in 0 until array.length()) {
            val row = array.optJSONObject(i) ?: continue
            add(
                NeuralHead(
                    layer = row.optInt("layer"),
                    head = row.optInt("head"),
                    parameters = row.optInt("parameters"),
                    meanAbsWeight = row.optDouble("mean_abs_weight"),
                    rmsWeight = row.optDouble("rms_weight"),
                    maxAbsWeight = row.optDouble("max_abs_weight"),
                )
            )
        }
    }
    val connections = buildList {
        val array = raw.optJSONArray("connections") ?: return@buildList
        for (i in 0 until array.length()) {
            val row = array.optJSONObject(i) ?: continue
            add(
                NeuralConnection(
                    from = row.optString("from"),
                    to = row.optString("to"),
                    kind = row.optString("kind"),
                )
            )
        }
    }
    return NeuralDiagnostics(
        kind = raw.optString("kind"),
        note = raw.optString("note"),
        architecture = architecture,
        components = components,
        heads = heads,
        connections = connections,
    )
}

private fun stringList(raw: JSONObject?, key: String): List<String> {
    val array = raw?.optJSONArray(key) ?: return emptyList()
    return buildList {
        for (i in 0 until array.length()) {
            val value = array.optString(i)
            if (value.isNotBlank()) add(value)
        }
    }
}

private fun intMap(raw: JSONObject?, key: String): Map<String, Int> {
    val obj = raw?.optJSONObject(key) ?: return emptyMap()
    return buildMap {
        for (name in obj.keys()) put(name, obj.optInt(name))
    }
}

private fun longMap(raw: JSONObject?, key: String): Map<String, Long> {
    val obj = raw?.optJSONObject(key) ?: return emptyMap()
    return buildMap {
        for (name in obj.keys()) put(name, obj.optLong(name))
    }
}

private fun doubleMap(raw: JSONObject?, key: String): Map<String, Double> {
    val obj = raw?.optJSONObject(key) ?: return emptyMap()
    return buildMap {
        for (name in obj.keys()) put(name, obj.optDouble(name))
    }
}

internal fun parseEvolutionSummary(raw: JSONObject?): EvolutionSummary {
    val growth = raw?.optJSONObject("data_growth")
    val repos = buildList {
        val array = growth?.optJSONArray("top_repositories") ?: return@buildList
        for (i in 0 until array.length()) {
            val row = array.optJSONObject(i) ?: continue
            add(
                EvolutionRepo(
                    repo = row.optString("repo"),
                    files = row.optInt("files"),
                    bytes = row.optLong("bytes"),
                    domains = intMap(row, "domains"),
                )
            )
        }
    }
    val dataGrowth = DataGrowthSummary(
        version = growth?.optString("version").orEmpty(),
        queries = stringList(growth, "queries"),
        desiredDomains = stringList(growth, "desired_domains"),
        addedFiles = growth?.optInt("added_files") ?: 0,
        addedBytes = growth?.optLong("added_bytes") ?: 0L,
        totalFiles = growth?.optInt("total_files") ?: 0,
        totalBytes = growth?.optLong("total_bytes") ?: 0L,
        domainFiles = intMap(growth, "domain_files"),
        domainBytes = longMap(growth, "domain_bytes"),
        repositoriesUsed = growth?.optInt("repositories_used") ?: 0,
        topRepositories = repos,
        rejectionCounts = intMap(growth, "rejection_counts"),
    )
    val curriculum = raw?.optJSONObject("curriculum_memory")
    val scaling = raw?.optJSONObject("progressive_scaling")
    val tokenizer = raw?.optJSONObject("progressive_tokenizer")
    val latest = raw?.optJSONObject("latest_research")
    val target = if (scaling?.isNull("target_parameters") == false) {
        scaling.optInt("target_parameters")
    } else {
        null
    }
    return EvolutionSummary(
        signals = stringList(raw, "signals"),
        domainWeights = doubleMap(raw, "domain_weights"),
        curriculumStored = curriculum?.optInt("stored") ?: 0,
        curriculumDomains = intMap(curriculum, "domains"),
        dataGrowth = dataGrowth,
        scaleCurrentParameters = scaling?.optInt("current_parameters") ?: 0,
        scaleTargetParameters = target,
        scaleCandidateGenerated = scaling?.optBoolean("candidate_generated") ?: false,
        tokenizerCurrentVocab = tokenizer?.optInt("current_vocab_size") ?: 0,
        tokenizerTargetVocab = tokenizer?.optInt("target_vocab_size") ?: 0,
        tokenizerMaxVocab = tokenizer?.optInt("maximum_vocab_size") ?: 0,
        tokenizerProbeGenerated = tokenizer?.optBoolean("probe_generated") ?: false,
        latestResearchId = latest?.optString("candidate_id")?.takeIf { it.isNotBlank() },
    )
}

private fun parseLabProbe(raw: JSONObject?): LabProbe? {
    if (raw == null) return null
    return LabProbe(
        ok = raw.optBoolean("ok"),
        toolCallValid = raw.optBoolean("tool_call_valid"),
        tool = raw.optString("tool").takeIf { it.isNotBlank() },
        error = raw.optString("error").takeIf { it.isNotBlank() },
        modelSummary = raw.optString("model_summary").takeIf { it.isNotBlank() },
    )
}

internal fun parseAiriPcLabReport(raw: JSONObject?): AiriPcLabReport {
    val snapshot = raw?.optJSONObject("snapshot")
    val learning = raw?.optJSONObject("learning")
    val modules = buildList {
        val array = snapshot?.optJSONArray("modules") ?: return@buildList
        for (i in 0 until array.length()) {
            val row = array.optJSONObject(i) ?: continue
            val symbols = buildList {
                val rawSymbols = row.optJSONArray("symbols") ?: return@buildList
                for (j in 0 until rawSymbols.length()) {
                    val symbol = rawSymbols.optJSONObject(j) ?: continue
                    val name = symbol.optString("name")
                    if (name.isNotBlank()) add(name)
                }
            }
            add(
                LabModule(
                    name = row.optString("name"),
                    path = row.optString("path"),
                    symbols = symbols,
                )
            )
        }
    }
    return AiriPcLabReport(
        version = raw?.optString("version").orEmpty(),
        mode = raw?.optString("mode").orEmpty(),
        trainingRows = learning?.optInt("training_rows") ?: 0,
        trainingDomains = intMap(learning, "domains"),
        capabilities = stringList(snapshot, "capabilities"),
        deniedCapabilities = stringList(snapshot, "denied_capabilities"),
        modules = modules,
        champion = parseLabProbe(raw?.optJSONObject("champion")),
        research = parseLabProbe(raw?.optJSONObject("research")),
    )
}
