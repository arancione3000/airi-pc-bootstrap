package com.airipc.generalist

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.util.Locale

enum class AppTab(val label: String, val glyph: String) {
    CHAT("Chat", "💬"),
    LIVE("Live", "◉"),
    NEURAL("Neural", "◎"),
    AIRI_PC("Airi-PC", "▣"),
}

@Composable
fun GeneralistBottomBar(selected: AppTab, onSelect: (AppTab) -> Unit) {
    NavigationBar {
        AppTab.entries.forEach { tab ->
            NavigationBarItem(
                selected = selected == tab,
                onClick = { onSelect(tab) },
                icon = { Text(tab.glyph) },
                label = { Text(tab.label) },
            )
        }
    }
}

@Composable
fun LiveEvolutionScreen(
    state: AppUiState,
    onRefreshLive: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val manifest = state.manifest
    val evolution = manifest?.evolution
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = PaddingValues(bottom = 20.dp),
    ) {
        item {
            SectionCard("Continuum H24") {
                val live = state.liveEvolution
                if (live == null) {
                    Text("Stato live non ancora disponibile.")
                } else {
                    Text(
                        "Run #${live.runNumber} · ${live.stageLabel}",
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        "${live.status}${live.conclusion?.let { " · $it" } ?: ""} · " +
                            "${live.completedJobs}/${live.jobs.size} job completati",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Spacer(Modifier.height(6.dp))
                    if (live.status == "in_progress") {
                        LinearProgressIndicator(modifier = Modifier.fillMaxWidth())
                    }
                    live.runningJobs.take(8).forEach {
                        Text("• ${it.name}", style = MaterialTheme.typography.bodySmall)
                    }
                }
                state.liveError?.let {
                    Text(it, color = MaterialTheme.colorScheme.error)
                }
                TextButton(onClick = onRefreshLive) { Text("Aggiorna stato live") }
            }
        }

        if (manifest != null && evolution != null) {
            item {
                SectionCard("Ultimo stato persistito") {
                    Text(
                        "Ciclo ${manifest.cycle} · ${manifest.generalistVersion}",
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        if (manifest.promotedThisCycle) {
                            "Questo ciclo ha promosso un nuovo champion."
                        } else {
                            "Nessuna promozione: ${manifest.promotionReason}"
                        },
                        style = MaterialTheme.typography.bodySmall,
                    )
                    if (evolution.signals.isNotEmpty()) {
                        Text("Segnali: ${evolution.signals.joinToString()}")
                    }
                    Text(
                        "Curriculum: ${evolution.curriculumStored} righe · " +
                            evolution.curriculumDomains.entries
                                .sortedBy { it.key }
                                .joinToString { "${it.key}=${it.value}" },
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            item {
                SectionCard("Autodata · cosa sta cercando") {
                    val growth = evolution.dataGrowth
                    Text(
                        "${formatBytes(growth.totalBytes)} · ${growth.totalFiles} file · " +
                            "+${growth.addedFiles} nell'ultimo ciclo",
                        fontWeight = FontWeight.SemiBold,
                    )
                    if (growth.desiredDomains.isNotEmpty()) {
                        Text("Priorità: ${growth.desiredDomains.joinToString()}")
                    }
                    if (growth.queries.isNotEmpty()) {
                        Spacer(Modifier.height(6.dp))
                        Text("Query GitHub reali:", fontWeight = FontWeight.SemiBold)
                        growth.queries.forEach { Text("• $it", style = MaterialTheme.typography.bodySmall) }
                    }
                    if (growth.domainBytes.isNotEmpty()) {
                        Spacer(Modifier.height(6.dp))
                        growth.domainBytes.entries
                            .sortedByDescending { it.value }
                            .forEach {
                                Text(
                                    "${it.key}: ${formatBytes(it.value)} · " +
                                        "${growth.domainFiles[it.key] ?: 0} file",
                                    style = MaterialTheme.typography.bodySmall,
                                )
                            }
                    }
                }
            }

            item {
                SectionCard("Dove prende i dati") {
                    val repos = evolution.dataGrowth.topRepositories
                    if (repos.isEmpty()) {
                        Text("Nessun repository nel riepilogo corrente.")
                    } else {
                        repos.take(10).forEach { repo ->
                            Text(repo.repo, fontWeight = FontWeight.SemiBold)
                            Text(
                                "${repo.files} file · ${formatBytes(repo.bytes)} · " +
                                    repo.domains.entries.joinToString { "${it.key}=${it.value}" },
                                style = MaterialTheme.typography.bodySmall,
                            )
                            Spacer(Modifier.height(4.dp))
                        }
                    }
                    val rejected = evolution.dataGrowth.rejectionCounts
                    if (rejected.isNotEmpty()) {
                        Divider()
                        Text("Scarti fail-closed", fontWeight = FontWeight.SemiBold)
                        rejected.entries.sortedByDescending { it.value }.take(8).forEach {
                            Text("• ${it.key}: ${it.value}", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }

            item {
                SectionCard("Scaling") {
                    Text(
                        "Parametri: ${evolution.scaleCurrentParameters}" +
                            (evolution.scaleTargetParameters?.let { " → target $it" } ?: ""),
                    )
                    Text(
                        "Tokenizer: ${evolution.tokenizerCurrentVocab} → " +
                            "${evolution.tokenizerTargetVocab} / max ${evolution.tokenizerMaxVocab}",
                    )
                    Text(
                        "Scale probe: ${yesNo(evolution.scaleCandidateGenerated)} · " +
                            "BPE probe: ${yesNo(evolution.tokenizerProbeGenerated)}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    evolution.latestResearchId?.let {
                        Text("Latest Research: $it", style = MaterialTheme.typography.bodySmall)
                    }
                }
            }

            item {
                SectionCard("Pesi del curriculum") {
                    if (evolution.domainWeights.isEmpty()) {
                        Text("Nessun peso adattivo disponibile.")
                    } else {
                        evolution.domainWeights.entries
                            .sortedByDescending { it.value }
                            .forEach {
                                Text("${it.key}: ${fmtObs(it.value)}×")
                            }
                    }
                }
            }
        }
    }
}

@Composable
fun NeuralScreen(
    state: AppUiState,
    onSelectSlot: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    val manifest = state.manifest
    val slot = manifest?.slots?.get(state.selectedSlot)
    val neural = slot?.neural
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = PaddingValues(bottom = 20.dp),
    ) {
        item {
            Row(
                modifier = Modifier.horizontalScroll(rememberScrollState()),
                horizontalArrangement = Arrangement.spacedBy(8.dp),
            ) {
                FilterChip(
                    selected = state.selectedSlot == "champion",
                    onClick = { onSelectSlot("champion") },
                    label = { Text("Champion") },
                    enabled = manifest?.slots?.containsKey("champion") == true,
                )
                FilterChip(
                    selected = state.selectedSlot == "research",
                    onClick = { onSelectSlot("research") },
                    label = { Text("Latest Research") },
                    enabled = manifest?.slots?.containsKey("research") == true,
                )
            }
        }

        if (slot == null || neural == null) {
            item {
                SectionCard("Neural View") {
                    Text(
                        "Il bundle attuale non contiene ancora diagnostica neurale. " +
                            "Arriverà al prossimo export del modello."
                    )
                }
            }
        } else {
            item {
                SectionCard("Architettura reale") {
                    val a = neural.architecture
                    Text(slot.id, fontWeight = FontWeight.Bold)
                    Text(
                        "${a.nLayers} layer · ${a.nHeads} head/layer · d_model ${a.dModel} · " +
                            "FF ${a.dFf} · ctx ${a.contextLength}"
                    )
                    Text(
                        "${a.positionEncoding} · ${a.normType} · ${a.ffVariant} · " +
                            "vocab ${a.vocabSize}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(neural.note, style = MaterialTheme.typography.bodySmall)
                    Spacer(Modifier.height(10.dp))
                    NeuralTopology(neural)
                }
            }

            item {
                SectionCard("Blocchi e pesi") {
                    neural.components.forEach { component ->
                        Text(component.label, fontWeight = FontWeight.SemiBold)
                        Text(
                            "${component.parameters} param · RMS ${fmtObs(component.rmsWeight)} · " +
                                "|w| medio ${fmtObs(component.meanAbsWeight)} · max ${fmtObs(component.maxAbsWeight)}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Spacer(Modifier.height(4.dp))
                    }
                }
            }

            item {
                SectionCard("Attention head") {
                    neural.heads.groupBy { it.layer }.toSortedMap().forEach { (layer, heads) ->
                        Text("Layer $layer", fontWeight = FontWeight.SemiBold)
                        Text(
                            heads.joinToString("  ") {
                                "H${it.head}: RMS ${fmtObs(it.rmsWeight)}"
                            },
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Spacer(Modifier.height(5.dp))
                    }
                }
            }
        }

        item {
            SectionCard("Trace ultimo messaggio") {
                val trace = state.lastInferenceTrace
                if (trace == null) {
                    Text("Invia un messaggio in Chat per vedere il flusso token.")
                } else {
                    Text(
                        "Prompt: ${trace.promptTokens} token · generati: ${trace.generatedTokenIds.size}",
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        "Token output: " + trace.generatedTokenIds.take(32).joinToString(),
                        style = MaterialTheme.typography.bodySmall,
                    )
                    if (trace.lastTopPredictions.isNotEmpty()) {
                        Spacer(Modifier.height(6.dp))
                        Text("Top logits all'ultimo passo:", fontWeight = FontWeight.SemiBold)
                        trace.lastTopPredictions.forEach {
                            Text(
                                "#${it.tokenId} '${visibleToken(it.text)}' · ${fmtObs(it.logit)}",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun NeuralTopology(neural: NeuralDiagnostics) {
    val primary = MaterialTheme.colorScheme.primary
    val secondary = MaterialTheme.colorScheme.secondary
    val outline = MaterialTheme.colorScheme.outline
    val nodes = neural.components
    val height = (nodes.size.coerceAtLeast(2) * 38).dp
    Canvas(
        modifier = Modifier
            .fillMaxWidth()
            .height(height),
    ) {
        if (nodes.isEmpty()) return@Canvas
        val x = size.width / 2f
        val step = size.height / (nodes.size + 1)
        val positions = nodes.indices.map { index ->
            Offset(x, step * (index + 1))
        }
        for (index in 0 until positions.lastIndex) {
            drawLine(
                color = outline,
                start = positions[index],
                end = positions[index + 1],
                strokeWidth = 5f,
            )
        }
        nodes.forEachIndexed { index, component ->
            val radius = 12f + (component.rmsWeight.coerceIn(0.0, 0.2) * 80.0).toFloat()
            drawCircle(
                color = if (component.id.contains("attention")) secondary else primary,
                radius = radius,
                center = positions[index],
            )
        }
    }
    nodes.forEach {
        Text("● ${it.label}", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
fun AiriPcLabScreen(
    state: AppUiState,
    modifier: Modifier = Modifier,
) {
    val report = state.manifest?.airiPcLab
    LazyColumn(
        modifier = modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
        contentPadding = PaddingValues(bottom = 20.dp),
    ) {
        item {
            SectionCard("AIRI-PC Lab copy") {
                if (report == null || report.version.isBlank()) {
                    Text(
                        "Il bundle attuale precede la Lab copy. Il prossimo ciclo H24 " +
                            "la popolerà automaticamente."
                    )
                } else {
                    Text("${report.version} · ${report.mode}", fontWeight = FontWeight.Bold)
                    Text(
                        "È una copia di ricerca read-only: nessuna shell, credenziale, " +
                            "phone control o promozione production.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        "Training verificato: ${report.trainingRows} righe · " +
                            report.trainingDomains.entries.joinToString { "${it.key}=${it.value}" },
                    )
                }
            }
        }

        if (report != null && report.version.isNotBlank()) {
            item {
                SectionCard("Capacità consentite") {
                    report.capabilities.forEach { Text("✓ $it") }
                    if (report.deniedCapabilities.isNotEmpty()) {
                        Spacer(Modifier.height(6.dp))
                        Text("Bloccate", fontWeight = FontWeight.SemiBold)
                        report.deniedCapabilities.forEach {
                            Text("✕ $it", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }

            item {
                SectionCard("Moduli della copia") {
                    report.modules.forEach { module ->
                        Text(module.name, fontWeight = FontWeight.SemiBold)
                        Text(module.path, style = MaterialTheme.typography.bodySmall)
                        if (module.symbols.isNotEmpty()) {
                            Text(
                                module.symbols.take(8).joinToString(),
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                        Spacer(Modifier.height(5.dp))
                    }
                }
            }

            item {
                SectionCard("Interazione del Champion") {
                    ProbeCard(report.champion)
                }
            }

            if (report.research != null) {
                item {
                    SectionCard("Interazione Latest Research") {
                        ProbeCard(report.research)
                    }
                }
            }
        }
    }
}

@Composable
private fun ProbeCard(probe: LabProbe?) {
    if (probe == null) {
        Text("Nessun episodio registrato.")
        return
    }
    Text(
        "Tool-call valida: ${yesNo(probe.toolCallValid)} · episodio: ${if (probe.ok) "OK" else "FAIL"}"
    )
    probe.tool?.let { Text("Tool: $it", fontWeight = FontWeight.SemiBold) }
    probe.modelSummary?.let {
        Text("Summary modello: $it", style = MaterialTheme.typography.bodySmall)
    }
    probe.error?.let {
        Text("Errore: $it", color = MaterialTheme.colorScheme.error)
    }
}

@Composable
private fun SectionCard(title: String, content: @Composable ColumnScope.() -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(16.dp),
    ) {
        Column(Modifier.padding(14.dp)) {
            Text(title, fontWeight = FontWeight.Bold)
            Spacer(Modifier.height(6.dp))
            content()
        }
    }
}

private fun formatBytes(value: Long): String {
    val bytes = value.coerceAtLeast(0)
    return when {
        bytes >= 1_000_000_000L -> String.format(Locale.US, "%.2f GB", bytes / 1_000_000_000.0)
        bytes >= 1_000_000L -> String.format(Locale.US, "%.2f MB", bytes / 1_000_000.0)
        bytes >= 1_000L -> String.format(Locale.US, "%.1f KB", bytes / 1_000.0)
        else -> "$bytes B"
    }
}

private fun fmtObs(value: Double): String =
    String.format(Locale.US, "%.5f", value)

private fun yesNo(value: Boolean): String = if (value) "sì" else "no"

private fun visibleToken(text: String): String =
    text.replace("\n", "↵").replace("\r", "␍").replace("\t", "⇥").ifEmpty { "∅" }
