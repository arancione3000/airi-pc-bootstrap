package com.airipc.generalist

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import java.util.Locale
import kotlin.math.max

enum class AppTab(val label: String) {
    CHAT("Chat"),
    LIVE("Live"),
    NEURAL("Neural"),
    AIRI_PC("Airi-PC"),
}

@Composable
fun GeneralistBottomBar(
    selected: AppTab,
    onSelect: (AppTab) -> Unit,
) {
    NavigationBar {
        AppTab.entries.forEach { tab ->
            NavigationBarItem(
                selected = selected == tab,
                onClick = { onSelect(tab) },
                icon = { Text(tab.label.take(1), fontWeight = FontWeight.Bold) },
                label = { Text(tab.label) },
            )
        }
    }
}

@Composable
fun LiveEvolutionScreen(
    state: AppUiState,
    onRefreshLive: () -> Unit,
) {
    val manifest = state.manifest
    val evolution = manifest?.evolution
    val live = state.liveEvolution
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            SectionCard("Continuum H24") {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Column(Modifier.weight(1f)) {
                        Text(
                            live?.stageLabel ?: "Caricamento stato GitHub…",
                            fontWeight = FontWeight.Bold,
                        )
                        if (live != null) {
                            Text(
                                "run #${live.runNumber} · ${live.status}" +
                                    (live.conclusion?.let { " · $it" } ?: ""),
                                style = MaterialTheme.typography.bodySmall,
                            )
                            Text(
                                "${live.completedJobs}/${live.jobs.size} job completati",
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                    }
                    TextButton(onClick = onRefreshLive) { Text("Refresh") }
                }
                state.liveError?.let {
                    Text(it, color = MaterialTheme.colorScheme.error)
                }
            }
        }

        if (manifest != null && evolution != null) {
            item {
                SectionCard("Stato evolutivo") {
                    Text("Bundle ciclo ${manifest.cycle} · ${manifest.generalistVersion}")
                    Text(
                        "Segnali: " + evolution.signals.ifEmpty { listOf("nessuno") }.joinToString(),
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        "Curriculum: ${evolution.curriculumStored} righe · " +
                            evolution.curriculumDomains.entries.joinToString { "${it.key}=${it.value}" },
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        "Scaling: ${evolution.scaleCurrentParameters} → " +
                            (evolution.scaleTargetParameters?.toString() ?: "nessun target") +
                            " · probe=${evolution.scaleCandidateGenerated}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        "Tokenizer: ${evolution.tokenizerCurrentVocab} → " +
                            "${evolution.tokenizerTargetVocab}/${evolution.tokenizerMaxVocab} · " +
                            "probe=${evolution.tokenizerProbeGenerated}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            item {
                SectionCard("Autodata · cosa cerca") {
                    val growth = evolution.dataGrowth
                    Text(
                        "Corpus ${humanBytes(growth.totalBytes)} · ${growth.totalFiles} file",
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        "Ultimo ciclo: +${growth.addedFiles} file / +${humanBytes(growth.addedBytes)}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    if (growth.desiredDomains.isNotEmpty()) {
                        Text(
                            "Priorità: ${growth.desiredDomains.joinToString()}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    if (growth.queries.isNotEmpty()) {
                        Spacer(Modifier.height(6.dp))
                        Text("Query GitHub reali", fontWeight = FontWeight.SemiBold)
                        growth.queries.forEach { Text("• $it", style = MaterialTheme.typography.bodySmall) }
                    }
                    if (growth.domainFiles.isNotEmpty()) {
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "Domini: " + growth.domainFiles.entries.joinToString { "${it.key}=${it.value}" },
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }

            if (evolution.dataGrowth.topRepositories.isNotEmpty()) {
                item {
                    SectionCard("Dati ammessi nel corpus") {
                        evolution.dataGrowth.topRepositories.take(8).forEach { source ->
                            Text(source.repo, fontWeight = FontWeight.SemiBold)
                            Text(
                                "${source.files} file · ${humanBytes(source.bytes)} · " +
                                    source.domains.entries.joinToString { "${it.key}:${it.value}" },
                                style = MaterialTheme.typography.bodySmall,
                            )
                            Spacer(Modifier.height(4.dp))
                        }
                    }
                }
            }

            if (live != null) {
                item {
                    SectionCard("Pipeline del ciclo corrente") {
                        live.jobs.forEach { job ->
                            Text(
                                "• ${job.name}: ${job.status}" +
                                    (job.conclusion?.let { " / $it" } ?: ""),
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
fun NeuralScreen(
    state: AppUiState,
    onSelectSlot: (String) -> Unit,
) {
    val manifest = state.manifest
    val slot = manifest?.slots?.get(state.selectedSlot)
    val neural = slot?.neural
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            SectionCard("Neural View · checkpoint reale") {
                Row {
                    FilterChip(
                        selected = state.selectedSlot == "champion",
                        onClick = { onSelectSlot("champion") },
                        label = { Text("Champion") },
                    )
                    Spacer(Modifier.width(8.dp))
                    FilterChip(
                        selected = state.selectedSlot == "research",
                        onClick = { onSelectSlot("research") },
                        enabled = manifest?.slots?.containsKey("research") == true,
                        label = { Text("Research") },
                    )
                }
                Text(
                    neural?.note ?: "Le statistiche neurali arriveranno col prossimo bundle esportato.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        if (neural != null) {
            item {
                val a = neural.architecture
                SectionCard("Architettura") {
                    Text(a.type.replace("_", " "), fontWeight = FontWeight.Bold)
                    Text(
                        "${a.nLayers} layer · ${a.nHeads} head/layer · head dim ${a.headDim}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        "d_model ${a.dModel} · FFN ${a.dFf} · ctx ${a.contextLength} · vocab ${a.vocabSize}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Text(
                        "${a.positionEncoding} · ${a.normType} · ${a.ffVariant}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            item {
                SectionCard("Flusso delle connessioni") {
                    NeuralFlowDiagram(neural)
                    Text(
                        "Le linee mostrano il flusso aggregato embedding → attention/norm/FFN → output. " +
                            "Non fingono di visualizzare ogni singolo neurone.",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            item {
                SectionCard("Blocchi e pesi") {
                    neural.components.forEach { component ->
                        Text(component.label, fontWeight = FontWeight.SemiBold)
                        Text(
                            "${component.parameters} param · |w| medio ${f6(component.meanAbsWeight)} · " +
                                "RMS ${f6(component.rmsWeight)} · max ${f6(component.maxAbsWeight)}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        Spacer(Modifier.height(4.dp))
                    }
                }
            }

            item {
                SectionCard("Attention heads") {
                    val scroll = rememberScrollState()
                    Row(
                        Modifier.fillMaxWidth().horizontalScroll(scroll),
                        horizontalArrangement = Arrangement.spacedBy(8.dp),
                    ) {
                        neural.heads.forEach { head ->
                            Surface(shape = RoundedCornerShape(12.dp), tonalElevation = 1.dp) {
                                Column(Modifier.padding(10.dp).width(130.dp)) {
                                    Text("L${head.layer} · H${head.head}", fontWeight = FontWeight.Bold)
                                    Text("|w| ${f6(head.meanAbsWeight)}", style = MaterialTheme.typography.bodySmall)
                                    Text("RMS ${f6(head.rmsWeight)}", style = MaterialTheme.typography.bodySmall)
                                    Text("max ${f6(head.maxAbsWeight)}", style = MaterialTheme.typography.bodySmall)
                                }
                            }
                        }
                    }
                }
            }
        }

        state.lastInferenceTrace?.let { trace ->
            item {
                SectionCard("Ultima inferenza locale") {
                    Text(
                        "Prompt ${trace.promptTokens} token · generati ${trace.generatedTokenIds.size}",
                        fontWeight = FontWeight.SemiBold,
                    )
                    Text(
                        "Token generati: ${trace.generatedTokenIds.take(48).joinToString()}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    if (trace.lastTopPredictions.isNotEmpty()) {
                        Spacer(Modifier.height(6.dp))
                        Text("Top logits all'ultimo step", fontWeight = FontWeight.SemiBold)
                        trace.lastTopPredictions.forEach { pred ->
                            val printable = pred.text.replace("\n", "↵").ifEmpty { "∅" }
                            Text(
                                "#${pred.tokenId} '$printable' · ${f4(pred.logit)}",
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
private fun NeuralFlowDiagram(neural: NeuralDiagnostics) {
    val primary = MaterialTheme.colorScheme.primary
    val surface = MaterialTheme.colorScheme.surfaceVariant
    val components = neural.components.take(18)
    val height = max(180, components.size * 34)
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .height(height.dp)
            .background(surface, RoundedCornerShape(12.dp)),
    ) {
        Canvas(Modifier.fillMaxSize().padding(12.dp)) {
            if (components.isEmpty()) return@Canvas
            val x = size.width * 0.18f
            val endX = size.width * 0.82f
            val step = size.height / max(1, components.size).toFloat()
            components.forEachIndexed { index, _ ->
                val y = step * (index + 0.5f)
                if (index + 1 < components.size) {
                    val nextY = step * (index + 1.5f)
                    drawLine(
                        color = primary.copy(alpha = 0.5f),
                        start = Offset(x, y),
                        end = Offset(endX, nextY),
                        strokeWidth = 3f,
                    )
                }
                drawCircle(primary, radius = 9f, center = Offset(x, y))
                drawCircle(primary.copy(alpha = 0.55f), radius = 7f, center = Offset(endX, y))
            }
        }
        Column(
            Modifier.fillMaxSize().padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.SpaceEvenly,
        ) {
            components.forEach { component ->
                Text(
                    component.label,
                    style = MaterialTheme.typography.labelSmall,
                    modifier = Modifier.align(Alignment.CenterHorizontally),
                )
            }
        }
    }
}

@Composable
fun AiriPcLabScreen(state: AppUiState) {
    val lab = state.manifest?.airiPcLab
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            SectionCard("Airi-PC Lab") {
                Text(
                    if (lab?.mode == "read_only_sandbox") {
                        "Copia sandboxata e read-only di Airi-PC"
                    } else {
                        "Il report completo apparirà dopo il prossimo ciclo H24."
                    },
                    fontWeight = FontWeight.Bold,
                )
                Text(
                    "Il modello può ispezionare moduli allowlist, provare tool-call e imparare dagli episodi verificati. " +
                        "Non ha shell, credenziali, controllo telefono o auto-promozione.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
        }

        if (lab != null) {
            item {
                SectionCard("Capacità consentite / negate") {
                    Text("Consentite", fontWeight = FontWeight.SemiBold)
                    lab.capabilities.forEach { Text("✓ $it", style = MaterialTheme.typography.bodySmall) }
                    Spacer(Modifier.height(6.dp))
                    Text("Negate", fontWeight = FontWeight.SemiBold)
                    lab.deniedCapabilities.forEach { Text("× $it", style = MaterialTheme.typography.bodySmall) }
                }
            }

            item {
                SectionCard("Apprendimento dal Lab") {
                    Text("${lab.trainingRows} esempi lab nel curriculum")
                    Text(
                        lab.trainingDomains.entries.joinToString { "${it.key}=${it.value}" },
                        style = MaterialTheme.typography.bodySmall,
                    )
                    LabProbeView("Champion", lab.champion)
                    LabProbeView("Latest Research", lab.research)
                }
            }

            if (lab.modules.isNotEmpty()) {
                item {
                    SectionCard("Copia Airi-PC esposta al modello") {
                        lab.modules.forEach { module ->
                            Text(module.name, fontWeight = FontWeight.SemiBold)
                            Text(module.path, style = MaterialTheme.typography.labelSmall)
                            if (module.symbols.isNotEmpty()) {
                                Text(
                                    module.symbols.take(8).joinToString(),
                                    style = MaterialTheme.typography.bodySmall,
                                )
                            }
                            Spacer(Modifier.height(6.dp))
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun LabProbeView(label: String, probe: LabProbe?) {
    Spacer(Modifier.height(8.dp))
    Text(label, fontWeight = FontWeight.SemiBold)
    if (probe == null) {
        Text("Nessun episodio ancora.", style = MaterialTheme.typography.bodySmall)
        return
    }
    Text(
        if (probe.ok) {
            "tool-call valida=${probe.toolCallValid} · tool=${probe.tool ?: "?"}"
        } else {
            "probe fallita · ${probe.error ?: "nessun dettaglio"}"
        },
        style = MaterialTheme.typography.bodySmall,
    )
    probe.modelSummary?.let {
        Text("Riassunto modello: $it", style = MaterialTheme.typography.bodySmall)
    }
}

@Composable
private fun SectionCard(
    title: String,
    content: @Composable ColumnScope.() -> Unit,
) {
    Card(Modifier.fillMaxWidth()) {
        Column(
            Modifier.padding(12.dp),
            verticalArrangement = Arrangement.spacedBy(3.dp),
        ) {
            Text(title, fontWeight = FontWeight.Bold)
            content()
        }
    }
}

private fun humanBytes(value: Long): String {
    if (value < 1024) return "$value B"
    val kb = value / 1024.0
    if (kb < 1024) return String.format(Locale.US, "%.1f KB", kb)
    val mb = kb / 1024.0
    return String.format(Locale.US, "%.2f MB", mb)
}

private fun f4(value: Double): String = String.format(Locale.US, "%.4f", value)
private fun f6(value: Double): String = String.format(Locale.US, "%.6f", value)
