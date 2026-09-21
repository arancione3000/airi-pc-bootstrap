package com.airipc.generalist

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.material3.*
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.util.Locale

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
                icon = {
                    Text(
                        when (tab) {
                            AppTab.CHAT -> "💬"
                            AppTab.LIVE -> "◉"
                            AppTab.NEURAL -> "◎"
                            AppTab.AIRI_PC -> "▣"
                        },
                        fontSize = 18.sp,
                    )
                },
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
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        item {
            SectionCard("Continuum H24") {
                val live = state.liveEvolution
                if (live == null) {
                    Text(state.liveError ?: "Carico lo stato GitHub Actions…")
                } else {
                    Text(
                        "#${live.runNumber} · ${live.stageLabel}",
                        fontWeight = FontWeight.Bold,
                        fontSize = 18.sp,
                    )
                    Text(
                        "status=${live.status}" +
                            (live.conclusion?.let { " · $it" } ?: "") +
                            " · job completati ${live.completedJobs}/${live.jobs.size}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    if (live.runningJobs.isNotEmpty()) {
                        Spacer(Modifier.height(6.dp))
                        live.runningJobs.forEach { job ->
                            Text("• ${job.name}", style = MaterialTheme.typography.bodySmall)
                        }
                    }
                    Spacer(Modifier.height(8.dp))
                    OutlinedButton(onClick = onRefreshLive) {
                        Text("Aggiorna stato live")
                    }
                }
            }
        }

        if (manifest != null && evolution != null) {
            item {
                SectionCard("Cosa sta cercando di migliorare") {
                    if (evolution.signals.isEmpty()) {
                        Text("Nessun segnale di debolezza esportato.")
                    } else {
                        Text(evolution.signals.joinToString(" · "))
                    }
                    Spacer(Modifier.height(8.dp))
                    Text("Pesi curriculum", fontWeight = FontWeight.SemiBold)
                    evolution.domainWeights
                        .toList()
                        .sortedByDescending { it.second }
                        .forEach { (domain, value) ->
                            Text("• $domain = ${format4(value)}")
                        }
                }
            }

            item {
                SectionCard("Autodata reale") {
                    val growth = evolution.dataGrowth
                    Text(
                        "${growth.totalFiles} file · ${humanBytes(growth.totalBytes)} totali",
                        fontWeight = FontWeight.Bold,
                    )
                    Text(
                        "ultimo ciclo: +${growth.addedFiles} file · +${humanBytes(growth.addedBytes)}"
                    )
                    if (growth.desiredDomains.isNotEmpty()) {
                        Spacer(Modifier.height(6.dp))
                        Text(
                            "Domini richiesti: ${growth.desiredDomains.joinToString(", ")}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                    if (growth.queries.isNotEmpty()) {
                        Spacer(Modifier.height(8.dp))
                        Text("Query di discovery", fontWeight = FontWeight.SemiBold)
                        growth.queries.forEach { Text("• $it") }
                    }
                }
            }

            if (evolution.dataGrowth.topRepositories.isNotEmpty()) {
                item {
                    SectionCard("Da dove prende i dati") {
                        evolution.dataGrowth.topRepositories.take(8).forEach { repo ->
                            Text(
                                "• ${repo.repo} · ${repo.files} file · ${humanBytes(repo.bytes)}",
                                style = MaterialTheme.typography.bodySmall,
                            )
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
                    Text("Probe scaling: ${if (evolution.scaleCandidateGenerated) "sì" else "no"}")
                    Text(
                        "BPE: ${evolution.tokenizerCurrentVocab} → ${evolution.tokenizerTargetVocab}" +
                            " / max ${evolution.tokenizerMaxVocab}"
                    )
                    Text("Probe tokenizer: ${if (evolution.tokenizerProbeGenerated) "sì" else "no"}")
                    Text(
                        "Curriculum persistito: ${evolution.curriculumStored} righe",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }
        }

        item {
            Text(
                "Live legge GitHub Actions; i pesi cambiano solo quando un ciclo finisce " +
                    "e viene esportato un bundle nuovo.",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.secondary,
            )
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
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState()),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                FilterChip(
                    selected = state.selectedSlot == "champion",
                    onClick = { onSelectSlot("champion") },
                    label = { Text("Champion") },
                )
                Spacer(Modifier.width(8.dp))
                FilterChip(
                    selected = state.selectedSlot == "research",
                    onClick = { onSelectSlot("research") },
                    label = { Text("Latest Research") },
                    enabled = manifest?.slots?.containsKey("research") == true,
                )
            }
        }

        if (neural == null) {
            item {
                SectionCard("Neural View") {
                    Text("La telemetria neurale comparirà al prossimo export mobile.")
                }
            }
        } else {
            item {
                SectionCard("Architettura reale") {
                    val a = neural.architecture
                    Text(a.type, fontWeight = FontWeight.Bold)
                    Text("${a.nLayers} layer · ${a.nHeads} head/layer · d_model ${a.dModel} · d_ff ${a.dFf}")
                    Text("head_dim ${a.headDim} · ctx ${a.contextLength} · vocab ${a.vocabSize}")
                    Text(
                        "${a.normType} · ${a.positionEncoding} · ${a.ffVariant}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            item {
                SectionCard("Flusso interno aggregato") {
                    NeuralFlow(neural)
                    Spacer(Modifier.height(8.dp))
                    Text(
                        neural.note,
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.secondary,
                    )
                }
            }

            items(neural.components) { component ->
                SectionCard(component.label) {
                    Text("${component.parameters} parametri")
                    Text(
                        "mean|w| ${format6(component.meanAbsWeight)} · " +
                            "RMS ${format6(component.rmsWeight)} · " +
                            "max|w| ${format6(component.maxAbsWeight)}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            item {
                SectionCard("Attention heads") {
                    neural.heads
                        .groupBy { it.layer }
                        .toSortedMap()
                        .forEach { (layer, heads) ->
                            Text("Layer $layer", fontWeight = FontWeight.SemiBold)
                            Text(
                                heads.joinToString("  ") {
                                    "H${it.head}:${format4(it.meanAbsWeight)}"
                                },
                                style = MaterialTheme.typography.bodySmall,
                            )
                        }
                }
            }

            state.lastInferenceTrace?.let { trace ->
                item {
                    SectionCard("Ultima inferenza locale") {
                        Text("prompt ${trace.promptTokens} token · generati ${trace.generatedTokenIds.size}")
                        Text(
                            "token IDs: ${trace.generatedTokenIds.take(32).joinToString(", ")}",
                            style = MaterialTheme.typography.bodySmall,
                        )
                        if (trace.lastTopPredictions.isNotEmpty()) {
                            Spacer(Modifier.height(6.dp))
                            Text("Top logits ultimo step", fontWeight = FontWeight.SemiBold)
                            trace.lastTopPredictions.forEach { pred ->
                                Text(
                                    "#${pred.tokenId} '${visibleToken(pred.text)}' → ${format4(pred.logit)}",
                                    style = MaterialTheme.typography.bodySmall,
                                )
                            }
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun NeuralFlow(neural: NeuralDiagnostics) {
    val count = neural.components.size.coerceAtLeast(1)
    val width = 760.dp
    val height = 145.dp
    val primary = MaterialTheme.colorScheme.primary
    val secondary = MaterialTheme.colorScheme.secondary
    Column(Modifier.horizontalScroll(rememberScrollState())) {
        Canvas(
            modifier = Modifier
                .width(width)
                .height(height),
        ) {
            val usableWidth = size.width - 60f
            val step = if (count <= 1) usableWidth else usableWidth / (count - 1)
            val y = size.height / 2f
            neural.components.forEachIndexed { index, component ->
                val x = 30f + index * step
                if (index > 0) {
                    val prev = 30f + (index - 1) * step
                    drawLine(
                        color = secondary.copy(alpha = 0.5f),
                        start = Offset(prev, y),
                        end = Offset(x, y),
                        strokeWidth = 5f,
                    )
                }
                val magnitude = component.rmsWeight.coerceAtLeast(0.000001)
                val radius = (
                    10.0 + magnitude.coerceAtMost(0.08) * 260.0
                ).coerceIn(11.0, 28.0).toFloat()
                drawCircle(
                    color = primary.copy(alpha = 0.82f),
                    radius = radius,
                    center = Offset(x, y),
                )
            }
        }
        Text(
            neural.components.joinToString(" → ") { it.label },
            style = MaterialTheme.typography.labelSmall,
        )
    }
}

@Composable
fun AiriPcLabScreen(state: AppUiState) {
    val report = state.manifest?.airiPcLab
    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        if (report == null || report.version.isBlank()) {
            item {
                SectionCard("Airi-PC Lab") {
                    Text("Il prossimo ciclo H24 creerà la prima copia-lab e il relativo report.")
                }
            }
        } else {
            item {
                SectionCard("Sandbox AIRI-PC") {
                    Text(report.mode, fontWeight = FontWeight.Bold)
                    Text(
                        "Training verificato: ${report.trainingRows} righe · ${report.trainingDomains}",
                        style = MaterialTheme.typography.bodySmall,
                    )
                    Spacer(Modifier.height(6.dp))
                    Text("Consentito", fontWeight = FontWeight.SemiBold)
                    Text(report.capabilities.joinToString(" · "))
                    Spacer(Modifier.height(6.dp))
                    Text("Bloccato", fontWeight = FontWeight.SemiBold)
                    Text(
                        report.deniedCapabilities.joinToString(" · "),
                        color = MaterialTheme.colorScheme.secondary,
                    )
                }
            }

            item {
                SectionCard("Ultimo tentativo del Champion") {
                    ProbeContent(report.champion)
                }
            }

            report.research?.let { research ->
                item {
                    SectionCard("Ultimo tentativo Latest Research") {
                        ProbeContent(research)
                    }
                }
            }

            items(report.modules) { module ->
                SectionCard(module.name) {
                    Text(module.path, style = MaterialTheme.typography.labelSmall)
                    if (module.symbols.isNotEmpty()) {
                        Spacer(Modifier.height(4.dp))
                        Text(
                            module.symbols.joinToString(", "),
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }

            item {
                Text(
                    "Copia read-only del Control Plane: il modello può imparare struttura/tool-use, " +
                        "ma non può comandare il tuo PC o telefono da qui.",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.secondary,
                )
            }
        }
    }
}

@Composable
private fun ProbeContent(probe: LabProbe?) {
    if (probe == null) {
        Text("Nessun probe ancora eseguito.")
        return
    }
    Text(
        if (probe.ok) "Tool-call valido ✅" else "Tool-call non riuscito",
        fontWeight = FontWeight.Bold,
    )
    probe.tool?.let { Text("tool: $it") }
    probe.modelSummary?.let {
        Spacer(Modifier.height(4.dp))
        Text(it)
    }
    probe.error?.let {
        Spacer(Modifier.height(4.dp))
        Text(it, color = MaterialTheme.colorScheme.error)
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
            verticalArrangement = Arrangement.spacedBy(2.dp),
        ) {
            Text(title, fontWeight = FontWeight.Bold, fontSize = 16.sp)
            Spacer(Modifier.height(4.dp))
            content()
        }
    }
}

private fun format4(value: Double): String =
    String.format(Locale.US, "%.4f", value)

private fun format6(value: Double): String =
    String.format(Locale.US, "%.6f", value)

private fun humanBytes(value: Long): String {
    val units = arrayOf("B", "KB", "MB", "GB")
    var amount = value.toDouble()
    var unit = 0
    while (amount >= 1024.0 && unit < units.lastIndex) {
        amount /= 1024.0
        unit += 1
    }
    return String.format(Locale.US, "%.1f %s", amount, units[unit])
}

private fun visibleToken(text: String): String =
    text
        .replace("\n", "↵")
        .replace("\r", "")
        .replace("\t", "⇥")
        .ifEmpty { "∅" }
