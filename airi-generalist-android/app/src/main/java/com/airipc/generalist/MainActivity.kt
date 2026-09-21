package com.airipc.generalist

import android.content.Context
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import java.io.Closeable
import java.util.Locale

data class ChatLine(
    val role: String,
    val content: String,
    val modelId: String = "",
    val elapsedMs: Long = 0L,
    val generatedTokens: Int = 0,
    val repetitionRate: Double = 0.0,
    val meanEntropy: Double = 0.0,
    val decodeMode: DecodeMode = DecodeMode.GREEDY,
)

data class ModelComparison(
    val prompt: String,
    val champion: InferenceTrace,
    val research: InferenceTrace,
    val championMs: Long,
    val researchMs: Long,
)

data class AppUiState(
    val manifest: MobileManifest? = null,
    val selectedSlot: String = "champion",
    val selectedTab: AppTab = AppTab.CHAT,
    val liveEvolution: LiveEvolutionSnapshot? = null,
    val liveError: String? = null,
    val lastInferenceTrace: InferenceTrace? = null,
    val loadedModelId: String = "",
    val decodeMode: DecodeMode = DecodeMode.GREEDY,
    val temperature: Double = 0.8,
    val topP: Double = 0.9,
    val topK: Int = 40,
    val repetitionPenalty: Double = 1.08,
    val comparison: ModelComparison? = null,
    val comparing: Boolean = false,
    val messages: List<ChatLine> = emptyList(),
    val loadingModel: Boolean = true,
    val generating: Boolean = false,
    val freshFromNetwork: Boolean = true,
    val status: String = "Connessione al bundle mobile…",
    val error: String? = null,
)

class GeneralistController(context: Context) : Closeable {
    private val repository = BundleRepository(context)
    private val liveRepository = LiveEvolutionRepository()
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)
    private val refreshMutex = Mutex()
    private val engineMutex = Mutex()
    private var engine: AiriOnnxEngine? = null
    private var engineKey: String = ""
    private var started = false

    private val _state = MutableStateFlow(AppUiState())
    val state: StateFlow<AppUiState> = _state

    fun start() {
        if (started) return
        started = true
        scope.launch { refresh() }
        scope.launch { refreshLiveInternal() }
        scope.launch {
            while (isActive) {
                delay(120_000L)
                refresh(silent = true)
            }
        }
        scope.launch {
            while (isActive) {
                delay(300_000L)
                refreshLiveInternal()
            }
        }
    }

    fun refresh(silent: Boolean = false) {
        scope.launch { refreshInternal(silent) }
    }

    fun selectTab(tab: AppTab) {
        _state.value = _state.value.copy(selectedTab = tab)
    }

    fun refreshLive() {
        scope.launch { refreshLiveInternal() }
    }

    fun selectSlot(slot: String) {
        if (slot == _state.value.selectedSlot) return
        _state.value = _state.value.copy(
            selectedSlot = slot,
            loadingModel = true,
            error = null,
            status = "Carico ${labelFor(slot)}…",
        )
        scope.launch { refreshInternal(silent = false) }
    }

    fun clearChat() {
        _state.value = _state.value.copy(messages = emptyList(), comparison = null)
    }

    fun setDecodeMode(mode: DecodeMode) {
        _state.value = _state.value.copy(decodeMode = mode)
    }

    fun setTemperature(value: Double) {
        _state.value = _state.value.copy(temperature = value.coerceIn(0.05, 2.0))
    }

    fun setTopP(value: Double) {
        _state.value = _state.value.copy(topP = value.coerceIn(0.05, 1.0))
    }

    fun setTopK(value: Int) {
        _state.value = _state.value.copy(topK = value.coerceIn(1, 100))
    }

    fun setRepetitionPenalty(value: Double) {
        _state.value = _state.value.copy(repetitionPenalty = value.coerceIn(1.0, 2.0))
    }

    private fun decodeSettings(state: AppUiState = _state.value) = DecodeSettings(
        mode = state.decodeMode,
        temperature = state.temperature,
        topP = state.topP,
        topK = state.topK,
        repetitionPenalty = state.repetitionPenalty,
    )

    fun compare(text: String) {
        val prompt = text.trim()
        val manifest = _state.value.manifest ?: return
        val championSlot = manifest.slots["champion"] ?: return
        val researchSlot = manifest.slots["research"] ?: return
        if (prompt.isEmpty() || _state.value.generating || _state.value.comparing) return
        val settings = decodeSettings()
        _state.value = _state.value.copy(comparing = true, comparison = null, error = null)
        scope.launch {
            try {
                val championBundle = repository.ensureBundle(championSlot)
                val researchBundle = repository.ensureBundle(researchSlot)

                val championStarted = SystemClock.elapsedRealtime()
                val championTrace = withContext(Dispatchers.Default) {
                    AiriOnnxEngine(championBundle).use { candidate ->
                        candidate.chat(
                            listOf(ChatMessage("user", prompt)),
                            maxNewTokens = 64,
                            settings = settings,
                        )
                    }
                }
                val championMs = SystemClock.elapsedRealtime() - championStarted

                val researchStarted = SystemClock.elapsedRealtime()
                val researchTrace = withContext(Dispatchers.Default) {
                    AiriOnnxEngine(researchBundle).use { candidate ->
                        candidate.chat(
                            listOf(ChatMessage("user", prompt)),
                            maxNewTokens = 64,
                            settings = settings,
                        )
                    }
                }
                val researchMs = SystemClock.elapsedRealtime() - researchStarted

                _state.value = _state.value.copy(
                    comparing = false,
                    comparison = ModelComparison(
                        prompt = prompt,
                        champion = championTrace,
                        research = researchTrace,
                        championMs = championMs,
                        researchMs = researchMs,
                    ),
                )
            } catch (exc: Exception) {
                _state.value = _state.value.copy(
                    comparing = false,
                    error = "Confronto fallito: ${exc.message}",
                )
            }
        }
    }

    fun send(text: String) {
        val content = text.trim()
        if (content.isEmpty() || _state.value.generating || engine == null) return

        val userLine = ChatLine("user", content)
        _state.value = _state.value.copy(
            messages = _state.value.messages + userLine,
            generating = true,
            error = null,
        )

        scope.launch {
            val history = _state.value.messages
                .filter { it.role == "user" || (it.role == "assistant" && it.content.isNotEmpty()) }
                .takeLast(10)
                .map { ChatMessage(it.role, it.content) }
            val modelId = _state.value.loadedModelId
            val settings = decodeSettings()
            val startedAt = SystemClock.elapsedRealtime()
            try {
                val trace = engineMutex.withLock {
                    val active = engine ?: error("Modello non caricato")
                    withContext(Dispatchers.Default) {
                        active.chat(history, maxNewTokens = 64, settings = settings)
                    }
                }
                val elapsed = SystemClock.elapsedRealtime() - startedAt
                Log.i(
                    "AiriGeneralistLab",
                    "AIRI_GENERALIST_INFERENCE=PASS model=$modelId chars=${trace.text.length} elapsed_ms=$elapsed",
                )
                _state.value = _state.value.copy(
                    messages = _state.value.messages + ChatLine(
                        role = "assistant",
                        content = trace.text,
                        modelId = modelId,
                        elapsedMs = elapsed,
                        generatedTokens = trace.generatedTokenIds.size,
                        repetitionRate = trace.repetitionRate,
                        meanEntropy = trace.meanEntropy,
                        decodeMode = settings.mode,
                    ),
                    generating = false,
                    lastInferenceTrace = trace,
                )
            } catch (exc: Exception) {
                _state.value = _state.value.copy(
                    generating = false,
                    error = "Inferenza fallita: ${exc.message}",
                )
            }
        }
    }

    private suspend fun refreshInternal(silent: Boolean) {
        refreshMutex.withLock {
            if (!silent) {
                _state.value = _state.value.copy(
                    loadingModel = true,
                    error = null,
                    status = "Controllo aggiornamenti su GitHub…",
                )
            }
            try {
                val fetched = repository.fetchManifest()
                var slotName = _state.value.selectedSlot
                if (!fetched.manifest.slots.containsKey(slotName)) {
                    slotName = "champion"
                }
                val slot = fetched.manifest.slots[slotName]
                    ?: error("Bundle champion non disponibile")
                val key = slot.files["model.onnx"]?.sha256
                    ?: error("Hash modello mancante")

                if (key != engineKey) {
                    if (!silent) {
                        _state.value = _state.value.copy(
                            status = "Scarico ${labelFor(slotName)} ciclo ${fetched.manifest.cycle}…"
                        )
                    }
                    val installed = repository.ensureBundle(slot)
                    val replacement = withContext(Dispatchers.Default) {
                        AiriOnnxEngine(installed)
                    }
                    engineMutex.withLock {
                        engine?.close()
                        engine = replacement
                        engineKey = key
                    }
                }

                _state.value = _state.value.copy(
                    manifest = fetched.manifest,
                    selectedSlot = slotName,
                    loadedModelId = slot.id,
                    loadingModel = false,
                    freshFromNetwork = fetched.freshFromNetwork,
                    status = if (fetched.freshFromNetwork) {
                        "${labelFor(slotName)} pronto · ciclo ${fetched.manifest.cycle}"
                    } else {
                        "${labelFor(slotName)} pronto · cache offline"
                    },
                    error = null,
                )
            } catch (exc: Exception) {
                _state.value = _state.value.copy(
                    loadingModel = false,
                    error = "Aggiornamento modello fallito: ${exc.message}",
                    status = if (engine != null) "Uso il modello già installato" else "Modello non disponibile",
                )
            }
        }
    }

    private suspend fun refreshLiveInternal() {
        try {
            val live = liveRepository.fetch()
            _state.value = _state.value.copy(
                liveEvolution = live,
                liveError = null,
            )
        } catch (exc: Exception) {
            _state.value = _state.value.copy(
                liveError = "Live GitHub: ${exc.message}",
            )
        }
    }

    private fun labelFor(slot: String) =
        if (slot == "research") "Latest Research" else "Champion"

    override fun close() {
        scope.cancel()
        runBlocking {
            engineMutex.withLock {
                engine?.close()
                engine = null
            }
        }
    }
}

class MainActivity : ComponentActivity() {
    private lateinit var controller: GeneralistController

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        controller = GeneralistController(applicationContext)
        ModelUpdateWorker.schedule(applicationContext)
        setContent {
            val state by controller.state.collectAsState()
            LaunchedEffect(Unit) { controller.start() }
            AiriGeneralistApp(
                state = state,
                onRefresh = { controller.refresh() },
                onRefreshLive = controller::refreshLive,
                onSelectTab = controller::selectTab,
                onSelectSlot = controller::selectSlot,
                onClear = controller::clearChat,
                onSend = controller::send,
                onCompare = controller::compare,
                onDecodeMode = controller::setDecodeMode,
                onTemperature = controller::setTemperature,
                onTopP = controller::setTopP,
                onTopK = controller::setTopK,
                onRepetitionPenalty = controller::setRepetitionPenalty,
            )
        }
    }

    override fun onDestroy() {
        controller.close()
        super.onDestroy()
    }
}

@Composable
private fun AiriGeneralistApp(
    state: AppUiState,
    onRefresh: () -> Unit,
    onRefreshLive: () -> Unit,
    onSelectTab: (AppTab) -> Unit,
    onSelectSlot: (String) -> Unit,
    onClear: () -> Unit,
    onSend: (String) -> Unit,
    onCompare: (String) -> Unit,
    onDecodeMode: (DecodeMode) -> Unit,
    onTemperature: (Double) -> Unit,
    onTopP: (Double) -> Unit,
    onTopK: (Int) -> Unit,
    onRepetitionPenalty: (Double) -> Unit,
) {
    val scheme = lightColorScheme(
        primary = Color(0xFFE86F17),
        secondary = Color(0xFF8A4F22),
        surface = Color(0xFFFFFBF7),
        background = Color(0xFFFFF7F0),
    )
    MaterialTheme(colorScheme = scheme) {
        Scaffold(
            topBar = {
                Surface(
                    shadowElevation = 2.dp,
                    modifier = Modifier.windowInsetsPadding(WindowInsets.statusBars),
                ) {
                    Row(
                        modifier = Modifier
                            .fillMaxWidth()
                            .padding(horizontal = 16.dp, vertical = 12.dp),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        Column(Modifier.weight(1f)) {
                            Text("AIRI Generalist Lab", fontWeight = FontWeight.Bold, fontSize = 20.sp)
                            Text(state.status, style = MaterialTheme.typography.bodySmall)
                        }
                        TextButton(onClick = onRefresh, enabled = !state.loadingModel) {
                            Text("Aggiorna")
                        }
                    }
                }
            },
            bottomBar = {
                GeneralistBottomBar(
                    selected = state.selectedTab,
                    onSelect = onSelectTab,
                )
            },
        ) { padding ->
            Box(
                modifier = Modifier
                    .padding(padding)
                    .fillMaxSize()
                    .background(MaterialTheme.colorScheme.background)
                    .padding(12.dp),
            ) {
                when (state.selectedTab) {
                    AppTab.CHAT -> Column(Modifier.fillMaxSize()) {
                        ModelPanel(state, onSelectSlot, onClear)
                        Spacer(Modifier.height(8.dp))
                        DecodePanel(
                            state = state,
                            onDecodeMode = onDecodeMode,
                            onTemperature = onTemperature,
                            onTopP = onTopP,
                            onTopK = onTopK,
                            onRepetitionPenalty = onRepetitionPenalty,
                        )
                        Spacer(Modifier.height(8.dp))
                        ChatPanel(
                            state = state,
                            onSend = onSend,
                            onCompare = onCompare,
                            modifier = Modifier.weight(1f),
                        )
                    }
                    AppTab.LIVE -> LiveEvolutionScreen(
                        state = state,
                        onRefreshLive = onRefreshLive,
                    )
                    AppTab.NEURAL -> NeuralScreen(
                        state = state,
                        onSelectSlot = onSelectSlot,
                    )
                    AppTab.AIRI_PC -> AiriPcLabScreen(state = state)
                }
            }
        }
    }
}

@Composable
private fun ModelPanel(
    state: AppUiState,
    onSelectSlot: (String) -> Unit,
    onClear: () -> Unit,
) {
    val manifest = state.manifest
    val selected = manifest?.slots?.get(state.selectedSlot)
    Card {
        Column(Modifier.padding(12.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                FilterChip(
                    selected = state.selectedSlot == "champion",
                    onClick = { onSelectSlot("champion") },
                    label = { Text("Champion") },
                    enabled = manifest?.slots?.containsKey("champion") != false,
                )
                Spacer(Modifier.width(8.dp))
                FilterChip(
                    selected = state.selectedSlot == "research",
                    onClick = { onSelectSlot("research") },
                    label = { Text("Latest Research") },
                    enabled = manifest?.slots?.containsKey("research") == true,
                )
                Spacer(Modifier.weight(1f))
                TextButton(onClick = onClear) { Text("Pulisci chat") }
            }

            if (selected != null) {
                Spacer(Modifier.height(6.dp))
                Text(
                    selected.id,
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = FontWeight.SemiBold,
                )
                Text(
                    "ciclo ${manifest.cycle} · ${selected.parameters} param · BPE ${selected.tokenizerVocabSize} · ctx ${selected.contextLength}",
                    style = MaterialTheme.typography.bodySmall,
                )
                Text(
                    "NLL ${fmt(selected.nllPerByte)} · similarity ${fmt(selected.generationSimilarity)} · exact ${fmt(selected.generationExactAccuracy)}",
                    style = MaterialTheme.typography.bodySmall,
                )
                if (selected.researchOnly) {
                    Text(
                        "Research-only: non ha superato tutti i gate di promozione.",
                        color = MaterialTheme.colorScheme.secondary,
                        style = MaterialTheme.typography.bodySmall,
                    )
                }
            }

            if (!state.freshFromNetwork && manifest != null) {
                Text(
                    "Offline: sto usando il manifest salvato sul telefono.",
                    style = MaterialTheme.typography.bodySmall,
                )
            }
            state.error?.let {
                Spacer(Modifier.height(4.dp))
                Text(it, color = MaterialTheme.colorScheme.error, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

@Composable
private fun DecodePanel(
    state: AppUiState,
    onDecodeMode: (DecodeMode) -> Unit,
    onTemperature: (Double) -> Unit,
    onTopP: (Double) -> Unit,
    onTopK: (Int) -> Unit,
    onRepetitionPenalty: (Double) -> Unit,
) {
    val sampling = state.decodeMode == DecodeMode.SAMPLING
    Card {
        Column(Modifier.padding(10.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                FilterChip(
                    selected = state.decodeMode == DecodeMode.GREEDY,
                    onClick = { onDecodeMode(DecodeMode.GREEDY) },
                    label = { Text("Greedy") },
                )
                Spacer(Modifier.width(8.dp))
                FilterChip(
                    selected = sampling,
                    onClick = { onDecodeMode(DecodeMode.SAMPLING) },
                    label = { Text("Sampling") },
                )
                Spacer(Modifier.weight(1f))
                Text(
                    if (sampling) "configurabile" else "RAW deterministico",
                    style = MaterialTheme.typography.labelSmall,
                )
            }
            if (sampling) {
                Text("temperature ${fmt(state.temperature)}", style = MaterialTheme.typography.bodySmall)
                Slider(
                    value = state.temperature.toFloat(),
                    onValueChange = { onTemperature(it.toDouble()) },
                    valueRange = 0.1f..1.5f,
                )
                Text("top-p ${fmt(state.topP)}", style = MaterialTheme.typography.bodySmall)
                Slider(
                    value = state.topP.toFloat(),
                    onValueChange = { onTopP(it.toDouble()) },
                    valueRange = 0.5f..1.0f,
                )
                Text("top-k ${state.topK}", style = MaterialTheme.typography.bodySmall)
                Slider(
                    value = state.topK.toFloat(),
                    onValueChange = { onTopK(it.toInt()) },
                    valueRange = 1f..100f,
                    steps = 98,
                )
                Text(
                    "repetition penalty ${fmt(state.repetitionPenalty)}",
                    style = MaterialTheme.typography.bodySmall,
                )
                Slider(
                    value = state.repetitionPenalty.toFloat(),
                    onValueChange = { onRepetitionPenalty(it.toDouble()) },
                    valueRange = 1.0f..1.3f,
                )
            }
        }
    }
}

@Composable
private fun ChatPanel(
    state: AppUiState,
    onSend: (String) -> Unit,
    onCompare: (String) -> Unit,
    modifier: Modifier = Modifier,
) {
    var draft by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    LaunchedEffect(state.messages.size, state.generating) {
        val count = state.messages.size + if (state.generating) 1 else 0
        if (count > 0) listState.animateScrollToItem(count - 1)
    }

    Column(modifier) {
        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f).fillMaxWidth(),
            verticalArrangement = Arrangement.spacedBy(8.dp),
        ) {
            itemsIndexed(state.messages) { _, message ->
                MessageBubble(message)
            }
            if (state.generating) {
                item {
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.Start) {
                        Surface(
                            shape = RoundedCornerShape(16.dp),
                            tonalElevation = 1.dp,
                        ) {
                            Text("Sta generando…", Modifier.padding(12.dp))
                        }
                    }
                }
            }
        }

        state.comparison?.let { comparison ->
            Card(Modifier.fillMaxWidth()) {
                Column(Modifier.padding(10.dp)) {
                    Text("Stesso prompt · Champion vs Latest Research", fontWeight = FontWeight.Bold)
                    Text("Prompt: ${comparison.prompt}", style = MaterialTheme.typography.bodySmall)
                    Spacer(Modifier.height(4.dp))
                    Text("Champion · ${comparison.championMs} ms", fontWeight = FontWeight.SemiBold)
                    Text(if (comparison.champion.text.isEmpty()) "∅" else comparison.champion.text)
                    Text(
                        "${comparison.champion.generatedTokenIds.size} tok · rep ${fmt(comparison.champion.repetitionRate)} · H ${fmt(comparison.champion.meanEntropy)}",
                        style = MaterialTheme.typography.labelSmall,
                    )
                    Spacer(Modifier.height(5.dp))
                    Text("Latest Research · ${comparison.researchMs} ms", fontWeight = FontWeight.SemiBold)
                    Text(if (comparison.research.text.isEmpty()) "∅" else comparison.research.text)
                    Text(
                        "${comparison.research.generatedTokenIds.size} tok · rep ${fmt(comparison.research.repetitionRate)} · H ${fmt(comparison.research.meanEntropy)}",
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
            }
            Spacer(Modifier.height(8.dp))
        }

        Spacer(Modifier.height(8.dp))
        Row(verticalAlignment = Alignment.Bottom) {
            OutlinedTextField(
                value = draft,
                onValueChange = { draft = it },
                modifier = Modifier.weight(1f),
                minLines = 1,
                maxLines = 4,
                placeholder = { Text("Scrivi al modello…") },
                enabled = !state.loadingModel && !state.generating && state.loadedModelId.isNotBlank(),
            )
            Spacer(Modifier.width(8.dp))
            Button(
                onClick = {
                    val text = draft
                    draft = ""
                    onSend(text)
                },
                enabled = draft.isNotBlank() &&
                    !state.loadingModel &&
                    !state.generating &&
                    state.loadedModelId.isNotBlank(),
            ) {
                Text("Invia")
            }
        }
        TextButton(
            onClick = { onCompare(draft) },
            enabled = draft.isNotBlank() &&
                !state.loadingModel &&
                !state.generating &&
                !state.comparing &&
                state.manifest?.slots?.containsKey("research") == true,
        ) {
            Text(if (state.comparing) "Confronto…" else "Confronta Champion / Latest Research")
        }
    }
}

@Composable
private fun MessageBubble(message: ChatLine) {
    val isUser = message.role == "user"
    Row(
        Modifier.fillMaxWidth(),
        horizontalArrangement = if (isUser) Arrangement.End else Arrangement.Start,
    ) {
        Surface(
            shape = RoundedCornerShape(16.dp),
            color = if (isUser) {
                MaterialTheme.colorScheme.primaryContainer
            } else {
                MaterialTheme.colorScheme.surface
            },
            tonalElevation = if (isUser) 0.dp else 1.dp,
            modifier = Modifier.fillMaxWidth(0.88f),
        ) {
            Column(Modifier.padding(12.dp)) {
                Text(
                    if (message.content.isEmpty()) "∅  (output vuoto)" else message.content,
                    style = MaterialTheme.typography.bodyLarge,
                )
                if (!isUser && message.modelId.isNotBlank()) {
                    Spacer(Modifier.height(4.dp))
                    Text(
                        "${message.modelId.take(28)} · ${message.elapsedMs} ms · " +
                            "${message.generatedTokens} tok · rep ${fmt(message.repetitionRate)} · " +
                            "H ${fmt(message.meanEntropy)} · ${message.decodeMode.name}",
                        style = MaterialTheme.typography.labelSmall,
                    )
                }
            }
        }
    }
}

private fun fmt(value: Double): String =
    String.format(Locale.US, "%.4f", value)
