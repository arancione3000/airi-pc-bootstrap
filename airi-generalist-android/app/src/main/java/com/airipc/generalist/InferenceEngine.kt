package com.airipc.generalist

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.Closeable
import java.nio.LongBuffer
import kotlin.math.exp
import kotlin.math.ln
import kotlin.random.Random

enum class DecodeMode {
    GREEDY,
    SAMPLING,
}

data class DecodeSettings(
    val mode: DecodeMode = DecodeMode.GREEDY,
    val temperature: Double = 0.8,
    val topP: Double = 0.9,
    val topK: Int = 40,
    val repetitionPenalty: Double = 1.08,
)

data class TokenPrediction(
    val tokenId: Int,
    val text: String,
    val logit: Double,
)

data class InferenceTrace(
    val text: String,
    val promptTokens: Int,
    val generatedTokenIds: List<Int>,
    val lastTopPredictions: List<TokenPrediction>,
    val repetitionRate: Double,
    val meanEntropy: Double,
)

class AiriOnnxEngine(bundle: InstalledBundle) : Closeable {
    private val environment = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    private val tokenizer: GeneralistTokenizer
    private val contextLength: Int

    init {
        val config = JSONObject(bundle.config.readText())
        contextLength = config.getInt("context_length")
        tokenizer = tokenizerFromJson(bundle.tokenizer.readText())
        check(tokenizer.vocabSize == config.getInt("vocab_size")) {
            "Tokenizer e modello hanno vocab diversi"
        }
        val options = OrtSession.SessionOptions()
        try {
            session = environment.createSession(bundle.model.absolutePath, options)
        } finally {
            options.close()
        }
    }

    fun chat(
        messages: List<ChatMessage>,
        maxNewTokens: Int = 64,
        settings: DecodeSettings = DecodeSettings(),
    ): InferenceTrace {
        val prompt = tokenizer.serializeMessages(messages)
        val allIds = prompt.toMutableList()
        val generated = mutableListOf<Int>()
        var lastTopPredictions: List<TokenPrediction> = emptyList()
        val entropies = mutableListOf<Double>()

        for (step in 0 until maxNewTokens.coerceIn(1, 192)) {
            val context = allIds.takeLast(contextLength)
            val raw = LongArray(context.size) { context[it].toLong() }
            val shape = longArrayOf(1L, context.size.toLong())
            val logits = OnnxTensor.createTensor(
                environment,
                LongBuffer.wrap(raw),
                shape,
            ).use { input ->
                session.run(mapOf("input_ids" to input)).use { result ->
                    extractLogits(result[0].value)
                }
            }

            lastTopPredictions = topPredictions(logits, 5)
            val decision = decode(logits, generated, settings)
            entropies += decision.entropy
            val next = decision.tokenId
            if (next == EOS) break
            allIds += next
            generated += next
        }

        val uniqueRatio = if (generated.isEmpty()) {
            0.0
        } else {
            generated.toSet().size.toDouble() / generated.size.toDouble()
        }
        return InferenceTrace(
            text = tokenizer.decode(generated),
            promptTokens = prompt.size,
            generatedTokenIds = generated.toList(),
            lastTopPredictions = lastTopPredictions,
            repetitionRate = if (generated.isEmpty()) 0.0 else 1.0 - uniqueRatio,
            meanEntropy = if (entropies.isEmpty()) 0.0 else entropies.average(),
        )
    }

    private data class DecodeDecision(
        val tokenId: Int,
        val entropy: Double,
    )

    private fun decode(
        rawLogits: FloatArray,
        generated: List<Int>,
        settings: DecodeSettings,
    ): DecodeDecision {
        val logits = DoubleArray(rawLogits.size) { rawLogits[it].toDouble() }

        if (settings.mode == DecodeMode.SAMPLING) {
            val penalty = settings.repetitionPenalty.coerceIn(1.0, 2.0)
            if (penalty > 1.0) {
                generated.toSet().forEach { token ->
                    if (token in logits.indices) {
                        logits[token] = if (logits[token] < 0.0) {
                            logits[token] * penalty
                        } else {
                            logits[token] / penalty
                        }
                    }
                }
            }
            val temperature = settings.temperature.coerceIn(0.05, 2.0)
            for (i in logits.indices) logits[i] /= temperature

            val keepK = settings.topK.coerceIn(1, logits.size)
            val topIndices = logits.indices
                .sortedByDescending { logits[it] }
                .take(keepK)
            val keep = BooleanArray(logits.size)
            topIndices.forEach { keep[it] = true }
            for (i in logits.indices) {
                if (!keep[i]) logits[i] = Double.NEGATIVE_INFINITY
            }

            if (settings.topP in 0.0..0.999999) {
                val current = probabilities(logits)
                var cumulative = 0.0
                val ordered = logits.indices.sortedByDescending { logits[it] }
                ordered.forEachIndexed { index, token ->
                    cumulative += current[token]
                    if (index > 0 && cumulative - current[token] >= settings.topP) {
                        logits[token] = Double.NEGATIVE_INFINITY
                    }
                }
            }
        }

        val probs = probabilities(logits)
        val entropy = probs
            .filter { it > 0.0 }
            .sumOf { -it * ln(it) }

        if (settings.mode == DecodeMode.GREEDY) {
            var best = 0
            for (i in 1 until logits.size) {
                if (logits[i] > logits[best]) best = i
            }
            return DecodeDecision(best, entropy)
        }

        val draw = Random.nextDouble()
        var cumulative = 0.0
        for (i in probs.indices) {
            cumulative += probs[i]
            if (draw <= cumulative) return DecodeDecision(i, entropy)
        }
        return DecodeDecision(probs.indices.maxByOrNull { probs[it] } ?: 0, entropy)
    }

    private fun probabilities(logits: DoubleArray): DoubleArray {
        val finite = logits.filter { it.isFinite() }
        if (finite.isEmpty()) {
            val out = DoubleArray(logits.size)
            out[0] = 1.0
            return out
        }
        val max = finite.max()
        val weights = DoubleArray(logits.size) { index ->
            if (logits[index].isFinite()) exp(logits[index] - max) else 0.0
        }
        val total = weights.sum().coerceAtLeast(1e-300)
        return DoubleArray(weights.size) { weights[it] / total }
    }

    private fun topPredictions(values: FloatArray, count: Int): List<TokenPrediction> =
        values.indices
            .sortedByDescending { values[it] }
            .take(count.coerceAtLeast(1))
            .map { tokenId ->
                TokenPrediction(
                    tokenId = tokenId,
                    text = tokenizer.decode(listOf(tokenId)),
                    logit = values[tokenId].toDouble(),
                )
            }

    private fun extractLogits(value: Any): FloatArray {
        if (value is Array<*>) {
            val first = value.firstOrNull()
            if (first is FloatArray) return first
            if (first is Array<*>) {
                val last = first.lastOrNull()
                if (last is FloatArray) return last
            }
        }
        error("Output ONNX inatteso: ${value::class.java.name}")
    }

    override fun close() {
        session.close()
    }
}
