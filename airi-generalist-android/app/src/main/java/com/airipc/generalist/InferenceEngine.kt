package com.airipc.generalist

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.io.Closeable
import java.io.File
import java.nio.LongBuffer

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

    fun chat(messages: List<ChatMessage>, maxNewTokens: Int = 64): String {
        val prompt = tokenizer.serializeMessages(messages)
        val allIds = prompt.toMutableList()
        val generated = mutableListOf<Int>()

        repeat(maxNewTokens.coerceIn(1, 192)) {
            val context = allIds.takeLast(contextLength)
            val raw = LongArray(context.size) { context[it].toLong() }
            val shape = longArrayOf(1L, context.size.toLong())
            val next = OnnxTensor.createTensor(
                environment,
                LongBuffer.wrap(raw),
                shape,
            ).use { input ->
                session.run(mapOf("input_ids" to input)).use { result ->
                    argmax(extractLogits(result[0].value))
                }
            }
            if (next == EOS) return@repeat
            allIds += next
            generated += next
            if (next == EOS) return@repeat
        }
        return tokenizer.decode(generated)
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

    private fun argmax(values: FloatArray): Int {
        check(values.isNotEmpty())
        var bestIndex = 0
        var bestValue = values[0]
        for (i in 1 until values.size) {
            if (values[i] > bestValue) {
                bestValue = values[i]
                bestIndex = i
            }
        }
        return bestIndex
    }

    override fun close() {
        session.close()
    }
}
