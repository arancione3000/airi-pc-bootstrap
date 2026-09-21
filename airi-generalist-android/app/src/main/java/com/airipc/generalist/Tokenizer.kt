package com.airipc.generalist

import org.json.JSONObject
import java.io.ByteArrayOutputStream

const val PAD = 0
const val BOS = 1
const val EOS = 2
const val SYSTEM = 3
const val USER = 4
const val ASSISTANT = 5
const val TOOL = 6
const val SEP = 7
const val BYTE_OFFSET = 8
const val BYTE_VOCAB_SIZE = BYTE_OFFSET + 256

data class ChatMessage(val role: String, val content: String)

interface GeneralistTokenizer {
    val version: String
    val vocabSize: Int
    fun encode(text: String): List<Int>
    fun decode(ids: Iterable<Int>): String
    fun serializeMessages(messages: List<ChatMessage>): List<Int>
}

private val ROLE_TOKENS = mapOf(
    "system" to SYSTEM,
    "user" to USER,
    "assistant" to ASSISTANT,
    "tool" to TOOL,
)

class ByteTokenizer : GeneralistTokenizer {
    override val version: String = "byte-v1"
    override val vocabSize: Int = BYTE_VOCAB_SIZE

    override fun encode(text: String): List<Int> =
        text.toByteArray(Charsets.UTF_8).map { BYTE_OFFSET + (it.toInt() and 0xff) }

    override fun decode(ids: Iterable<Int>): String {
        val bytes = ByteArrayOutputStream()
        for (token in ids) {
            if (token in BYTE_OFFSET until BYTE_VOCAB_SIZE) {
                bytes.write(token - BYTE_OFFSET)
            }
        }
        return bytes.toByteArray().toString(Charsets.UTF_8)
    }

    override fun serializeMessages(messages: List<ChatMessage>): List<Int> {
        val out = mutableListOf(BOS)
        for (message in messages) {
            val role = ROLE_TOKENS[message.role.lowercase()]
                ?: error("Ruolo chat non supportato: ${message.role}")
            out += role
            out += encode(message.content)
            out += SEP
        }
        out += ASSISTANT
        return out
    }
}

class BpeTokenizer(val merges: List<Pair<Int, Int>>) : GeneralistTokenizer {
    override val version: String = "bpe-v1"
    override val vocabSize: Int = BYTE_VOCAB_SIZE + merges.size

    private val tokenBytes: Map<Int, ByteArray>

    init {
        val bytes = mutableMapOf<Int, ByteArray>()
        for (value in 0..255) {
            bytes[BYTE_OFFSET + value] = byteArrayOf(value.toByte())
        }
        var nextId = BYTE_VOCAB_SIZE
        val seen = mutableSetOf<Pair<Int, Int>>()
        for (pair in merges) {
            require(pair.first >= BYTE_OFFSET && pair.second >= BYTE_OFFSET)
            require(pair.first < nextId && pair.second < nextId)
            require(seen.add(pair))
            val left = bytes[pair.first] ?: error("Merge BPE sinistro sconosciuto")
            val right = bytes[pair.second] ?: error("Merge BPE destro sconosciuto")
            bytes[nextId] = left + right
            nextId += 1
        }
        tokenBytes = bytes
    }

    private fun replacePair(tokens: List<Int>, pair: Pair<Int, Int>, newId: Int): List<Int> {
        if (tokens.size < 2) return tokens
        val out = ArrayList<Int>(tokens.size)
        var i = 0
        while (i < tokens.size) {
            if (
                i + 1 < tokens.size &&
                tokens[i] == pair.first &&
                tokens[i + 1] == pair.second
            ) {
                out += newId
                i += 2
            } else {
                out += tokens[i]
                i += 1
            }
        }
        return out
    }

    override fun encode(text: String): List<Int> {
        var tokens: List<Int> = text.toByteArray(Charsets.UTF_8)
            .map { BYTE_OFFSET + (it.toInt() and 0xff) }
        var nextId = BYTE_VOCAB_SIZE
        for (pair in merges) {
            tokens = replacePair(tokens, pair, nextId)
            nextId += 1
        }
        return tokens
    }

    override fun decode(ids: Iterable<Int>): String {
        val bytes = ByteArrayOutputStream()
        for (token in ids) {
            val raw = tokenBytes[token]
            if (raw != null) bytes.write(raw)
        }
        return bytes.toByteArray().toString(Charsets.UTF_8)
    }

    override fun serializeMessages(messages: List<ChatMessage>): List<Int> {
        val out = mutableListOf(BOS)
        for (message in messages) {
            val role = ROLE_TOKENS[message.role.lowercase()]
                ?: error("Ruolo chat non supportato: ${message.role}")
            out += role
            out += encode(message.content)
            out += SEP
        }
        out += ASSISTANT
        return out
    }

    companion object {
        fun fromJson(json: String): BpeTokenizer {
            val root = JSONObject(json)
            require(root.getString("version") == "bpe-v1")
            val raw = root.getJSONArray("merges")
            val merges = buildList {
                for (i in 0 until raw.length()) {
                    val pair = raw.getJSONArray(i)
                    add(pair.getInt(0) to pair.getInt(1))
                }
            }
            val tokenizer = BpeTokenizer(merges)
            val declared = root.optInt("vocab_size", tokenizer.vocabSize)
            require(declared == tokenizer.vocabSize)
            return tokenizer
        }
    }
}

fun tokenizerFromJson(json: String): GeneralistTokenizer {
    val root = JSONObject(json)
    return when (root.getString("version")) {
        "byte-v1" -> ByteTokenizer()
        "bpe-v1" -> BpeTokenizer.fromJson(json)
        else -> error("Tokenizer non supportato: ${root.optString("version")}")
    }
}
