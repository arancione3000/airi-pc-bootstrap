package com.airipc.generalist

import org.junit.Assert.assertEquals
import org.junit.Test

class TokenizerTest {
    @Test
    fun byteTokenizerRoundTripsUtf8() {
        val tokenizer = ByteTokenizer()
        val text = "ciao AIRI 🌸"
        assertEquals(text, tokenizer.decode(tokenizer.encode(text)))
    }

    @Test
    fun bpeTokenizerPreservesPythonMergeSemantics() {
        val a = BYTE_OFFSET + 'a'.code
        val b = BYTE_OFFSET + 'b'.code
        val tokenizer = BpeTokenizer(listOf(a to b))
        val encoded = tokenizer.encode("ab ab")
        assertEquals(BYTE_VOCAB_SIZE, encoded.first())
        assertEquals("ab ab", tokenizer.decode(encoded))
    }

    @Test
    fun chatSerializationUsesRoleTokens() {
        val tokenizer = ByteTokenizer()
        val ids = tokenizer.serializeMessages(
            listOf(ChatMessage("user", "hi"))
        )
        assertEquals(BOS, ids.first())
        assertEquals(USER, ids[1])
        assertEquals(ASSISTANT, ids.last())
    }
}
