package com.airipc.generalist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class SyncStateTest {
    @Test
    fun modelIsCurrentWhenStateShaMatchesLiveRevision() {
        assertFalse(
            isModelBehindState(
                "abcdef123456",
                "abcdef123456",
            )
        )
    }

    @Test
    fun modelIsBehindWhenLiveStateHasAdvanced() {
        assertTrue(
            isModelBehindState(
                "abcdef123456",
                "fedcba654321",
            )
        )
    }

    @Test
    fun missingRevisionDoesNotPretendThereIsKnownLag() {
        assertFalse(isModelBehindState("", "fedcba654321"))
        assertFalse(isModelBehindState("abcdef123456", null))
        assertFalse(isModelBehindState("abcdef123456", ""))
    }


    private fun slot(name: String, modelSha: String) = ModelSlot(
        name = name,
        id = name,
        path = name,
        cycle = 1,
        parameters = 7_000_000,
        score = 0.0,
        nllPerByte = 0.0,
        generationSimilarity = 0.0,
        generationExactAccuracy = 0.0,
        generationNonemptyRate = 1.0,
        tokenizerVersion = "bpe-v1",
        tokenizerVocabSize = 384,
        contextLength = 128,
        researchOnly = name != "champion",
        allSeedEligible = name == "champion",
        neural = null,
        files = mapOf(
            "model.onnx" to BundleFileInfo(modelSha, 123L),
        ),
    )

    @Test
    fun duplicateResearchSlotIsRecognizedAsSameLiveModel() {
        assertTrue(
            sameModelArtifact(
                slot("champion", "same-model"),
                slot("research", "same-model"),
            )
        )
    }

    @Test
    fun genuinelyDifferentResearchSlotRemainsDistinct() {
        assertFalse(
            sameModelArtifact(
                slot("champion", "live-model"),
                slot("research", "different-model"),
            )
        )
    }


    @Test
    fun pinnedMobileRevisionDoesNotNeedFallback() {
        var fallbackCalls = 0
        val resolved = resolveDownloadRevision("abc123") {
            fallbackCalls += 1
            "fallback"
        }
        assertEquals("abc123", resolved)
        assertEquals(0, fallbackCalls)
    }

    @Test
    fun missingMobileRevisionIsRecoveredFromBranchResolver() {
        val resolved = resolveDownloadRevision("") { "fresh-branch-sha" }
        assertEquals("fresh-branch-sha", resolved)
    }
}
