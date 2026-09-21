package com.airipc.generalist

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
}
