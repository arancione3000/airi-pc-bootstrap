package com.airipc.generalist

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Test

class TokenizerTest {
    @Test
    fun cacheBustingKeepsBranchPathAndAddsVersionKey() {
        val url = cacheBustedUrl(
            "https://raw.githubusercontent.com/a/b/generalist-mobile/manifest.json",
            "cycle-93",
        )
        assertTrue(url.startsWith("https://raw.githubusercontent.com/a/b/generalist-mobile/manifest.json?"))
        assertTrue("airi_v=cycle-93" in url)
    }

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

    @Test
    fun parsesEvolutionNeuralAndAiriPcLabTelemetry() {
        val neural = parseNeuralDiagnostics(
            org.json.JSONObject(
                """{
                  "kind":"aggregated_checkpoint_weights",
                  "note":"real aggregate",
                  "architecture":{
                    "type":"decoder_only_causal_transformer",
                    "vocab_size":384,
                    "context_length":128,
                    "d_model":64,
                    "n_layers":2,
                    "n_heads":4,
                    "head_dim":16,
                    "d_ff":128,
                    "norm_type":"layernorm",
                    "position_encoding":"learned",
                    "ff_variant":"swiglu",
                    "tied_embedding_lm_head":true
                  },
                  "components":[
                    {"id":"embedding","label":"Embedding","parameters":24576,
                     "mean_abs_weight":0.01,"rms_weight":0.02,"max_abs_weight":0.08}
                  ],
                  "heads":[
                    {"layer":0,"head":0,"parameters":3072,
                     "mean_abs_weight":0.01,"rms_weight":0.02,"max_abs_weight":0.07}
                  ],
                  "connections":[]
                }"""
            )
        )
        assertNotNull(neural)
        assertEquals(2, neural!!.architecture.nLayers)
        assertEquals(4, neural.architecture.nHeads)
        assertEquals(1, neural.components.size)

        val evolution = parseEvolutionSummary(
            org.json.JSONObject(
                """{
                  "signals":["language_gap"],
                  "domain_weights":{"language":3.0},
                  "curriculum_memory":{"stored":700,"domains":{"language":98}},
                  "data_growth":{
                    "version":"generalist-data-growth-v3",
                    "queries":["natural language corpus license:mit"],
                    "desired_domains":["language-it"],
                    "added_files":2,
                    "added_bytes":1000,
                    "total_files":1188,
                    "total_bytes":25767831,
                    "domain_files":{"code":501},
                    "domain_bytes":{"code":7210611},
                    "repositories_used":1,
                    "top_repositories":[
                      {"repo":"owner/repo","files":3,"bytes":900,"domains":{"code":3}}
                    ],
                    "rejection_counts":{"persistent_repo_cap":4}
                  },
                  "progressive_scaling":{
                    "current_parameters":115328,
                    "target_parameters":250000,
                    "candidate_generated":true
                  },
                  "progressive_tokenizer":{
                    "current_vocab_size":384,
                    "target_vocab_size":512,
                    "maximum_vocab_size":1024,
                    "probe_generated":true
                  },
                  "latest_research":{"candidate_id":"research-1"}
                }"""
            )
        )
        assertEquals(1188, evolution.dataGrowth.totalFiles)
        assertEquals(512, evolution.tokenizerTargetVocab)
        assertEquals("research-1", evolution.latestResearchId)

        val lab = parseAiriPcLabReport(
            org.json.JSONObject(
                """{
                  "version":"airi-pc-lab-v1",
                  "mode":"read_only_sandbox",
                  "learning":{"training_rows":8,"domains":{"tools":2,"coding":6}},
                  "snapshot":{
                    "capabilities":["list_capabilities","inspect_module"],
                    "denied_capabilities":["shell","phone_control"],
                    "modules":[
                      {"name":"task_engine","path":"computer/control_plane/task_engine.py",
                       "symbols":[{"name":"TaskEngine","kind":"class","doc":""}]}
                    ]
                  },
                  "champion":{"ok":false,"tool_call_valid":false,"error":"invalid"}
                }"""
            )
        )
        assertEquals("read_only_sandbox", lab.mode)
        assertTrue("shell" in lab.deniedCapabilities)
        assertEquals("TaskEngine", lab.modules.first().symbols.first())
    }
}
