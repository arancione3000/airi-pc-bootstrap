package com.airipc.generalist

import org.json.JSONObject
import org.junit.Assert.*
import org.junit.Test

class ObservabilityModelsTest {
    @Test
    fun parsesNeuralDiagnostics() {
        val raw = JSONObject(
            """
            {
              "kind":"aggregated_checkpoint_weights",
              "note":"real checkpoint aggregate",
              "architecture":{
                "type":"decoder_only_causal_transformer",
                "vocab_size":384,
                "context_length":128,
                "d_model":48,
                "n_layers":2,
                "n_heads":4,
                "head_dim":12,
                "d_ff":128,
                "norm_type":"rmsnorm",
                "position_encoding":"rope",
                "ff_variant":"swiglu",
                "tied_embedding_lm_head":true
              },
              "components":[
                {"id":"embedding","label":"Embedding","parameters":100,"mean_abs_weight":0.1,"rms_weight":0.2,"max_abs_weight":0.7}
              ],
              "heads":[
                {"layer":0,"head":0,"parameters":64,"mean_abs_weight":0.02,"rms_weight":0.03,"max_abs_weight":0.2}
              ],
              "connections":[
                {"from":"embedding","to":"layer_0_attention","kind":"residual_flow"}
              ]
            }
            """.trimIndent()
        )
        val parsed = parseNeuralDiagnostics(raw)
        assertNotNull(parsed)
        assertEquals(2, parsed!!.architecture.nLayers)
        assertEquals(4, parsed.architecture.nHeads)
        assertEquals(1, parsed.components.size)
        assertEquals(1, parsed.heads.size)
        assertEquals("embedding", parsed.connections.single().from)
    }

    @Test
    fun parsesEvolutionAndLabTelemetry() {
        val evolution = parseEvolutionSummary(
            JSONObject(
                """
                {
                  "signals":["language_gap","coding_gap"],
                  "domain_weights":{"language":2.0},
                  "curriculum_memory":{"stored":42,"domains":{"language":12,"tools":8}},
                  "data_growth":{
                    "version":"generalist-data-growth-v3",
                    "queries":["natural language corpus license:mit"],
                    "desired_domains":["language-it","general"],
                    "added_files":3,
                    "added_bytes":4096,
                    "total_files":99,
                    "total_bytes":100000,
                    "domain_files":{"general":50},
                    "domain_bytes":{"general":50000},
                    "repositories_used":2,
                    "top_repositories":[{"repo":"owner/repo","files":4,"bytes":1000,"domains":{"general":4}}],
                    "rejection_counts":{"license:unknown":2}
                  },
                  "progressive_scaling":{"current_parameters":115328,"target_parameters":250000,"candidate_generated":true},
                  "progressive_tokenizer":{"current_vocab_size":384,"target_vocab_size":512,"maximum_vocab_size":1024,"probe_generated":true},
                  "latest_research":{"candidate_id":"candidate-x"}
                }
                """.trimIndent()
            )
        )
        assertEquals(2, evolution.signals.size)
        assertEquals(99, evolution.dataGrowth.totalFiles)
        assertEquals(512, evolution.tokenizerTargetVocab)
        assertEquals("candidate-x", evolution.latestResearchId)

        val lab = parseAiriPcLabReport(
            JSONObject(
                """
                {
                  "version":"airi-pc-lab-v1",
                  "mode":"read_only_sandbox",
                  "learning":{"training_rows":10,"domains":{"tools":4,"coding":6}},
                  "snapshot":{
                    "capabilities":["list_capabilities"],
                    "denied_capabilities":["shell","phone_control"],
                    "modules":[
                      {"name":"task_engine","path":"computer/control_plane/task_engine.py","symbols":[{"name":"TaskEngine"}]}
                    ]
                  },
                  "champion":{"ok":false,"tool_call_valid":false,"error":"not ready"}
                }
                """.trimIndent()
            )
        )
        assertEquals("read_only_sandbox", lab.mode)
        assertTrue(lab.deniedCapabilities.contains("shell"))
        assertEquals("task_engine", lab.modules.single().name)
        assertFalse(lab.champion!!.ok)
    }
}
