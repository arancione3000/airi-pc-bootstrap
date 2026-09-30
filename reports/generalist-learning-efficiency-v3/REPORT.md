# AIRI generalist learning efficiency v3 — audit e risultato negativo

**Obiettivo non raggiunto. Nessuna nuova policy è approvata per il training live.** Sono stati aggiunti strumenti offline e prove riproducibili, non una presunta ottimizzazione validata. Nei 14 trial al learning rate corretto nessun aggiornamento supera i gate. Non esiste prova di 3x, né di 8k/16k stabili. Questo branch deve restare in draft.

## 1. Baseline_start dinamica

La fotografia autorevole è `baseline_start.json`: include timestamp, genome completo, tokenizer, digest verificati, contatori, loss, salute, optimizer, recovery, replay e target dinamici. Codice iniziale `532b2b81d50e288ff8a759efe8722e95acaad8dc`; stato live iniziale `83d735f392c91163cade8746cccbbcc574d52724`, aggiornato 2026-09-30T20:25:59Z. Le misure sono della copia fissata all'inizio, non dell'HEAD live alla consegna.

| Metrica | Baseline |
|---|---:|
| Lineage | airi-5d3d25177d2e83f7 |
| Parametri | 50.041.536 |
| Causal tokens persistiti | 33.157.229 |
| Valid tokens | 33.162.883 |
| Target | 100.000.000 |
| Attempted/h, tempo training | 952.499,62 |
| Accepted/h, tempo training | 92.838,76 |
| Target minimo, baseline × 3 | 278.516,28/h |
| Target forte, baseline × 5 | 464.193,81/h |
| Tentativi / accettati / rollback | 466 / 75 / 391 |
| Acceptance / rollback | 16,09% / 83,91% |
| NLL / validation loss | 2,023482 / 2,086585 |
| Repetition / durable repetition | 0,236158 / 0,157721 |
| Hard boundary / headroom | 0,237721 / 0,001563 |
| Multiword / unique / patologica | 1 / 0,763842 / false |
| Segmento effettivo / richiesto | 4.064 / 1.000.000 |
| LR scale / LR effettivo | 1/128 / 4,5587e-7 |
| Replay dichiarato / peso / KL | 0,80 / 4 / 0,5 |
| Ultimo replay effettivo | 44 / 4.064 new, 1,071% |

Il contatore accepted non certifica da solo apprendimento semanticamente utile: 15 degli ultimi 32 accepted hanno language-quality delta negativo. Non rinominiamo questi token come una misura validata di utilità. Il report live di throughput esclude acquisizione, avvio, scheduling e push. La finestra cronologica conservata di 64 record implica circa 5.361 accepted causal token/h sul tempo calendario (popolazione diversa dal totale); non è un confronto diretto con il throughput di training.

## 2. Forensic audit e root cause

`forensic_rollbacks.json` conserva righe, ragioni, misure disponibili e aggregati. Tutti gli ultimi 32 rollback conservati superano il limite durable di repetition; tre hanno anche regressione locale. Taglie: 28 × 4.064, due × 32.512, una × 8.128 e una × 503.936. LR scale sempre 1/128. La finestra cronologica di 64 tentativi ha 23 accepted e rollback 64,06%; non rappresenta tutti i 391 rollback.

La perdita dominante è quindi il rejection, non il conteggio di step. La repetition attuale ha solo 0,00156 di margine: può passare da un lato all'altro del confine con piccoli cambiamenti autoregressivi. Le correlazioni causali tra LR, optimizer e probabilità di rifiuto NON sono identificabili dalla storia conservata: mancano stati e gradienti per tentativo, e quasi tutti i segmenti recenti sono uguali. I campi mancanti rimangono null; non sono ricostruiti da impressioni.

### Replay: loss fraction non equivale a token fraction

Il replay 0,80 corrisponde a un rapporto di loss normalizzate `Lc + 4 Lr`; non riserva l'80% dei token. Il campionamento fisso di pochissime risposte, ulteriormente ristretto agli esempi elementari in recovery, spiega il rapporto token circa 1%. Peso maggiore non produce copertura, diversità o riduzione della varianza equivalenti a un batch più ampio. Duplicare la stessa risposta lascia invariata una media normalizzata; moltiplicare la loss cambia invece il gradiente.

`objective-gradients.json`: su un microbatch deterministico di 32 label causali e due replay, norma CE causale 2,22136, replay CE grezza 18,98414 e pesata 75,93655: rapporto 34,18. Coseno tra direzioni 0,01354. Anti-repetition causale pesata 0,00370, circa 0,166% della CE. La KL contro il medesimo checkpoint ha gradiente zero al primo step. Queste sono misure locali, non stime della media live. Il replay è povero di copertura ma non necessariamente debole come gradiente; aumentarlo alla cieca non risolve il problema.

### Architettura e migrazione

Genome corrente: width 96, 12 layer, quattro head MHA, FFN SwiGLU 14.336, context 128, BPE-v1 vocab384. FFN 49.545.216 parametri (99,008%); attention/norm 447.168; embedding/pos 49.152. Geometria estrema confermata, inefficienza causale ancora non dimostrata.

Quattro geometrie circa 50M sono state valutate tramite la funzione di trasferimento già esistente: width256/layer16/FF3712, width384/layer12/FF3072, width512/layer12/FF2048 MHA, width512/layer12/FF2304 GQA. Nessuna preserva la funzione: soltanto due tensori trasferiti, quasi tutti i pesi appresi non compatibili, NLL circa 6, repetition patologica, gate respinto. `architecture-geometry.json` contiene tutti i valori. È una prova di insufficienza della migrazione disponibile, NON un confronto addestrato che dimostri geometrie più larghe inferiori. Nessuna migration/promotion eseguita.

## 3. Dataset

`dataset_audit.json` analizza il replay persistito: 22.851 documenti, 977.148 token, duplicati esatti/normalizzati/template numerici zero; near-duplicate stimati 0,1007% mediante char-5 Jaccard ≥ 0,9 e candidati MinHash deterministici limitati (recall non misurata). 173 documenti a diversità bassa, 73 molto brevi. Quantili lunghezza in token: 2, 19, 26, 36, 118, 1.194. English circa 60,2%, Italian circa 39,8%; language 777.639 token, dialogue 199.509. Il manifesto originale indica 32.001.668 token selezionati e un mix più ampio.

La copia persistita non contiene il corpus completo né la validation originale. Non si estrapolano duplicazione e diversità a tutto il corpus; non viene dichiarata una validation loss post-trial. Non è ancora dimostrato che duplicazione del corpus sia la causa primaria. Nessuna modifica al sampling live e nessuna contaminazione holdout.

## 4. Esperimenti e risultati falliti

Ogni trial ricarica lo stesso checkpoint, stesso seed 5.000.000 e slice causale identica per varianti con stesso curriculum. La slice diagnostica deriva da replay già addestrato: serve a studiare stabilità, non a dimostrare nuovo apprendimento persistente sul corpus fresco. Il harness può ricevere JSONL di documenti training verificati; non acquisisce teacher/weights esterni.

**Errore sperimentale dichiarato:** i primi 16 trial usavano base LR 1e-4 ricavato dal genome, anziché 3e-4 configurato dal workflow. Sono archiviati in `exploratory-low-lr.jsonl.gz` con marcatura esplicita di esclusione dalla prova di successo; alcune prime misure avevano concorrenza locale e non sono timing affidabili. Varianti esplorate: baseline, replay elementare32/broad32/proporzionale, normalizzazione token, bilanciamento gradienti, KL0/5 e durable, anti1, sampling repetition, cosine, resume momentum, warmup, clipping, curriculum generale. Non si usano questi risultati per confronti di throughput.

I 14 trial successivi usano il LR del workflow, 4 thread e nessun altro benchmark concorrente: replay proporzionale, bilanciamento esatto, optimizer SGD a due scale, prefix acceptance e durable KL500 su causal, contro baseline8k/16k. Tutti respinti. Repetition prima 0,236158 e NLL prima 2,023482 per tutti. Tabella calcolata dal journal completo `controlled-trials.jsonl.gz`:

| Trial | Token causali | Attempted/h | Accepted-equivalent/h | Replay effettivo | Repetition dopo | NLL dopo | Multiword | Unique | Patologica | Wall s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| baseline | 8128 | 735880 | 0 | 0.87% | 0.260385 | 2.024444 | 1.000 | 0.739615 | False | 63.3 |
| proportional25 | 8128 | 510109 | 0 | 25.57% | 0.249537 | 2.016287 | 1.000 | 0.750463 | False | 69.2 |
| balanced_exact | 8128 | 393564 | 0 | 45.98% | 0.248714 | 2.017541 | 1.000 | 0.751286 | False | 85.9 |
| balanced_elementary | 8128 | 760214 | 0 | 0.87% | 0.259562 | 2.024174 | 1.000 | 0.740438 | False | 50.2 |
| sgd_direction | 8128 | 766559 | 0 | 0.87% | 0.256768 | 2.024066 | 1.000 | 0.743232 | False | 50.1 |
| sgd_direction_strong | 8128 | 796901 | 0 | 0.87% | 0.251948 | 2.026199 | 1.000 | 0.748052 | False | 48.5 |
| sgd_broad | 8128 | 563937 | 0 | 25.57% | 0.352953 | 1.983436 | 0.929 | 0.647047 | False | 62.7 |
| prefix_acceptance | 8128 | 536889 | 0 | 0.87% | 0.260385 | 2.024444 | 1.000 | 0.739615 | False | 66.5 |
| baseline | 16256 | 875240 | 0 | 1.19% | 0.258938 | 2.026759 | 1.000 | 0.741062 | False | 78.8 |
| proportional25 | 16256 | 622417 | 0 | 25.45% | 0.303234 | 2.001138 | 1.000 | 0.696766 | False | 105.8 |
| durable_kl_strong | 8128 | 375726 | 0 | 0.87% | 0.256768 | 2.023366 | 1.000 | 0.743232 | False | 101.8 |
| durable_kl_broad | 8128 | 299399 | 0 | 25.57% | 0.248607 | 2.023172 | 1.000 | 0.751393 | False | 111.3 |
| durable_kl_strong | 16256 | 419640 | 0 | 1.19% | 0.274550 | 2.023261 | 1.000 | 0.725450 | False | 152.1 |
| durable_kl_broad | 16256 | 343161 | 0 | 25.45% | 0.264281 | 2.022472 | 1.000 | 0.735719 | False | 183.8 |

Acceptance 0%, rollback 100%, checkpoint persist overhead 0s per ciascuno perché respinto, live tokens persistiti zero. Rapporti/norme per step, metriche quality, conversational trace, LR/schedule, optimizer, hash, timing training/eval/instrumentation/persistence sono nel journal. Il tempo wall include diagnostica; attempted/h usa il tempo training+evaluation dichiarato e non va confuso con il calendario live.

La loss migliore non salva SGD broad: NLL diminuisce circa 0,040 ma repetition raggiunge 0,353. Anche replay reale circa 25%, gradient balancing esatto, KL durable forte e checkpoint interni non hanno prodotto 8k/16k accettati in questo seed. Non è una finestra statistica rappresentativa: un tentativo per variante/taglia non dimostra la superiorità o inferiorità generale. Nessun 32k è dichiarato stabile e non sono stati consumati ulteriori trial più grandi dopo il fallimento8k/16k.

## 5. Soluzione scelta e file modificati

Si consegna una infrastruttura diagnostica offline; nessuna modifica di prodotto al training è giustificata dai risultati.

- `computer/generalist_lm/learning_efficiency_audit.py`: baseline con verifica digest, forensic audit e corpus snapshot audit, distinguendo dati disponibili/mancanti.
- `computer/generalist_lm/learning_efficiency_benchmark.py`: ablations isolate, pressione gradienti, token replay reali, prefix restore, verifier invariato, checkpoint candidate isolato; journal fsync+replace e lock, resume solo con input/config compatibili, verifica fonte immutata.
- `tests/test_generalist_learning_efficiency.py`: 19 regression test per accounting, sampling, equivalenza dei pesi, KL, holdout exclusion, sorgente immutabile, rejection/acceptance offline, resume/interruzione e soglia durable invariata.
- `.ai/PROJECT_MEMORY.md`: risultato negativo e limiti sperimentali.
- `reports/generalist-learning-efficiency-v3/*`: evidenza, questo report e risultati compressi.

Il codice live di recovery/growth/shrink, optimizer/persistence e i test esistenti restano invariati; non si afferma di aver implementato una nuova health policy. Nessuna modifica a main, workflow, stato live, anchor, context, tokenizer, gate, probe o definizione dei token accepted. Nessun external pretrained LLM/teacher. La copia benchmark non è un lineage live parallelo e non viene promossa.

## 6. Verifiche

`PYTHONPATH=computer python -m pytest -q tests/test_generalist_learning_efficiency.py tests/test_generalist_phase5.py tests/test_generalist_lm.py tests/test_generalist_architecture.py tests/test_generalist_single_lineage.py`: **227 passed, 1 skipped**, 37,09s. Skip backend opzionale transformers/tokenizers non installato. Un primo run falliva per z3 non installato; installata la dipendenza già richiesta da mathesis, poi suite verde. Phase5 incluso (89 test). Eseguiti py_compile, Black e controllo whitespace del diff staged. Nessuna prova unit può sostituire i risultati empirici negativi.

## 7. Prima/dopo e rischi rimasti

| Criterio | Live baseline | Candidate offline al LR corretto |
|---|---|---|
| Accepted/h | 92.838,76 training-only | 0 accepted-equivalent; nessun live trial |
| Rollback | 83,91% cumulativo | 100% dei 14 trial, finestra piccola |
| Repetition/headroom | 0,236158 / +0,001563 | tutti oltre confine; headroom negativo |
| Largest stable segment | 4.064 recente; un32.512 accepted non prova stabilità | nessun8k/16k stabile dimostrato |
| Validation loss | 2,086585 snapshot | non disponibile senza corpus holdout |
| Conversational relevance | debole, trace snapshot | trace registrate; nessun miglioramento validato |

Rischi: proxy replay diverso dai dati freschi; un solo seed controllato; probes discreti sensibili a piccoli shift; assenza di storia completa/gradient telemetry live; geometry migration non conservativa; NLL può migliorare mentre generazione peggiora. Il target di headroom anchor+0,04 non è raggiunto. La causa ultima della dinamica autoregressiva non è ancora isolata. Rimangono da fare ablations corrette multi-seed sui dati originali, validation corpus reale e una successione accepted nello stesso candidate checkpoint prima di parlare di stabilità; non basta ripartire da baseline indipendente.

## 8. Riproduzione

Ambiente utilizzato: Python runtime della sessione, torch2.14.1+cpu, pytest9, Black26, z3-solver, quattro torch thread (runner scratch circa8CPU/8GiB). Nessun claim di identico hardware live.

Preparare una copia esportata dello stato pinned `83d735f392c91163cade8746cccbbcc574d52724` in `$STATE_COPY`, directory output distinta in `$RESULTS_DIR`. Non usare il checkout live come output.

```bash
PYTHONPATH=computer python -m generalist_lm.learning_efficiency_audit --state-dir "$STATE_COPY" --state-sha 83d735f392c91163cade8746cccbbcc574d52724 --code-sha 532b2b81d50e288ff8a759efe8722e95acaad8dc --output-dir "$RESULTS_DIR/audit"
PYTHONPATH=computer python -m generalist_lm.learning_efficiency_benchmark --state-dir "$STATE_COPY" --output "$RESULTS_DIR/paired.jsonl" --segments 8128,16256 --seeds 5000000 --threads 4 --trials baseline,proportional25,balanced_exact,balanced_elementary,sgd_direction,sgd_direction_strong,sgd_broad,prefix_acceptance --measure-gradients
PYTHONPATH=computer python -m generalist_lm.learning_efficiency_benchmark --state-dir "$STATE_COPY" --output "$RESULTS_DIR/anchor.jsonl" --segments 8128,16256 --seeds 5000000 --threads 4 --trials durable_kl_strong,durable_kl_broad
PYTHONPATH=computer python -m generalist_lm.learning_efficiency_benchmark --state-dir "$STATE_COPY" --output "$RESULTS_DIR/objective.json" --gradient-probe-only --threads 4
PYTHONPATH=computer python -m generalist_lm.learning_efficiency_benchmark --state-dir "$STATE_COPY" --output "$RESULTS_DIR/geometry.json" --geometry-probe-only --threads 4
```

La matrice più estesa include ulteriori trial16k non eseguiti nell'archivio; config effettive esatte sono in ogni riga, insieme a checkpoint/slice hash. Eseguire sequenzialmente per timing comparabili. Per dati freschi aggiungere `--causal-documents` con JSONL reviewed training-only; nuove condizioni richiedono nuovo journal, non append a risultati di input diverso.

## 9. Rollout, rollback e monitoraggio

**Non effettuare rollout della learning policy: non c'è un candidato vincente.** Questo branch è per revisione degli strumenti e dell'evidenza. Anche un merge solo diagnostico potrebbe attivare workflow osservatori: verificare path filters prima di qualsiasi merge; nessun merge/deployment eseguito in questa task.

Per un futuro candidato validato: salvare nuova baseline corrente, congelare digest/source/data, richiedere stessa-copia paired multi-seed e sequenza candidate persistita8k/16k con gate invariati; poi run live limitato su lineage corrente con normali snapshot/verifier/rollback. Monitorare tempo totale e tempo training separati, actual new/replay labels, gradient pressure, acceptance e Wilson/window, repetition e headroom durable, slope, NLL/validation, unique/multiword/pathological, conversational trace, optimizer e checkpoint SHA. Arrestare la crescita se salute/headroom peggiorano; un fallimento del guard deve ripristinare tramite il rollback esistente checkpoint e contatori. Nessuna promotion basata sulla loss. Contraddizione live/offline falsifica la prova e richiede diagnosi, non modifica soglie.

Rollback di questi strumenti: revert del commit feature tramite Git, senza toccare generalist-state. Non è necessario alcun rollback del checkpoint live: il modello non è stato modificato. Per una futura modifica training utilizzare il normale snapshot precedente e verifier; non riscrivere manualmente stato/counters.

## 10. Consegna Git

Branch: `feature/generalist-learning-efficiency-v3`, creato dall'HEAD main indicato sopra. SHA finale e link della draft PR sono comunicati nella consegna e verificati contro HEAD remoto (non inseriti circolarmente nel contenuto del proprio commit). Nessun merge. L'obiettivo completo resta aperto: questo report documenta risultati negativi e limiti, non li presenta come successo.
