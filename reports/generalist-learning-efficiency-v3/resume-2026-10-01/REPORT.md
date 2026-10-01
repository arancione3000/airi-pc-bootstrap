# Continuazione del 1 ottobre — esperimenti offline, obiettivo ancora aperto

Questo documento aggiorna il report della prima fase. Nessun candidato è autorizzato per il live: i successi isolati non si mantengono nelle sequenze. `baseline_start` della task resta quella del 30 settembre: 92.838,76 accepted/h, target minimo 278.516,28/h. La nuova fotografia serve a fissare il checkpoint dei nuovi esperimenti, non a cambiare il denominatore.

## Stato e corpus

Nuovo stato fissato: `f5c4c334afbef2c8437ffd9a26efee5ac517cd1d`, modello SHA256 `ab500e0170c35081b42f5337cb9b504b28dc44706b94f22b480cf29fffc4eebc`. Stesso lineage e architettura, 33.173.485 causal token persistiti, 79/491 segmenti accettati, 412 rollback (83,91%). NLL 2,0245527; repetition 0,2361575; headroom 0,0015631. `baseline_start.json` conserva tutti i campi disponibili e i digest verificati.

Sono state recuperate le sorgenti esatte inglese Tatoeba CC0 e OASST1 umano, verificando SHA e conteggi del manifest live: 60.180 documenti train e 3.146 validation con split originale. Il download italiano corrente non corrisponde alla SHA fissata: escluso. FineWeb non recuperato. Non è il corpus live completo. La validation riportata è un sottoinsieme deterministico di 32 blocchi, non la full validation del workflow. Riproduzione e provenance in `recover_pinned_corpus.py` e `recovered-corpus.json`; il file inglese CC0 è conservato perché la sorgente rolling cambia.

## Risultati controllati

43 trial indipendenti da checkpoint identico; journal compressi e tabella completa in `continuation-trials-summary.json`. Learning rate di base corretto 0,0003, context 128, seed 5.000.000, quattro thread. Le varianti cambiano obiettivi, parametri allenabili, quantità/normalizzazione del replay e learning rate per gruppo; non modificano i gate o i probe.

- Aumentare la penalità teacher-forced di repetition fino a 100 non crea headroom; tutti i sette trial sono respinti.
- Congelare FFN consente alcuni 8k con repetition invariata, ma i piccoli cambiamenti di NLL non costituiscono prova di apprendimento utile. Sbloccare solo FFN o solo down/up projection e ridurre il rate down non risolve la degenerazione. La FFN è coinvolta, ma non è dimostrata una migliore architettura a pari compute.
- Unlikelihood su continuazioni generate dal modello da prefissi esclusivamente train consente alcuni 8k. Non usa target positivi generati, teacher esterni o probe di evaluation. Con peso 10/100 il gradiente ausiliario domina: non è prova che il modello impari più velocemente dai nuovi token.
- Replay causale in testo normale dalla memoria train già persistita: 1.016 label replay per step contro 4.064 nuove (20% fisico), peso 0,25. Con otto prefissi e penalità autoregressiva 0,1 passa il primo 8k, migliora NLL di circa 0,00139 e validation subset di 0,000437, senza migliorare repetition. Le norme sono circa 2,46 causal, 1,05 replay e 3,87 ausiliario pesato al primo step. Migliore bilanciamento locale, stabilità successiva assente.

## Falsificazione tramite sequenze

Ogni segmento accettato alimenta il successivo nella sola directory temporanea offline; ogni rifiuto conserva esattamente il checkpoint precedente e non aumenta i token accettati. Il checkpoint live e il durable anchor non vengono scritti. La sequenza riparte deterministicamente dal pin se interrotta; non dichiara un resume dei pesi temporanei cancellati.

| Strategia | Taglie tentate | Segmenti accettati | Rollback | Evidenza |
|---|---|---:|---:|---|
| Embedding, replay normalizzato | 8k, 8k, 16k, 16k | 2/4 | 50% | 16k respinti; circa 147.275 accepted-equivalent/h end-to-end sulla sequenza |
| Unlikelihood autoregressiva 10, replay SFT | 8k, 8k, 16k, 16k, 32k, 32k | 1/6 | 83,33% | Secondo 8k già respinto |
| Replay causale 20%, unlikelihood 0,1 su otto prefissi | 8k, 8k, 16k, 16k, 32k, 32k | 1/6 | 83,33% | Secondo 8k già respinto |

Nessuna sequenza crea headroom o dimostra 16k stabile. Le taglie successive ai rifiuti sono diagnostica offline, non una proposta di ladder live. Nessun token degli esperimenti è stato persistito nel lineage live. Throughput offline accepted-equivalent non equivale a useful persistent learning.

## Sovrapposizione fra formati di holdout

Audit: 33 esempi della pool SFT ampia hanno contenuto coincidente con documenti corpus-validation, perché gli split originali sono basati rispettivamente sul digest di conversazione e documento. Pool elementary: zero; causal train contro corpus validation: zero. I nuovi trial filtrano quelle righe prima del campionamento, preservando tutte le popolazioni di evaluation. I vecchi risultati della validation subset con replay SFT ampio non sono affidabili e non giustificano una promotion. Gli stessi testi possono avere esposizione storica nel checkpoint live: questo sottoinsieme non va presentato come holdout certamente mai visto. I probe Phase-5 restano esclusi dal training. Il problema nella pipeline di produzione non è ancora corretto da questo branch.

## Codice, verifiche e stato del rollout

Esteso soltanto il benchmark offline e i suoi regression test: contributi e contatori separati, replay causale reale, congelamento prima della creazione dell'optimizer, split e digest verificati, validation subset separata, esclusione replay sovrapposto, sequenze con checkpoint/counter rollback-safe, penalità negative autoregressive, regolarizzazione prossimale sperimentale verso l'anchor esistente. Regolarizzazione prossimale: quattro dei cinque trial respinti; il solo trial accettato non crea headroom. Nei trial senza obiettivo autoregressivo la repetition cresce da 0,268 a 0,307 al crescere della forza. La distanza parametrica dall’anchor cala, ma la generazione peggiora: ipotesi non validata.

251 test superati, un backend opzionale saltato, nelle suite learning efficiency, Phase-5, LM, architecture e single lineage. Nessuna modifica di prodotto a training, model, runtime, workflow, gate, soglie, context, tokenizer, anchor o stato live. Nessun peso esterno. PR #200 deve restare draft, senza merge o rollout. Per questi strumenti non esiste un checkpoint live da ripristinare: il training corrente continua normalmente.

Prossimo criterio: creare prima headroom reale e mantenere qualità; poi sequenze ripetibili 8k/16k con confronto temporale completo. Se una nuova policy supera queste prove, solo allora implementare recovery/growth, test di compatibilità e primo run live limitato. Non dichiarare >=3x sulla base di un unico segmento fortunato.
