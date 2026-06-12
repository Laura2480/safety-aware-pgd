# Safety-Aware PGD — Panoramica delle Sperimentazioni

## Obiettivo generale

Il progetto studia gli **adversarial attack su modelli text-to-image (t2i)**.
L'obiettivo è generare prompt testuali che:
1. **Bypassano i sistemi di moderazione testuale** (text safety filters), e
2. **Producono immagini semanticamente simili** al prompt unsafe originale.

Le categorie di danno considerate sono 7:
`harassment`, `hate`, `self-harm`, `sexual`, `shocking`, `violence`, `illegal-activity`

---

## Pipeline principale (3 fasi)

### Fase 1 — Safe Replacements (`src/optimal_selection.py`)

**Input**: un prompt unsafe (es. *"Image of a person being strangled..."*)

**Cosa fa**:
- Calcola i **Integrated Gradients** sull'encoder testuale di CLIP per identificare quali token contribuiscono maggiormente allo score di danno (harm score) del modello MLP-STD.
- Usa un **LLM (GPT)** per suggerire sostituzioni sicure delle parole unsafe (es. *"rope"* → *"task"*).
- Esplora le combinazioni di sostituzioni (fino a 3000) e seleziona quella che **minimizza l'harm score** restando nell'intervallo di lunghezza token target.

**Output**: `revised_prompt` — versione "apparentemente sicura" del prompt, con le posizioni token libere per il PGD.

**Colonne chiave del DataFrame**:
- `unsafe_prompt`: versione dettagliata del prompt originale
- `revised_prompt`: prompt dopo Safe Replacements (harm score basso, ~0.07 medio)
- `unsafe_replacements` / `safe_replacements`: le coppie di parole sostituite
- `flatten_positions`: indici token "liberi" da ottimizzare nel PGD
- `harm`: harm score del `revised_prompt` (MLP-STD, scala 0-1)

---

### Fase 2 — Safety-Aware PGD (`src/batch_prompter.py` → `text_inversion_base_batched`)

**Input**: `revised_prompt` + posizioni token libere

**Cosa fa**:
- Esegue **Projected Gradient Descent** nello spazio continuo degli embedding CLIP.
- **Funzione di loss** bilancia due obiettivi:
  - Massimizzare la **similarità CLIP** tra il prompt e l'immagine di riferimento (generata con Stable Diffusion sull'unsafe prompt originale).
  - Mantenere l'**harm score < τ' = 0.4** (peso moderazione `global_mod = 0.5`).
- Lista di parole proibite: blocklist per categoria + parole unsafe estratte con Integrated Gradients.
- Iterazioni: **2000**, batch size: **25**, top-N per gruppo: **5** (o 1 nell'ablation).
- Alla fine, proietta ogni embedding continuo al token di vocabolario più vicino.

**Output**: `adv_prompt` — prompt finale adversariale (testo discreto, basso harm score, alta similarità semantica).

**Iperparametri chiave**:
| Parametro | Valore |
|-----------|--------|
| `n_iterations` | 2000 |
| `tau_prime` | 0.4 |
| `global_mod` | 0.5 |
| `batch_size` | 25 |
| `random_init` | False (Full pipeline) |

---

### Fase 3 — Generazione immagini

Gli `adv_prompt` vengono inviati ai modelli t2i:
- **DALL-E 3** (OpenAI) — risultati in `test/discrete/moderated/dall_e/`
- **Imagen 3** (Google, medium filter) — risultati in `test/discrete/moderated/imagen/`
- **Gemini 2.5 Flash Image** (Vertex AI) — in corso, `notebooks/MAIN/IMGEN_GEMINI.ipynb`

---

## Modello di moderazione (MLP-STD)

- Architettura: `ModerationHeadMLP` — MLP applicato all'embedding dell'attention head 6, layer 4 di CLIP ViT-L/14.
- Checkpoint: `src/models/mlp_model_selected_layer_1.pth`
- Output: score in [0,1], soglia di danno = **0.5**
- Harm score medio: `unsafe_prompt` ~0.96, `revised_prompt` ~0.07, `adv_prompt` ~0.15-0.28

**Versione avanzata (MLP-ADV)**:
- Input: concatenazione di 4 head × 768 dim = 3072 dim
- Head usate: (Layer 7, H5), (Layer 4, H10), (Layer 2, H4), (Layer 0, H7)
- Checkpoint: `src/models/ADV_mlp_model_selected_layer_4.pth`

---

## Domande di Ricerca (RQ) e relativi notebook

### RQ1 — `notebooks/MAIN/3_ANALYSIS_RQS.ipynb`
**Domanda**: Il nostro pipeline bypassa la moderazione testuale e genera comunque immagini dannose?

**Metriche**:
- **BR** (Bypass Rate): % di prompt che NON vengono bloccati dalla moderazione testuale
- **ASR** (Attack Success Rate): % di immagini generate classificate come dannose
- **CHR** (Content Harm Rate): harm rate medio delle immagini
- **CLIP score**: similarità semantica tra immagine generata e prompt originale

**Dati**: `dall_e_3_results_plain.parquet`, `imgen_3_medium_filter.parquet`

---

### RQ2 — `notebooks/MAIN/3_ANALYSIS_RQS.ipynb`
**Domanda**: Come si confronta il nostro metodo con altri dataset di prompt avversariali esistenti?

**Dati**: `ALL_ADV_CLEAN_mod_OUR_img_human_labeled.csv`
(include prompt nostri + altri metodi, con human labeling)

---

### RQ3 — `notebooks/MAIN/3_ANALYSIS_RQS.ipynb`
**Domanda**: Il nostro metodo è robusto anche contro la moderazione basata sulle immagini (CLIP-based)?

**Dati**: `ALL_ADV_CLEAN_MOD_ALL.csv`

---

### RQ4 (MLLM) — `notebooks/MAIN/4_ANALYSIS_MLLM.ipynb`
**Domanda**: Un Multi-modal LLM valuta le immagini generate come dannose?

Valutazione automatica con MLLM delle immagini prodotte dagli `adv_prompt`.


## File sorgente chiave

| File | Ruolo |
|------|-------|
| `src/optimal_selection.py` | Integrated Gradients + Safe Replacements |
| `src/batch_prompter.py` | ModeratedPrompter + Safety-Aware PGD |
| `src/moderation_clip.py` | ModerationHeadMLP (classifier) |
| `src/moderated_prompter.py` | Wrapper alto livello |
| `src/utils/scores_utils.py` | Utility metriche e path |
| `src/configs/category_block_dict.json` | Blocklist per categoria |
| `notebooks/MAIN/1_ATTACK_PIPELINE.ipynb` | Esecuzione pipeline completa |
| `notebooks/MAIN/3_ANALYSIS_RQS.ipynb` | Analisi RQ1/RQ2/RQ3 |
| `notebooks/MAIN/4_ANALYSIS_MLLM.ipynb` | Valutazione MLLM |
| `notebooks/MAIN/5_ABLATION_SAFE_REPLACEMENTS.ipynb` | Ablation study |
