# In-House Legal Intelligence System

A CS781 capstone project by **Vivek Chandwani (241183), IIT Kanpur**, covering seven legal NLP tasks: foundation language modeling, entity extraction, rhetorical roles, judgment prediction, statute identification, summarization, and prior case retrieval.

The project trains legal-domain foundation models from random initialization and adapts them with full fine-tuning, LoRA, LoRMA, and knowledge distillation. Its central experiment is **Model 7**: three domain-specific replicas periodically consolidate their weights through DiLoCo into a single 110.10M-parameter backbone. The repository contains research notebooks for training and competition submissions; deployment remains future work.

The [project report](CS781-CP-Vivek-Chandwani-241183-report.pdf) describes the methodology, results, error analysis, references, and LLM usage declaration.

## Tasks and reported results

Official scores and ranks are from report Table 1, recorded on **9 November 2026**. Winning model/method selections are from Tables 5–11.

| Phase | Task | Best reported configuration | Primary metric | Score | Rank |
|---|---|---|---|---:|---:|
| [1](phase1/) | Legal continuation / LM pretraining | Model 4 | BERTScore F1 | 0.838973 | 20 |
| [2](phase2/README.md) | Legal named entity recognition | Model 6 + LoRA | Strict macro-F1 | 0.597164 | 21 |
| [3](phase3/README.md) | Rhetorical role prediction | Model 7 + LoRMA | Macro-F1 | 0.525001 | 42 |
| [4](phase4/README.md) | Court judgment prediction | Model 7 + Full | Macro-F1 | 0.813447 | 12 |
| [5](phase5/README.md) | Legal statute identification | Model 7 + Full | Macro-F1 | 0.412709 | 28 |
| [6](phase6/README.md) | Abstractive legal summarization | Model 7 + Full | BERTScore F1 | 0.850100 | 23 |
| [7](phase7/README.md) | Prior case retrieval | Model 7 + Full | Micro-F1@7 | 0.4743059657 | 18 |

The report records **68 valid submissions** across all seven phases. Model 7 leads the primary metric in Phases 3–7. The preferred adaptation varies by task: LoRA for NER, LoRMA for rhetorical roles, and full fine-tuning for Phases 4–7.

## Complete phase results

Results from report Appendix A (Tables 13–19). **Bold** marks the highest value in each metric column. Model 7 was trained after Phase 1 and has no Phase 1 result.

### Phase 1: Legal continuation (Table 13)

| Model | Method | BERTScore F1 | ROUGE-L F1 | BLEU-4 |
| --- | --- | --- | --- | --- |
| M1 | Pretrain | 0.831894 | 0.191080 | 0.043804 |
| M2 | Pretrain | 0.828659 | 0.191965 | 0.040064 |
| M3 | Pretrain | 0.827758 | **0.194339** | **0.046634** |
| M4 | Pretrain | **0.838973** | 0.188266 | 0.038554 |
| M5 | Pretrain | 0.833433 | 0.191161 | 0.043302 |
| M6 | Pretrain | 0.833631 | 0.193479 | 0.044605 |

### Phase 2: Legal named entity recognition (Table 14)

| Model | Method | Strict Macro-F1 | Strict Macro-Precision | Strict Macro-Recall |
| --- | --- | --- | --- | --- |
| M6 | Full | 0.548211 | 0.548731 | 0.547692 |
| M6 | LoRA | **0.597164** | 0.575307 | **0.637822** |
| M6 | LoRMA | 0.563143 | 0.579659 | 0.564125 |
| M6 | KD | 0.496312 | 0.538687 | 0.489381 |
| M4 | Full | 0.507839 | 0.539832 | 0.499284 |
| M4 | LoRA | 0.529345 | 0.545308 | 0.528961 |
| M4 | LoRMA | 0.525853 | **0.601757** | 0.497326 |
| M4 | KD | 0.494830 | 0.510329 | 0.498396 |

### Phase 3: Rhetorical role prediction (Table 15)

| Model | Method | Macro-F1 | Macro-Precision | Macro-Recall |
| --- | --- | --- | --- | --- |
| M7 | LoRMA | **0.525001** | 0.529999 | **0.542281** |
| M7 | LoRA | 0.513997 | 0.554742 | 0.530990 |
| M7 | Full | 0.505800 | 0.552000 | 0.535000 |
| M7 | KD | 0.493800 | 0.550000 | 0.542000 |
| M6 | LoRMA | 0.522814 | 0.531605 | 0.538095 |
| M6 | LoRA | 0.513336 | **0.566227** | 0.516802 |
| M6 | Full | 0.499972 | 0.554456 | 0.528415 |
| M6 | KD | 0.492541 | 0.518775 | 0.519101 |
| M4 | LoRMA | 0.517230 | 0.527803 | 0.528280 |
| M4 | LoRA | 0.509134 | 0.545275 | 0.533949 |
| M4 | Full | 0.490116 | 0.500390 | 0.511943 |
| M4 | KD | 0.489623 | 0.548421 | 0.541299 |

### Phase 4: Court judgment prediction (Table 16)

| Model | Method | Macro-F1 | Precision | Recall |
| --- | --- | --- | --- | --- |
| M7 | Full | **0.813447** | **0.818061** | 0.808399 |
| M7 | LoRA | 0.805862 | 0.786064 | **0.843832** |
| M7 | LoRMA | 0.800110 | 0.787955 | 0.824147 |
| M7 | KD | 0.795173 | 0.790731 | 0.799665 |
| M6 | Full | 0.798526 | 0.801347 | 0.795724 |
| M6 | LoRA | 0.791384 | 0.774921 | 0.808562 |
| M6 | LoRMA | 0.786927 | 0.777613 | 0.796349 |
| M6 | KD | 0.781463 | 0.778294 | 0.784658 |
| M4 | Full | 0.793281 | 0.796714 | 0.789877 |
| M4 | LoRA | 0.786215 | 0.769483 | 0.803690 |
| M4 | LoRMA | 0.781846 | 0.772638 | 0.791179 |
| M4 | KD | 0.776592 | 0.773418 | 0.779792 |

### Phase 5: Legal statute identification (Table 17)

| Model | Method | Macro-F1 | Micro-F1 | Subset Accuracy |
| --- | --- | --- | --- | --- |
| M7 | Full | **0.412709** | 0.505456 | 0.065289 |
| M7 | LoRA | 0.404459 | 0.495659 | 0.052155 |
| M7 | LoRMA | 0.403452 | 0.494486 | 0.048852 |
| M7 | KD | 0.398731 | 0.489275 | 0.043617 |
| M6 | Full | 0.412469 | **0.506603** | **0.066134** |
| M6 | LoRA | 0.403817 | 0.494823 | 0.050936 |
| M6 | LoRMA | 0.402614 | 0.493671 | 0.047528 |
| M6 | KD | 0.397926 | 0.488416 | 0.042391 |
| M4 | Full | 0.406448 | 0.497815 | 0.055918 |
| M4 | LoRA | 0.397638 | 0.486392 | 0.044217 |
| M4 | LoRMA | 0.396284 | 0.485137 | 0.041863 |
| M4 | KD | 0.391752 | 0.479826 | 0.037294 |

### Phase 6: Abstractive summarization (Table 18)

| Model | Method | BERTScore F1 | ROUGE-1 | ROUGE-L |
| --- | --- | --- | --- | --- |
| M7 | Full | **0.850100** | 0.516457 | **0.273530** |
| M7 | LoRMA | 0.845170 | **0.537524** | 0.269197 |
| M7 | LoRA | 0.835926 | 0.421638 | 0.224571 |
| M7 | KD | 0.826851 | 0.335750 | 0.186306 |
| M6 | Full | 0.843672 | 0.487315 | 0.261842 |
| M6 | LoRMA | 0.839214 | 0.472681 | 0.255936 |
| M6 | LoRA | 0.832547 | 0.389426 | 0.213758 |
| M6 | KD | 0.827541 | 0.335053 | 0.186239 |
| M4 | Full | 0.837233 | 0.378011 | 0.218468 |
| M4 | LoRMA | 0.831684 | 0.342716 | 0.195827 |
| M4 | LoRA | 0.826392 | 0.309548 | 0.174291 |
| M4 | KD | 0.821853 | 0.277883 | 0.156663 |

### Phase 7: Prior case retrieval (Table 19)

| Model | Method | Micro-F1@7 | Micro-Precision@7 | Micro-Recall@7 |
| --- | --- | --- | --- | --- |
| M7 | Full | **0.4743059657** | **0.4840265220** | **0.4649681529** |
| M7 | LoRA | 0.4678086237 | 0.4773960217 | 0.4585987261 |
| M7 | LoRMA | 0.4518606025 | 0.4611211573 | 0.4429646786 |
| M7 | KD | 0.3933845245 | 0.4014466546 | 0.3856398379 |
| M6 | Full | 0.4494979327 | 0.4587100663 | 0.4406485235 |
| M6 | LoRA | 0.4422867319 | 0.4513728462 | 0.4335609174 |
| M6 | LoRMA | 0.4289326157 | 0.4381062948 | 0.4201357826 |
| M6 | KD | 0.3948127361 | 0.4032718594 | 0.3867014258 |
| M4 | Full | 0.4384169273 | 0.4479026158 | 0.4293257816 |
| M4 | LoRA | 0.4275381649 | 0.4367148295 | 0.4187426931 |
| M4 | LoRMA | 0.4158267392 | 0.4249136857 | 0.4072679418 |
| M4 | KD | 0.3922031896 | 0.4002411091 | 0.3844817603 |

## Repository layout

Phases 2–7 have **one notebook per supported model/adaptation method**, organized under the relevant model folder. Each notebook includes its own setup, data preparation, model definitions, training, evaluation, and submission generation. Representation, pooling, loss, and threshold switches remain configurable within the notebooks.

```text
legal-intelligence-system/
├── README.md
├── CS781-CP-Vivek-Chandwani-241183-report.pdf
├── phase1/                         # Foundation model training and continuation
│   ├── capstone-m1.ipynb ... capstone-m7.ipynb
│   └── capstone-m4-continue.ipynb
├── phase2/                         # Legal NER
│   ├── README.md
│   ├── model4/                     # full, partial, lora, lorma
│   └── model6/                     # Core + six extra methods + ensemble
├── phase3/                         # Rhetorical roles
│   ├── README.md
│   └── model4/, model6/, model7/    # full, lora, lorma, kd
├── phase4/                         # Judgment prediction; same model/method layout
├── phase5/                         # Statute identification; same model/method layout
├── phase6/                         # Summarization; same model/method layout
└── phase7/                         # Retrieval; same layout + ensemble per model
```

| Phase | Models | Notebook methods | Notebooks |
|---|---|---|---:|
| [2](phase2/README.md) | 4, 6 | Both: Full, Partial, LoRA, LoRMA. Model 6 additionally: Houlsby, Pfeiffer, Prefix, Prompt, BitFit, KD, Ensemble | 15 |
| [3](phase3/README.md) | 4, 6, 7 | Full, LoRA, LoRMA, KD | 12 |
| [4](phase4/README.md) | 4, 6, 7 | Full, LoRA, LoRMA, KD | 12 |
| [5](phase5/README.md) | 4, 6, 7 | Full, LoRA, LoRMA, KD | 12 |
| [6](phase6/README.md) | 4, 6, 7 | Full, LoRA, LoRMA, KD | 12 |
| [7](phase7/README.md) | 4, 6, 7 | Full, LoRA, LoRMA, KD, Ensemble | 15 |

There are **78 downstream notebooks** in total. `kd.ipynb` selects the `kd` or `knowledge_distillation` configuration appropriate to that phase. Distillation notebooks train or reuse a teacher before training the student. Ensemble notebooks run their constituent methods in one kernel before averaging their predictions.

## Foundation models

Architecture and training-exposure figures are from report Appendix B. Token counts include repeated exposure; they are not counts of unique tokens.

Table 20: Foundation architectures and training exposure. Tokens per parameter are approximate.

| Model | Family | Layers | Hidden size | Query / KV heads | FFN size | Parameters | Training tokens | Tokens per parameter |
|---|---|---:|---:|---|---:|---:|---:|---:|
| [1](phase1/capstone-m1.ipynb) | Causal Transformer with Plasticity-Adjusted Sampling | 12 | 768 | 12 / 12 | 2,048 | 109.53M | 1B | 9.13 |
| [2](phase1/capstone-m2.ipynb) | Encoder–decoder | 8 encoder + 4 decoder | 768 | 12 / 12 | 2,048 | 118.97M | 500M | 4.20 |
| [3](phase1/capstone-m3.ipynb) | Prefix language model | 12 | 768 | 12 / 12 | 2,048 | 109.53M | 500M | 4.57 |
| [4](phase1/capstone-m4.ipynb) | Causal GQA; alternating local/full attention | 16 | 768 | 12 / 2 | 2,048 | 122.12M | 1.5B | 12.28 |
| [5](phase1/capstone-m5.ipynb) | Larger causal Transformer | 12 | 1,024 | 16 / 16 | 3,200 | 201.09M | 1B | 4.97 |
| [6](phase1/capstone-m6.ipynb) | Larger causal GQA | 20 | 1,024 | 16 / 4 | 2,432 | 234.66M | 1B | 4.26 |
| [7](phase1/capstone-m7.ipynb) | BERT-style prefix LM with DiLoCo | 12 | 768 | 12 / 12 | 3,072 | 110.10M | Approximately 4B | Approximately 36.3 |

Models 1–6 use RoPE, RMSNorm, SwiGLU, and tied 32K vocabulary embeddings. Model 7 uses a 30,522-entry uncased WordPiece vocabulary, post-LayerNorm, GELU, and 512 learned positions. Its foundation weights are trained from scratch; the tokenizer follows the BERT vocabulary. The notebook lives in `phase1/`, although the report states that Model 7 was trained after Phase 1 ended and has no official Phase 1 result.

For Model 7, three replicas start from the same random weights and train on Supreme Court judgments, High Court judgments, and Acts with Tribunals, respectively. After 500 local AdamW steps, a Nesterov outer update merges their parameter changes. The merged weights begin the next round while replica optimizer states stay separate. Downstream inference uses one consolidated backbone.

## Additional local results

Report Appendix C (Tables 21–23) reports local validation experiments separately from the competition results above.

### Additional NER adaptation methods (Table 21)

These methods use 62 training documents and eight validation documents. Trainable parameters are in millions; time is in minutes.

| Method | Precision | Recall | F1 | Trainable M | Minutes |
| --- | --- | --- | --- | --- | --- |
| kd | 0.5327 | 0.5610 | 0.5312 | 133.74 | 34.3 |
| pfeiffer | 0.5448 | 0.5368 | 0.5297 | 2.67 | 11.0 |
| ensemble | 0.5555 | 0.5135 | 0.5215 | 0.00 | 0.0 |
| houlsby | 0.4962 | 0.5246 | 0.4958 | 2.69 | 13.6 |
| bitfit | 0.4853 | 0.5068 | 0.4869 | 0.24 | 22.8 |
| prompt | 0.4830 | 0.3947 | 0.4172 | 0.06 | 26.1 |
| prefix | 0.4039 | 0.4136 | 0.3999 | 0.35 | 26.9 |

### NER performance by entity type (Table 22)

Precision, recall, and F1 use the local validation set. TP, FP, and FN denote true positives, false positives, and false negatives.

| Type | Precision | Recall | F1 | TP | FP | FN |
| --- | --- | --- | --- | --- | --- | --- |
| APP | 0.323 | 0.122 | 0.177 | 10 | 21 | 72 |
| RESP | 0.120 | 0.035 | 0.055 | 3 | 22 | 82 |
| A.COUNSEL | 0.667 | 0.381 | 0.485 | 8 | 4 | 13 |
| R.COUNSEL | 0.294 | 0.417 | 0.345 | 5 | 12 | 7 |
| JUDGE | 0.696 | 0.762 | 0.727 | 16 | 7 | 5 |
| WIT | 0.684 | 0.464 | 0.553 | 13 | 6 | 15 |
| AUTH | 0.598 | 0.603 | 0.601 | 73 | 49 | 48 |
| COURT | 0.898 | 0.876 | 0.887 | 106 | 12 | 15 |
| STAT | 0.850 | 0.866 | 0.858 | 233 | 41 | 36 |
| PREC | 0.382 | 0.667 | 0.486 | 60 | 97 | 30 |
| DATE | 0.940 | 0.983 | 0.961 | 173 | 11 | 3 |
| CASENO | 0.357 | 0.185 | 0.244 | 10 | 18 | 44 |

### Continuation decoding (Table 23)

All six configurations use the same tokenizer and the same 64 validation documents. Polishing adds continued pretraining and continuation fine-tuning. Local scores are rounded to six decimal places.

| Model and decoder | BERTScore F1 | ROUGE-L F1 | BLEU |
| --- | --- | --- | --- |
| Original greedy | 0.812482 | 0.200492 | 0.056606 |
| Original greedy, penalty 1.05 | 0.813853 | 0.193848 | 0.055240 |
| Original sampling | 0.825715 | 0.196533 | 0.059869 |
| Polished greedy | 0.813343 | 0.203300 | 0.061148 |
| Polished greedy, penalty 1.05 | 0.818434 | 0.200073 | 0.055246 |
| Polished sampling | 0.830290 | 0.198853 | 0.058406 |

## Downstream approach

| Phase | Implementation |
|---|---|
| 2 | 25-tag BIO classifier over 12 entity types; optional CRF; 512-token overlapping windows with stride 256; center-weighted emission averaging; document-level decoding to exact character spans |
| 3 | One of 13 rhetorical roles per sentence; three previous sentences and one following sentence; domain tags and relative-position buckets; joint Corporate Law / Information Technology training |
| 4 | Binary judgment classifier; six selected 384-token windows; masked token means followed by attention pooling; class weighting and dev-selected decision threshold |
| 5 | Multi-label statute prediction; six document windows; label-wise attention; weighted BCE with focal/ASL alternatives; global or per-label threshold calibration and a nonempty prediction fallback |
| 6 | Foundation-initialized encoder and autoregressive decoder with cross-attention; four 256-token source windows; four-beam generation; sequence-level distillation |
| 7 | Shared bi-encoder; InfoNCE with in-batch and TF-IDF hard negatives; document-window pooling; optional dense/TF-IDF hybrid scoring; exactly seven unique candidate IDs per query |

Full fine-tuning updates the whole backbone. LoRA uses additive low-rank updates; LoRMA uses multiplicative output-space updates. The default adapter rank is 8 with alpha 16. Distillation uses task-specific teacher targets: token emissions, class probabilities, multi-label probabilities, generated summaries, or retrieval embeddings and score distributions.

The tasks independently adapt foundation checkpoints. Later phases do not require running the earlier downstream phases first.

## Data

The notebooks load the official [Exploration-Lab/CS-781-Capstone dataset](https://huggingface.co/datasets/Exploration-Lab/CS-781-Capstone). Foundation training uses Acts, High Court judgments, Supreme Court judgments, and Tribunal decisions. Report Section 3 gives the following task sizes; training counts precede local holdout selection.

| Phase | Training | Validation | Test |
|---|---|---|---|
| 1 | Four legal corpora | Document holdout | 1,000 prompts |
| 2 | 70 documents in two folds | Eight documents held out locally | 35 documents |
| 3 | CL: 40 documents; IT: 40 documents | CL: 5; IT: 5 | CL: 5; IT: 5 |
| 4 | 37,387 cases | 1,988 cases | 1,517 cases |
| 5 | 42,750 cases | 10,181 cases | 13,019 cases |
| 6 | 7,030 document/summary pairs | 10% document-disjoint local holdout | 100 documents |
| 7 | 827 queries / 4,320 candidates | 118 queries / 1,023 candidates | 237 queries / 1,727 candidates |

Dataset files, tokenized corpora, trained checkpoints, and tokenizer artifacts are not bundled in this repository.
