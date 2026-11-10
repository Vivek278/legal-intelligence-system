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

| Model | Family | Layers | Hidden size | Query / KV heads | Parameters | Training tokens |
|---|---|---:|---:|---|---:|---:|
| [1](phase1/capstone-m1.ipynb) | Causal Transformer with Plasticity-Adjusted Sampling | 12 | 768 | 12 / 12 | 109.53M | 1B |
| [2](phase1/capstone-m2.ipynb) | Encoder–decoder | 8 encoder + 4 decoder | 768 | 12 / 12 | 118.97M | 500M |
| [3](phase1/capstone-m3.ipynb) | Prefix language model | 12 | 768 | 12 / 12 | 109.53M | 500M |
| [4](phase1/capstone-m4.ipynb) | Causal GQA; alternating local/full attention | 16 | 768 | 12 / 2 | 122.12M | 1.5B |
| [5](phase1/capstone-m5.ipynb) | Larger causal Transformer | 12 | 1,024 | 16 / 16 | 201.09M | 1B |
| [6](phase1/capstone-m6.ipynb) | Larger causal GQA | 20 | 1,024 | 16 / 4 | 234.66M | 1B |
| [7](phase1/capstone-m7.ipynb) | BERT-style prefix LM with DiLoCo | 12 | 768 | 12 / 12 | 110.10M | Approximately 4B |

Models 1–6 use RoPE, RMSNorm, SwiGLU, and tied 32K vocabulary embeddings. Model 7 uses a 30,522-entry uncased WordPiece vocabulary, post-LayerNorm, GELU, and 512 learned positions. Its foundation weights are trained from scratch; the tokenizer follows the BERT vocabulary. The notebook lives in `phase1/`, although the report states that Model 7 was trained after Phase 1 ended and has no official Phase 1 result.

For Model 7, three replicas start from the same random weights and train on Supreme Court judgments, High Court judgments, and Acts with Tribunals, respectively. After 500 local AdamW steps, a Nesterov outer update merges their parameter changes. The merged weights begin the next round while replica optimizer states stay separate. Downstream inference uses one consolidated backbone.

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

Dataset files, tokenized corpora, trained checkpoints, and tokenizer artifacts are not bundled in this repository. Attach or provide them as described below.

## Running the notebooks

The workflows primarily target **Kaggle GPU notebooks**. The report describes Tesla T4 runs with approximately 15.6 GB GPU memory. Full fine-tuning, encoder–decoder training, and distillation can require different memory/time budgets. Batch sizes, gradient accumulation, seeds, early stopping, and mixed-precision settings are configured in each notebook.

1. Clone the repository and open the relevant phase index. For example, [Phase 2 Model 6 LoRA](phase2/model6/lora.ipynb) or [Phase 7 Model 7 Full](phase7/model7/full.ipynb).
2. Import that notebook into Kaggle, enable a GPU, and enable Internet for dataset downloads. Phase 6 also downloads the BERTScore evaluation model.
3. Attach the chosen foundation model's checkpoint (`best.pt`) and its matching tokenizer. Models 4/6 require their trained BPE tokenizer folder, including `tokenizer.json`; Model 7 uses its WordPiece tokenizer artifacts (`vocab.txt` and/or `tokenizer.json` with the associated configuration).
4. Add a Kaggle secret named **`HF`** with a Hugging Face token authorized for the dataset. Phases 4–7 also check `HF_TOKEN` in the environment. Phases 2–3 use Kaggle Secrets directly.
5. Set `CKPT_PATH` and `TOK_DIR` explicitly when several input datasets are attached. Automatic discovery can select a different model's file when several `best.pt` files or tokenizers are present. Confirm the selected paths and model/tokenizer sanity checks before training.
6. Review the notebook configuration. The model and adaptation method are already selected; other switches are configurable. `SMOKE = True` enables a small-run mode for checking the pipeline.
7. Run all cells from top to bottom in a fresh kernel. Download the prediction file, submission ZIP, checkpoints, and reports from the printed output paths.

The notebooks use PyTorch, NumPy, Transformers, Hugging Face Hub, tokenizers, tqdm, scikit-learn, and safetensors. Generation/evaluation notebooks additionally use BERTScore and ROUGE; foundation notebooks also use packages such as ijson and matplotlib. Follow each notebook's install cells. Several install cells constrain Transformers to `>=4.44,<5`; the repository does not provide a fully pinned environment.

For a local Jupyter environment, provide the required artifacts, install the notebook's dependencies, and adjust its Kaggle-specific paths and secret access.

Each notebook has a distinct experiment output directory, for example `/kaggle/working/phase2_model6_lora` or `/kaggle/working/cjpe_p4_model7_full`. Some phases also write ZIPs directly under `/kaggle/working`, so use separate sessions/working directories or download them after each run. Phase 6 writes a smoke prediction instead of a competition submission when smoke mode is active.

## Submission formats

| Phase | JSON content | JSON member in ZIP |
|---|---|---|
| 2 | Document ID → list of `[start, end, entity_type]` spans | `prediction.json` |
| 3 | Document ID → ordered list of sentence-role labels | `prediction.json` |
| 4 | List of `{id, pred}` records, with integer `pred` equal to 0 or 1 | `submission.json` |
| 5 | List of `{id, pred}` records, with a nonempty list of unique integer statute IDs | `prediction.json` |
| 6 | List of `{id, summary}` records | `prediction.json` |
| 7 | Query ID → exactly seven unique IDs from that split's candidate pool | `prediction.json` |
