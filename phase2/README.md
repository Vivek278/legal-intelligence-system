# Phase 2: Legal Named Entity Recognition

Open a model folder, then choose an adaptation notebook. Each notebook includes setup, data loading, model definitions, training, evaluation, and submission generation for the selected experiment. Run top to bottom in a fresh kernel.

| Model | Adaptation notebooks |
|---|---|
| Model 4 | [Full fine-tuning](model4/full.ipynb), [Partial fine-tuning](model4/partial.ipynb), [LoRA](model4/lora.ipynb), [LoRMA](model4/lorma.ipynb) |
| Model 6 | [Full fine-tuning](model6/full.ipynb), [Partial fine-tuning](model6/partial.ipynb), [LoRA](model6/lora.ipynb), [LoRMA](model6/lorma.ipynb), [Houlsby adapters](model6/houlsby.ipynb), [Pfeiffer adapters](model6/pfeiffer.ipynb), [Prefix tuning](model6/prefix.ipynb), [Prompt tuning](model6/prompt.ipynb), [BitFit](model6/bitfit.ipynb), [Knowledge distillation](model6/kd.ipynb), [Ensemble](model6/ensemble.ipynb) |

Models 4 and 6 support Full, Partial, LoRA, and LoRMA adaptation. Model 6 also supports Houlsby adapters, Pfeiffer adapters, Prefix tuning, Prompt tuning, BitFit, knowledge distillation, and a logit ensemble.

The ensemble notebook runs its constituent methods in one kernel before averaging their predictions. Single-method notebooks run only their selected method (plus a teacher when needed for KD).

Each notebook includes configurable task-specific settings for training and inference.

## Running

See the root [README](../README.md) for prerequisites, checkpoint/tokenizer setup, dataset access, and results from the report.

Each experiment writes to its own output folder. Some submission files use shared filenames under the working directory; use separate sessions/working directories or download submissions after each run.
