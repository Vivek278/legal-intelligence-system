# Phase 7: Prior Case Retrieval

Open a model folder, then choose an adaptation notebook. Each notebook includes setup, data loading, model definitions, training, evaluation, and submission generation for the selected experiment. Run top to bottom in a fresh kernel.

| Model | Adaptation notebooks |
|---|---|
| Model 4 | [Full fine-tuning](model4/full.ipynb), [LoRA](model4/lora.ipynb), [LoRMA](model4/lorma.ipynb), [Knowledge distillation](model4/kd.ipynb), [Ensemble](model4/ensemble.ipynb) |
| Model 6 | [Full fine-tuning](model6/full.ipynb), [LoRA](model6/lora.ipynb), [LoRMA](model6/lorma.ipynb), [Knowledge distillation](model6/kd.ipynb), [Ensemble](model6/ensemble.ipynb) |
| Model 7 | [Full fine-tuning](model7/full.ipynb), [LoRA](model7/lora.ipynb), [LoRMA](model7/lorma.ipynb), [Knowledge distillation](model7/kd.ipynb), [Ensemble](model7/ensemble.ipynb) |

The ensemble notebook runs its constituent methods in one kernel before averaging their predictions. Single-method notebooks run only their selected method (plus a teacher when needed for KD).

Each notebook includes configurable task-specific settings for training and inference.

## Running

See the root [README](../README.md) for prerequisites, checkpoint/tokenizer setup, dataset access, and results from the report.

Each experiment writes to its own output folder. Some submission files use shared filenames under the working directory; use separate sessions/working directories or download submissions after each run.
