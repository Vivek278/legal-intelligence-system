# Phase 3: Rhetorical Role Prediction

Open a model folder, then choose an adaptation notebook. Each notebook contains the original setup, data loading, model, training, evaluation, and submission code, with that experiment preselected. Run top to bottom in a fresh kernel.

| Model | Adaptation notebooks |
|---|---|
| Model 4 | [Full fine-tuning](model4/full.ipynb), [LoRA](model4/lora.ipynb), [LoRMA](model4/lorma.ipynb), [Knowledge distillation](model4/kd.ipynb) |
| Model 6 | [Full fine-tuning](model6/full.ipynb), [LoRA](model6/lora.ipynb), [LoRMA](model6/lorma.ipynb), [Knowledge distillation](model6/kd.ipynb) |
| Model 7 | [Full fine-tuning](model7/full.ipynb), [LoRA](model7/lora.ipynb), [LoRMA](model7/lorma.ipynb), [Knowledge distillation](model7/kd.ipynb) |

Representation, pooling, loss, thresholding, and other existing switches remain configurable inside the notebooks. Their original defaults are retained.

## Running and provenance

See the root [README](../README.md) for prerequisites, checkpoint/tokenizer setup, dataset access, and results from the report. Each notebook's first cell and `metadata.experiment` record its source and configuration selections.

Experiment output folders are separate, but original root-level submission names are preserved. Use separate sessions/working directories or download submissions after each run. Saved outputs were cleared; no training was rerun for this split.
