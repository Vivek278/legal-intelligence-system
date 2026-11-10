# Phase 2: Legal Named Entity Recognition

Open a model folder, then choose an adaptation notebook. Each notebook contains the original setup, data loading, model, training, evaluation, and submission code, with that experiment preselected. Run top to bottom in a fresh kernel.

| Model | Adaptation notebooks |
|---|---|
| Model 4 | [Full fine-tuning](model4/full.ipynb), [Partial fine-tuning](model4/partial.ipynb), [LoRA](model4/lora.ipynb), [LoRMA](model4/lorma.ipynb) |
| Model 6 | [Full fine-tuning](model6/full.ipynb), [Partial fine-tuning](model6/partial.ipynb), [LoRA](model6/lora.ipynb), [LoRMA](model6/lorma.ipynb), [Houlsby adapters](model6/houlsby.ipynb), [Pfeiffer adapters](model6/pfeiffer.ipynb), [Prefix tuning](model6/prefix.ipynb), [Prompt tuning](model6/prompt.ipynb), [BitFit](model6/bitfit.ipynb), [Knowledge distillation](model6/kd.ipynb), [Ensemble](model6/ensemble.ipynb) |

Models 4 and 6 use the four core methods. The six additional methods and their logit ensemble use the existing Model 6 configuration. In particular, the extra-method notebook's `KD_KEEP` map is specific to its 20-layer backbone. No Model 4 distillation map or Model 7 NER implementation has been invented.

The ensemble notebook keeps the original in-memory dependencies: it runs the constituent methods before averaging their predictions. Single-method notebooks run only their selected method (plus a teacher when needed for KD).

Representation, pooling, loss, thresholding, and other existing switches remain configurable inside the notebooks. Their original defaults are retained.

## Running and provenance

See the root [README](../README.md) for prerequisites, checkpoint/tokenizer setup, dataset access, and results from the report. Each notebook's first cell and `metadata.experiment` record its source and configuration selections.

Experiment output folders are separate, but original root-level submission names are preserved. Use separate sessions/working directories or download submissions after each run. Saved outputs were cleared; no training was rerun for this split.
