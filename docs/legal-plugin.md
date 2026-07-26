# Legal plugin

The legal classifier is an optional domain plugin, not a startup dependency.

## Required assets

- base model ID or local directory;
- PEFT adapter directory containing `adapter_config.json`;
- sequence-classification head state dict;
- JSON label-to-ID mapping with contiguous IDs starting at zero.

Configure the paths through `DOCMIND_LEGAL_*` environment variables. The project does
not copy or commit the legacy model weights.

## Safety and reproducibility

- imports of Torch, Transformers and PEFT occur only on first inference;
- `torch.load` uses CPU mapping and `weights_only=True` when supported;
- the classification-head state dict is loaded with `strict=True`;
- CPU/GPU is selected explicitly or through `auto`;
- inputs are truncated to the configured maximum length;
- the plugin status endpoint never loads the model.

## Existing experiment limitation

The legacy checkpoints reached step 930, about 0.077 epoch in the stored Trainer state.
No validation accuracy, Macro-F1, best metric or reproducible `test_result` artifact was
found. Therefore no legal classification score is claimed in this project until a new
evaluation run is completed.
