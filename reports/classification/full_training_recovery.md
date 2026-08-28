# v2 Full QLoRA recovery report

Status: `running_resumed`

Audit time: 2026-08-21 (UTC+8 server time)

## Evidence before recovery

- Experiment: `artifacts/experiments/qwen3_4b_qlora_v2_full`
- No active trainer process was found during the initial read-only audit.
- The newest complete checkpoint was `checkpoint-500`.
- `trainer_state.json` recorded `global_step=500`, `epoch=0.13289478`, and `max_steps=7526`.
- The checkpoint contains the LoRA adapter, optimizer state, scheduler state, RNG state, scaler state, and trainer state.
- Validation at step 500 recorded Micro-F1 `0.825327866626843`. This is an intermediate validation result, not the final Full result.
- The progress value previously interpreted as `2568/3758` belonged to validation inference progress. It did not prove training completion.

## Data integrity

The processed-v2 inputs still match the hashes associated with the experiment:

| File | SHA256 |
|---|---|
| `data/processed_v2/train.jsonl` | `8d57c9719083eb544d08a7c4aed60b6942a94c0c094e43a43b9dc95524db23b0` |
| `data/processed_v2/validation.jsonl` | `9d5e763a50e37b58749ab996f4b83ff7ca7a62183a751bb8a90541568123dfa0` |
| `data/processed_v2/label_mapping.json` | `0595297115fd9df0e78d42a64dbb3d6286da9b807d3005a4278bfc3e51ca301f` |

The final test split has not been used by this recovery procedure.

## Recovery action

The first safe resume attempt was blocked by Transformers because PyTorch 2.5.1 is affected by CVE-2025-32434 when loading optimizer state with `torch.load`. No safety check was bypassed. The project Conda environment was upgraded to the official PyTorch `2.6.0+cu124` build and Triton 3.2.0. CUDA and the focused training tests passed after the upgrade.

Training was then resumed from `checkpoint-500` in detached screen session `legalmind_full_resume`. The live log confirmed that optimization continued at step 501 instead of restarting at step zero.

Recovery command:

```bash
cd /root/autodl-tmp/LegalMind-RAG
env OMP_NUM_THREADS=8 PYTHONPATH=src \
  /root/autodl-tmp/conda/envs/legalmind/bin/python scripts/train_qlora.py \
  --training-config configs/classification/qwen3_4b_qlora_full.yaml \
  --resume-from-checkpoint
```

Live log:

`artifacts/experiments/qwen3_4b_qlora_v2_full/resume_20260821.log`

The final training time, peak memory, throughput, best checkpoint, and validation Micro-F1 remain `pending_running`. The final test evaluation remains `not_run` until the validation configuration is frozen.
