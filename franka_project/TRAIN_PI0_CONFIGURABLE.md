# Configurable PI0 training

Use `franka_project/scripts/train_pi0.py` for user-managed Franka training.
Unlike `train_pi0_full.py`, this entry point has no planning-stage whitelist.
An optional `stage` value is inert metadata and may also be omitted.

The trainer keeps the runtime safety checks that are independent of planning:

- dataset root/profile/task/rates/K/scope hash and effective 10D state/7D action stats;
- strict local PI0 checkpoint loading and canonical tied/pruned model graph;
- BF16/CUDA/world-size checks;
- first-backward finite/nonzero gradient coverage for the selected components;
- complete local model checkpoints, selected-parameter optimizer state, scheduler,
  per-rank RNG, and exact resume trainability signatures.
- exact resume locks both the dataset profile and `pi0_eef_stats.json` SHA; a
  normalization-stat change must start a new run.

## Trainability presets

Configure the selection under `model.trainability`:

```yaml
model:
  trainability:
    preset: action_expert_paligemma
    components: null
```

| Preset | Trainable components | Frozen components |
|---|---|---|
| `full` | all six components | none |
| `action_expert` | Action Expert + action/state/time projections | complete PaliGemma |
| `action_expert_paligemma` | multimodal projector + PaliGemma Transformer + Action Expert + projections | Vision Tower + tied text embedding |

The canonical local `pi0_base` audit gives:

| Preset | Trainable tensors | Trainable parameters | Fraction |
|---|---:|---:|---:|
| `full` | 776 | 3,238,048,528 | 100% |
| `action_expert` | 173 | 314,713,120 | 9.7192% |
| `action_expert_paligemma` | 338 | 2,298,958,880 | 70.9983% |

`action_expert` includes the five required PI0 projection modules; it does not
include the unused Expert LM output head, which is removed before the optimizer
is constructed.

## Custom component selection

`custom` accepts only the six audited atomic components. Arbitrary parameter
name patterns are intentionally unsupported.

```yaml
model:
  trainability:
    preset: custom
    components:
      - multimodal_projector
      - paligemma_transformer
      - action_expert
      - action_projections
```

Available components:

- `vision_encoder`
- `multimodal_projector`
- `text_embedding`
- `paligemma_transformer`
- `action_expert`
- `action_projections`

At startup the six groups must be disjoint and must cover every unique model
Parameter. Any unclassified/overlapping tensor fails before optimizer creation.
The PaliGemma LM-head weight is pointer-tied to `text_embedding`, so it cannot
accidentally remain trainable when that component is frozen.

## Move-cups run

The checked move-cups config defaults to `action_expert_paligemma`:

```bash
cd /m2v_intern/tujiahang/Projects/lerobot

env -u LD_LIBRARY_PATH \
  /ytech_milm_intern/tujiahang/miniconda3/envs/lerobot/bin/torchrun \
  --standalone --nproc_per_node=2 \
  franka_project/scripts/train_pi0.py \
  --config franka_project/configs/train/pi0_ActTrans_eef_move_cups_30hz.yaml
```

For a short invocation that retains the YAML scheduler horizon:

```bash
env -u LD_LIBRARY_PATH \
  /ytech_milm_intern/tujiahang/miniconda3/envs/lerobot/bin/torchrun \
  --standalone --nproc_per_node=2 \
  franka_project/scripts/train_pi0.py \
  --config franka_project/configs/train/pi0_ActTrans_eef_move_cups_30hz.yaml \
  --max-steps 2
```

Resume the same immutable run, including optimizer/scheduler/RNG state:

```bash
env -u LD_LIBRARY_PATH \
  /ytech_milm_intern/tujiahang/miniconda3/envs/lerobot/bin/torchrun \
  --standalone --nproc_per_node=2 \
  franka_project/scripts/train_pi0.py \
  --config franka_project/configs/train/pi0_ActTrans_eef_move_cups_30hz.yaml \
  --resume-run-dir /absolute/path/to/the/run
```

Changing the preset/component list requires a new run. Resume deliberately
rejects a different trainability signature so Adam moments cannot be assigned
to the wrong Parameter order. On restart, complete `.staging` checkpoints are
published automatically, incomplete staging directories are moved to a
recoverable `.abandoned-*` quarantine, and a valid checkpoint newer than the
pointer is adopted.
