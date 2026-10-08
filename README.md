# one_step_fm_based_vla

one_step_fm_based_vla is the V15 LIBERO vision-language-action model with  Mamba. This repository contains the training and evaluation code for the run evaluated from 85k through 99k steps. The best checkpoint is step 95,000.

## Model and training

- Two DINOv3 camera views, BERT language features, R3M features, and robot state feed the policy.
- A 12-step history encoder uses  Mamba. R3M belief and tacit features condition the flow-matching action head.
- Robot state is zero-padded into the action-head condition. The action chunk length is 12; the evaluated rollout executes 10 actions before replanning.
- The training objective is action flow matching. There is no KL constraint or auxiliary future-state prediction.
- The published run trained from scratch for 100,000 optimizer steps using four GPUs.

## Results

The official LIBERO evaluation used 500 episodes for each suite at each checkpoint. All results from 85k to 99k are in [85k_99k_all4.json](experiments/libero/results/85k_99k_all4.json).

| Suite | Step 95k successes | Success rate |
| --- | ---: | ---: |
| LIBERO-Spatial | 496 / 500 | 99.2% |
| LIBERO-Object | 500 / 500 | 100.0% |
| LIBERO-Goal | 491 / 500 | 98.2% |
| LIBERO-10 | 475 / 500 | 95.0% |
| **All four** | **1962 / 2000** | **98.10%** |



## Checkpoint

Download all 26 assets from the [v15-95k release](https://github.com/xzhuzhu/one_step_fm_based_vla/releases/tag/v15-95k), then reconstruct and verify:

```bash
cat one_step_fm_based_vla_95k.pth.part-* > one_step_fm_based_vla_95k.pth
printf '%s  %s\n' '63383e42566ff5f3eef88ef5025b343797ef79d7a5325c3cb6e098e6b0df8710' 'one_step_fm_based_vla_95k.pth' | sha256sum -c -
```

The full training checkpoint includes the model and optimizer state. It was checked against this source tree with zero missing or unexpected model keys. The 26 release assets are split because one checkpoint exceeds GitHub's per-asset size limit.

## Setup

Use Python 3.10+ with a CUDA-enabled PyTorch installation. Install this package and its LIBERO dependencies:

```bash
pip install -e '.[libero]'
```

Provide local copies of DINOv3 ViT-B/16, `bert-base-uncased`, and the R3M ResNet-18 backbone. The default paths are `pretrained/dinov3-vitb16`, `pretrained/bert-base-uncased`, and `pretrained/r3m-resnet18/backbone.pth`. These pretrained files and the LIBERO RLDS datasets are not bundled in Git.

## Train

Set `DATASET_DIRS` to the four comma-separated LIBERO RLDS directories. The launcher defaults to GPUs 0,1,2,3; set `TRAIN_GPUS` to change them. `PYTHON_BIN` can point to the desired Python executable.

```bash
DATASET_DIRS='/path/to/libero_spatial,/path/to/libero_object,/path/to/libero_goal,/path/to/libero_10' \
  bash scripts/libero/train_100k.sh
```

Check `one_step_fm_based_vla-train --help` for available runtime arguments. Both `one_step_fm_based_vla-train` and the training launcher use `turbovla.training.train_mixed`. The launcher fixes the architecture and optimizer recipe used for the published run. Set `--head_lr` and `--dinov3_lr` to change the two learning rates.

## Evaluate

The evaluation entry point runs all four suites sequentially on one GPU, with
32 episode shards per suite. The protocol is fixed: 50 initial states per task,
BF16, seed 42, 12 predicted actions, and 10 executed actions before replanning.
DINOv3 and R3M update at every policy query. Mamba runs on CUDA using the
matched BF16 scan.

```bash
one_step_fm_based_vla-eval --ckpt one_step_fm_based_vla_95k.pth --gpu 2 \
  --output-dir outputs/evaluation/95k_32slice_gpu2
```

Each suite has 500 episodes; the full run has 2,000. The evaluator reads the
model architecture from the checkpoint and uses local pretrained resources.
Use `one_step_fm_based_vla-eval --help` for path options and optional video output.
Completed shards are resumed after validating their protocol and coverage;
changing weights, resources, code, package versions, or GPU requires a new output
directory. EGL rendering uses the packaged NVIDIA vendor configuration.

## License

Apache-2.0. See [LICENSE](LICENSE).
