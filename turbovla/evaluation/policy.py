"""TurboVLA policy adapter for aligned LIBERO evaluation.

This module keeps the GroundingDINO evaluation protocol local to the model:
256px DINOv3 preprocessing, GroundingDINO state normalization, hard min/max
action denormalization, and the original gripper sign rule.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import deque
from typing import Any, Sequence

import numpy as np
from PIL import Image
import torch

from ..models.flow_checkpoint import (
    assert_flow_checkpoint_compatible,
    assert_flow_parameters_loaded,
    infer_flow_checkpoint_architecture,
)


EXPECTED_IMAGE_SIZE = 256
DINO_PATCH_SIZE = 16
ACTION_DIM = 7
STATE_DIM = 8


ACTION_MIN = np.asarray(
    [
        -0.9375,
        -0.9375,
        -0.9375,
        -0.23642857372760773,
        -0.3053571283817291,
        -0.3675000071525574,
        -1.0,
    ],
    dtype=np.float32,
)

ACTION_MAX = np.asarray(
    [
        0.9375,
        0.9375,
        0.9375,
        0.30000001192092896,
        0.29357144236564636,
        0.375,
        1.0,
    ],
    dtype=np.float32,
)

PROPRIO_MEAN = np.asarray(
    [
        -0.04190646484494209,
        0.03539437800645828,
        0.8257066607475281,
        2.908315658569336,
        -0.5562158823013306,
        -0.16649103164672852,
        0.02831534668803215,
        -0.028561558574438095,
    ],
    dtype=np.float32,
)

PROPRIO_STD = np.asarray(
    [
        0.10743443667888641,
        0.14424759149551392,
        0.25723373889923096,
        0.34413808584213257,
        1.234430193901062,
        0.35798805952072144,
        0.013308786787092686,
        0.013174591585993767,
    ],
    dtype=np.float32,
)


def configure_transformers_offline() -> None:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("USE_TORCH", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed_everywhere(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_preprocessor_config(local_model_path: str) -> dict[str, Any]:
    cfg_path = os.path.join(local_model_path, "preprocessor_config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_manual_rgb_normalizer(
    image_mean: Sequence[float],
    image_std: Sequence[float],
    rescale_factor: float,
    expected_size: int,
    patch_size: int,
    backbone_name: str,
):
    mean = torch.tensor(image_mean, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(image_std, dtype=torch.float32).view(3, 1, 1)
    rescale_factor = float(rescale_factor)

    def process_one(img: Image.Image | np.ndarray) -> torch.Tensor:
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")

        width, height = img.size
        if height != expected_size or width != expected_size:
            raise ValueError(
                f"{backbone_name} expects pre-rotated {expected_size}x{expected_size} RGB input, "
                f"but got {height}x{width}. Do not apply StarVLA/OpenVLA 224px resize here."
            )
        if height % patch_size != 0 or width % patch_size != 0:
            raise ValueError(
                f"{backbone_name} input size {(height, width)} must be divisible by patch size {patch_size}."
            )

        arr = np.asarray(img, dtype=np.float32) * rescale_factor
        x = torch.from_numpy(arr).permute(2, 0, 1)
        return (x - mean) / std

    def processor(images: Image.Image | np.ndarray | Sequence[Image.Image | np.ndarray]) -> dict[str, torch.Tensor]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        pixel_values = torch.stack([process_one(im) for im in images], dim=0)
        return {"pixel_values": pixel_values}

    return processor


def build_dinov3_manual_processor(local_dinov3_path: str):
    cfg = load_preprocessor_config(local_dinov3_path)
    return _build_manual_rgb_normalizer(
        image_mean=cfg.get("image_mean", [0.485, 0.456, 0.406]),
        image_std=cfg.get("image_std", [0.229, 0.224, 0.225]),
        rescale_factor=cfg.get("rescale_factor", 1.0 / 255.0),
        expected_size=EXPECTED_IMAGE_SIZE,
        patch_size=DINO_PATCH_SIZE,
        backbone_name="DINOv3",
    )


def build_r3m_manual_processor():
    """Return unnormalized uint8 224px center crops for the R3M encoder."""

    def process_one(image: Image.Image | np.ndarray) -> torch.Tensor:
        array = np.asarray(image)
        if array.shape != (EXPECTED_IMAGE_SIZE, EXPECTED_IMAGE_SIZE, 3):
            raise ValueError(
                f"R3M input must be {EXPECTED_IMAGE_SIZE}x{EXPECTED_IMAGE_SIZE} RGB, "
                f"got {array.shape}"
            )
        start = (EXPECTED_IMAGE_SIZE - 224) // 2
        crop = np.ascontiguousarray(array[start : start + 224, start : start + 224])
        return torch.from_numpy(crop).permute(2, 0, 1)

    def processor(
        images: Image.Image | np.ndarray | Sequence[Image.Image | np.ndarray],
    ) -> dict[str, torch.Tensor]:
        if not isinstance(images, (list, tuple)):
            images = [images]
        return {"pixel_values": torch.stack([process_one(image) for image in images])}

    return processor


def rotate_libero_image(image: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(image)[::-1, ::-1])


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float32).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = math.sqrt(max(0.0, 1.0 - float(quat[3]) * float(quat[3])))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(float(quat[3])) / den).astype(np.float32)


def state_from_libero_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.concatenate(
        [
            np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1),
            quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32).reshape(-1),
        ],
        axis=0,
    ).astype(np.float32)


def normalize_state(state_np: np.ndarray) -> torch.Tensor:
    state = np.asarray(state_np, dtype=np.float32).reshape(-1)
    if state.shape[0] != STATE_DIM:
        raise ValueError(f"GroundingDINO state must have {STATE_DIM} dims, got {state.shape}")
    return torch.from_numpy((state - PROPRIO_MEAN) / (PROPRIO_STD + 1e-6)).float()


def normalized_action_to_env_action(
    action_norm_np: np.ndarray,
    gripper_deadband: float = 0.0,
) -> np.ndarray:
    return normalized_action_chunk_to_env_actions(
        np.asarray(action_norm_np, dtype=np.float32).reshape(1, -1),
        gripper_deadband=gripper_deadband,
    )[0]


def normalized_action_chunk_to_env_actions(
    action_norm_np: np.ndarray,
    gripper_deadband: float = 0.0,
    action_min: np.ndarray = ACTION_MIN,
    action_max: np.ndarray = ACTION_MAX,
) -> np.ndarray:
    """Vectorized normalized-to-environment conversion for an action chunk."""
    actions = np.asarray(action_norm_np, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions[None, :]
    if actions.ndim != 2 or actions.shape[1] < ACTION_DIM:
        raise ValueError(f"action chunk must have shape (N, D>={ACTION_DIM}), got {actions.shape}")

    minimum = np.asarray(action_min, dtype=np.float32).reshape(-1)
    maximum = np.asarray(action_max, dtype=np.float32).reshape(-1)
    if minimum.size < 6 or maximum.size < 6:
        raise ValueError("action_min and action_max must contain at least six arm dimensions")

    arm = 0.5 * (actions[:, :6] + 1.0) * (maximum[:6] - minimum[:6]) + minimum[:6]
    gripper_source = actions[:, 6]
    gripper = np.where(
        gripper_source > gripper_deadband,
        1.0,
        np.where(gripper_source < -gripper_deadband, -1.0, 1.0),
    ).astype(np.float32, copy=False)
    return np.concatenate([arm, gripper[:, None]], axis=1).astype(np.float32, copy=False)


def sanitize_pred_chunk(pred_chunk: np.ndarray) -> np.ndarray:
    pred_chunk = np.asarray(pred_chunk, dtype=np.float32)
    if pred_chunk.ndim == 1:
        pred_chunk = pred_chunk[None, :]
    elif pred_chunk.ndim > 2:
        pred_chunk = pred_chunk.reshape(pred_chunk.shape[0], -1)

    valid_actions = []
    for row in pred_chunk:
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.size < ACTION_DIM:
            continue
        if row.size > ACTION_DIM:
            row = row[:ACTION_DIM]
        row = np.nan_to_num(row, nan=0.0, posinf=1.0, neginf=-1.0)
        valid_actions.append(row.astype(np.float32))

    if not valid_actions:
        return np.zeros((1, ACTION_DIM), dtype=np.float32)
    return np.stack(valid_actions, axis=0)


def get_libero_dummy_action() -> list[float]:
    return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]


def _checkpoint_state_dict(checkpoint: Any) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "model", "state_dict"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    return cleaned


class TurboVLAPolicy:
    def __init__(
        self,
        ckpt_path: str,
        dinov3_path: str,
        bert_path: str,
        r3m_path: str = "",
        device: str | torch.device | None = None,
        verbose: bool = True,
    ) -> None:
        from ..models.configuration import TurboVLAConfig
        from ..models.turbovla import TurboVLA

        configure_transformers_offline()
        self.ckpt_path = str(ckpt_path)
        self.dinov3_path = str(dinov3_path)
        self.bert_path = str(bert_path)
        self.r3m_path = str(r3m_path)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.precision = "bf16"
        self.model_dtype = torch.bfloat16
        self._dinov3_tokens: torch.Tensor | None = None
        self.verbose = bool(verbose)
        for name, path in (("DINOv3", self.dinov3_path), ("BERT", self.bert_path)):
            if not path:
                raise ValueError(f"{name} model path is required")
            if not os.path.isdir(path):
                raise FileNotFoundError(f"local {name} directory not found: {path}")

        self._checkpoint = torch.load(self.ckpt_path, map_location="cpu")
        model_config = self._checkpoint.get("model_config") if isinstance(self._checkpoint, dict) else None
        if model_config is None:
            raise ValueError("one_step_fm_based_vla evaluation requires a checkpoint containing model_config")
        config = TurboVLAConfig.from_mapping(model_config)
        config.text.model_name_or_path = self.bert_path
        config.text.local_files_only = True
        config.vision.model_name_or_path = self.dinov3_path
        config.vision.local_files_only = True
        config.vision.compute_precision = "bf16"
        if config.r3m.enabled:
            configured_r3m_path = self.r3m_path or config.r3m.checkpoint_path
            configured_r3m_path = os.path.abspath(os.path.expanduser(configured_r3m_path))
            if not os.path.isfile(configured_r3m_path):
                raise FileNotFoundError(f"R3M checkpoint not found: {configured_r3m_path}")
            config.r3m.checkpoint_path = configured_r3m_path
            self.r3m_path = configured_r3m_path
        self.chunk_size = int(config.action.horizon)
        self.action_dim = int(config.action.action_dim)
        defaults = {
            "action_head": "flow_matching",
            "flow_num_heads": 4,
            "flow_condition_layers": 4,
            "flow_dit_layers": 16,
            "flow_target_tokens": 16,
            "flow_static_tokens": 8,
            "flow_state_dim": 8,
            "flow_bijection_blocks": 6,
            "flow_state_encoding": "state_tokens_zero_pad",
        }
        ckpt_flow_args = {name: self._checkpoint.get(name, default) for name, default in defaults.items()}
        inferred_flow = infer_flow_checkpoint_architecture(self._checkpoint, state_dim=int(config.action.state_dim))
        if inferred_flow is not None:
            (ckpt_flow_args["flow_state_dim"], ckpt_flow_args["flow_bijection_blocks"],
             ckpt_flow_args["flow_state_encoding"]) = inferred_flow
        self.model = TurboVLA(config, **ckpt_flow_args)
        self.history_length = int(config.history.length) if config.history.enabled else 0
        self.history_r3m = bool(config.history.r3m_enabled)
        self.r3m_enabled = bool(config.r3m.enabled)
        self._state_history: deque[np.ndarray] = deque(maxlen=self.history_length or 1)
        self._image_history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.history_length or 1)
        if self.verbose:
            print("[TurboVLAPolicy] model source: turbovla.models.turbovla", flush=True)
        self._load_checkpoint()
        self._set_eval_precision()
        self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        self._verify_model_precision()
        self.dinov3_processor = build_dinov3_manual_processor(self.dinov3_path)
        self.r3m_processor = build_r3m_manual_processor() if self.r3m_enabled else None

    def _set_eval_precision(self) -> None:
        self.model.to(dtype=torch.bfloat16)

    def _verify_model_precision(self) -> None:
        floating_dtypes = {
            param.dtype
            for param in self.model.parameters()
            if param.is_floating_point()
        }
        expected = {self.model_dtype}
        if floating_dtypes != expected:
            raise RuntimeError(
                f"precision={self.precision} expected model parameter dtypes {expected}, "
                f"got {floating_dtypes}"
            )
        if self.verbose:
            dtype_name = str(self.model_dtype).removeprefix("torch.")
            print(
                f"[TurboVLAPolicy] precision={self.precision}, model_parameter_dtype={dtype_name}",
                flush=True,
            )

    def _load_checkpoint(self) -> None:
        checkpoint = self._checkpoint
        flow_architecture = assert_flow_checkpoint_compatible(checkpoint, self.model)
        source_state = _strip_module_prefix(_checkpoint_state_dict(checkpoint))
        target_state = self.model.state_dict()

        loadable = {}
        skipped_shape = []
        skipped_missing = []
        for key, tensor in source_state.items():
            if key not in target_state:
                skipped_missing.append(key)
                continue
            if target_state[key].shape != tensor.shape:
                skipped_shape.append((key, tuple(tensor.shape), tuple(target_state[key].shape)))
                continue
            loadable[key] = tensor

        missing, unexpected = self.model.load_state_dict(loadable, strict=False)
        if flow_architecture is not None:
            assert_flow_parameters_loaded(self.model, list(missing), skipped_missing)
        if self.verbose:
            print(f"[TurboVLAPolicy] loaded ckpt: {self.ckpt_path}", flush=True)
            print(
                "[TurboVLAPolicy] "
                f"loadable={len(loadable)}, missing_after_load={len(missing)}, "
                f"unexpected_after_load={len(unexpected)}, skipped_missing={len(skipped_missing)}, "
                f"skipped_shape={len(skipped_shape)}",
                flush=True,
            )
            if missing:
                print(f"  first missing keys: {list(missing)[:20]}", flush=True)
            if unexpected:
                print(f"  first unexpected keys: {list(unexpected)[:20]}", flush=True)
            if skipped_shape:
                print(f"  first shape mismatches: {skipped_shape[:10]}", flush=True)

    def _build_batch(
        self,
        primary_images: Sequence[np.ndarray],
        wrist_images: Sequence[np.ndarray],
        states: Sequence[np.ndarray],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        flat_images: list[np.ndarray] = []
        for primary, wrist in zip(primary_images, wrist_images):
            flat_images.extend([primary, wrist])

        batch_size = len(primary_images)
        samples = {}
        dinov3_pixel_values = self.dinov3_processor(flat_images)["pixel_values"]
        samples["dinov3"] = dinov3_pixel_values.view(
            batch_size, 2, *dinov3_pixel_values.shape[1:]
        ).to(self.device)
        if self.r3m_processor is not None:
            r3m_pixel_values = self.r3m_processor(flat_images)["pixel_values"]
            samples["r3m"] = r3m_pixel_values.view(
                batch_size, 2, *r3m_pixel_values.shape[1:]
            ).to(self.device)
        state_tensors = torch.stack([normalize_state(state) for state in states], dim=0).to(self.device)
        return samples, state_tensors

    def _prepare_model_inputs(
        self,
        samples: dict[str, torch.Tensor],
        states: torch.Tensor,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        samples = {
            key: value.to(dtype=self.model_dtype) if value.is_floating_point() else value
            for key, value in samples.items()
        }
        return samples, states.to(dtype=self.model_dtype)

    def reset_history(self) -> None:
        """Clear synchronized robot-state and visual history."""
        self._state_history.clear()
        self._image_history.clear()
        self._dinov3_tokens = None


    @property
    def history_size(self) -> int:
        return len(self._state_history)

    def record_history_state(
        self,
        state_or_obs: np.ndarray | dict[str, Any],
    ) -> None:
        """Record one observed state and, when enabled, its two camera frames."""
        if not self.history_length:
            return
        state = (
            state_from_libero_obs(state_or_obs)
            if isinstance(state_or_obs, dict)
            else np.asarray(state_or_obs, dtype=np.float32).reshape(-1)
        )
        if state.shape != (STATE_DIM,):
            raise ValueError(f"history state must have shape ({STATE_DIM},), got {state.shape}")
        if not np.isfinite(state).all():
            raise ValueError("history state must be finite")
        self._state_history.append(state.copy())
        if (
            getattr(self, "history_r3m", False)
        ):
            if not isinstance(state_or_obs, dict):
                raise ValueError("visual history requires a LIBERO observation mapping")
            primary = rotate_libero_image(state_or_obs["agentview_image"])
            wrist = rotate_libero_image(state_or_obs["robot0_eye_in_hand_image"])
            self._image_history.append((primary, wrist))

    def _history_tensors(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[None, None]:
        if not self.history_length:
            return None, None

        states = np.zeros((self.history_length, STATE_DIM), dtype=np.float32)
        mask = np.zeros(self.history_length, dtype=np.bool_)
        count = len(self._state_history)
        if count:
            raw_states = np.stack(self._state_history)
            proprio_mean = np.asarray(getattr(self, "proprio_mean", PROPRIO_MEAN), dtype=np.float32)
            proprio_std = np.asarray(getattr(self, "proprio_std", PROPRIO_STD), dtype=np.float32)
            states[-count:] = (raw_states - proprio_mean) / (proprio_std + 1e-6)
            mask[-count:] = True

        states_tensor = torch.from_numpy(states).unsqueeze(0).to(
            device=self.device, dtype=self.model_dtype
        )
        mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(device=self.device)
        return states_tensor, mask_tensor


    def _history_r3m_tensor(self) -> torch.Tensor | None:
        if not self.history_r3m:
            return None
        if self.r3m_processor is None:
            raise RuntimeError("R3M history requires the R3M image processor")
        if len(self._image_history) != len(self._state_history):
            raise RuntimeError("R3M image history is not synchronized with state history")
        output = torch.zeros(
            1,
            self.history_length,
            2,
            3,
            224,
            224,
            dtype=torch.uint8,
        )
        count = len(self._image_history)
        if count:
            flat_images = []
            for primary, wrist in self._image_history:
                flat_images.extend([primary, wrist])
            pixels = self.r3m_processor(flat_images)["pixel_values"]
            output[0, -count:] = pixels.view(count, 2, 3, 224, 224)
        return output.to(device=self.device)

    def predict_normalized_action_chunk(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
    ) -> np.ndarray:
        if isinstance(state_or_obs, dict):
            state = state_from_libero_obs(state_or_obs)
        else:
            state = np.asarray(state_or_obs, dtype=np.float32)

        samples, states = self._build_batch(
            [primary_image],
            [wrist_image],
            [state],
        )
        history_states, history_mask = self._history_tensors()
        history_r3m = self._history_r3m_tensor()
        if history_r3m is not None:
            samples["r3m_history"] = history_r3m
        samples, states = self._prepare_model_inputs(samples, states)
        with torch.inference_mode():
            # Keep the verified operation order; refresh both views every query.
            self._dinov3_tokens = self.model.encode_vision(samples.pop("dinov3")).detach()
            samples["dinov3_tokens"] = self._dinov3_tokens
            pred = self.model(
                [instruction],
                samples,
                states,
                history_states=history_states,
                history_mask=history_mask,
            )
        if pred.dtype != self.model_dtype:
            raise RuntimeError(
                f"precision={self.precision} expected forward output dtype {self.model_dtype}, got {pred.dtype}"
            )
        return sanitize_pred_chunk(pred.detach().float().cpu().numpy()[0])

    def predict_env_action_chunk(
        self,
        primary_image: np.ndarray,
        wrist_image: np.ndarray,
        instruction: str,
        state_or_obs: np.ndarray | dict[str, Any],
        execute_steps: int | None = None,
    ) -> np.ndarray:
        pred_norm = self.predict_normalized_action_chunk(primary_image, wrist_image, instruction, state_or_obs)
        if execute_steps is not None:
            pred_norm = pred_norm[: int(execute_steps)]
        return normalized_action_chunk_to_env_actions(pred_norm)

    def normalized_action_chunk_to_env_actions(self, chunk: np.ndarray) -> np.ndarray:
        return normalized_action_chunk_to_env_actions(chunk)

    def predict_env_action_chunk_from_obs(
        self,
        obs: dict[str, Any],
        instruction: str,
        execute_steps: int | None = None,
    ) -> np.ndarray:
        primary = rotate_libero_image(obs["agentview_image"])
        wrist = rotate_libero_image(obs["robot0_eye_in_hand_image"])
        return self.predict_env_action_chunk(primary, wrist, instruction, obs, execute_steps=execute_steps)
