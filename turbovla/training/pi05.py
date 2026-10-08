#!/usr/bin/env python3
"""Original one_step_fm_based_vla AdamW recipe and checkpointed EMA shadows."""


import torch.optim as torch_optim

from . import trainer


EMA_DECAY = 0.999
_ACTIVE_OPTIMIZER = None
_ORIGINAL_BUILD_OPTIMIZER = trainer.build_param_group_optimizer
_ORIGINAL_TORCH_SAVE = trainer.torch.save


class Pi05AdamW(torch_optim.AdamW):
    def __init__(self, params, *args, **kwargs):
        kwargs.setdefault("betas", (0.9, 0.95))
        kwargs.setdefault("eps", 1e-8)
        super().__init__(params, *args, **kwargs)
        self.ema_decay = EMA_DECAY
        self._ema_params = {}
        self._ema_param_names = {}
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and param.is_floating_point():
                    self._ema_params[param] = param.detach().clone()

    def step(self, closure=None):
        result = super().step(closure=closure)
        decay = self.ema_decay
        with trainer.torch.no_grad():
            for param, ema_param in self._ema_params.items():
                ema_param.mul_(decay).add_(param.detach(), alpha=1.0 - decay)
        return result

    def load_ema_model_state_dict(self, state_dict):
        """Restore EMA shadows when continuing a saved training run."""
        if not isinstance(state_dict, dict):
            raise TypeError("ema_model_state_dict must be a mapping")
        missing = []
        with trainer.torch.no_grad():
            for param, name in self._ema_param_names.items():
                tensor = state_dict.get(name)
                if tensor is None:
                    missing.append(name)
                    continue
                shadow = self._ema_params[param]
                if tensor.shape != shadow.shape:
                    raise ValueError(
                        f"EMA shape mismatch for {name}: checkpoint={tuple(tensor.shape)}, "
                        f"optimizer={tuple(shadow.shape)}"
                    )
                shadow.copy_(tensor.to(device=shadow.device, dtype=shadow.dtype))
        if missing:
            raise KeyError(f"EMA checkpoint is missing {len(missing)} trainable tensors: {missing[:10]}")


def build_param_group_optimizer_with_pi05_ema(model, args):
    global _ACTIVE_OPTIMIZER
    optimizer, optimizer_summary = _ORIGINAL_BUILD_OPTIMIZER(model, args)
    param_names = {}
    for name, param in model.named_parameters():
        clean_name = name[7:] if name.startswith("module.") else name
        if param in optimizer._ema_params:
            param_names[param] = clean_name
    optimizer._ema_param_names = param_names
    _ACTIVE_OPTIMIZER = optimizer
    return optimizer, optimizer_summary


def torch_save_with_pi05_ema(obj, *args, **kwargs):
    optimizer = _ACTIVE_OPTIMIZER
    if isinstance(obj, dict) and optimizer is not None and "model_state_dict" in obj:
        model_state = obj["model_state_dict"]
        if isinstance(model_state, dict):
            ema_state = dict(model_state)
            for param, name in optimizer._ema_param_names.items():
                if name in ema_state:
                    ema_tensor = optimizer._ema_params[param].detach()
                    ema_state[name] = ema_tensor.to(device="cpu", dtype=ema_state[name].dtype)
            obj = dict(obj)
            obj["ema_decay"] = EMA_DECAY
            obj["ema_model_state_dict"] = ema_state
    return _ORIGINAL_TORCH_SAVE(obj, *args, **kwargs)


trainer.AdamW = Pi05AdamW
trainer.build_param_group_optimizer = build_param_group_optimizer_with_pi05_ema
trainer.torch.save = torch_save_with_pi05_ema
