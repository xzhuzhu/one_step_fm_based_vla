# VLA-Adapter rollout compatibility

This directory contains only the LIBERO task and episode rollout protocol
adapted for TurboVLA evaluation. It does not vendor the complete VLA-Adapter
repository. The retained MIT license is in
`third_party/licenses/VLA-Adapter.txt`.

`vla_adapter.rollout` is the internal episode worker for `finalvla-eval`.
Its protocol is fixed by `turbovla.evaluation.protocol`.
