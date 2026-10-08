"""The verified FinalVLA single-GPU evaluation protocol."""

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
SHARDS = 32
TRIALS_PER_TASK = 50
CHUNK_SIZE = 12
OPEN_LOOP_STEPS = 10
SETTLE_STEPS = 10
IMAGE_SIZE = 256
SEED = 42
PRECISION = "bf16"

PROTOCOL = {
    "num_trials_per_task": TRIALS_PER_TASK,
    "chunk_size": CHUNK_SIZE,
    "num_open_loop_steps": OPEN_LOOP_STEPS,
    "dinov3_update_interval": 1,
    "r3m_update_interval": 1,
    "query_interval": OPEN_LOOP_STEPS,
    "seed": SEED,
    "precision": PRECISION,
}
