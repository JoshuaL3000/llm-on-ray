
from typing import Dict, Any

finetune_config: Dict[str, Any] = {
        "General": {"config": {}},
        "Dataset": {"validation_file": None, "validation_split_percentage": 0},
        "Training": {
            "optimizer": "AdamW",
            "lr_scheduler": "linear",
            "weight_decay": 0.0,
            "device": "CPU",
            "num_training_workers": 2,
            "resources_per_worker": {"CPU": 24},
            "accelerate_mode": "CPU_DDP",
        },
        "failure_config": {"max_failures": 5},
    }

ray_init_config: Dict[str, Any] = {
    "runtime_env": {
        "env_vars": {
            "OMP_NUM_THREADS": "24",
            "ACCELERATE_USE_CPU": "True",
            "ACCELERATE_MIXED_PRECISION": "no",
            "CCL_WORKER_COUNT": "1",
            "CCL_LOG_LEVEL": "info",
            "WORLD_SIZE": "2",
        }
    },
    "address": "auto",
}