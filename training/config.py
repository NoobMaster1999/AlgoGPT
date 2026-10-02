"""
Finetuning Configuration for Indian Derivatives
================================================
Centralized config for tokenizer and predictor finetuning on NIFTY/BANKNIFTY.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class FinetuneConfig:
    """Configuration for Kronos finetuning on Indian derivatives."""

    # ================================================================
    # Data Parameters
    # ================================================================
    symbols: List[str] = field(default_factory=lambda: ["NIFTY", "BANKNIFTY"])
    data_dir: str = "./data"
    data_source: str = "csv"  # "csv", "upstox", "nse", "yahoo"
    fetcher_kwargs: dict = field(default_factory=dict)

    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    interval: str = "5min"  # "1min", "5min", "15min", "1h", "1d"

    lookback_window: int = 90
    predict_window: int = 10
    max_context: int = 512
    clip: float = 5.0

    feature_list: List[str] = field(default_factory=lambda: ["open", "high", "low", "close", "volume", "amount"])
    time_feature_list: List[str] = field(default_factory=lambda: ["minute", "hour", "weekday", "day", "month"])

    train_ratio: float = 0.8

    # ================================================================
    # Tokenizer Architecture (match pretrained)
    # ================================================================
    tok_d_in: int = 6
    tok_d_model: int = 256
    tok_n_heads: int = 8
    tok_ff_dim: int = 1024
    tok_n_enc_layers: int = 3
    tok_n_dec_layers: int = 3
    tok_ffn_dropout_p: float = 0.1
    tok_attn_dropout_p: float = 0.1
    tok_resid_dropout_p: float = 0.1
    tok_s1_bits: int = 8
    tok_s2_bits: int = 8
    tok_beta: float = 0.25
    tok_gamma0: float = 1.0
    tok_gamma: float = 1.0
    tok_zeta: float = 1.0
    tok_group_size: int = 8

    # ================================================================
    # Predictor Architecture (match pretrained)
    # ================================================================
    pred_s1_bits: int = 8
    pred_s2_bits: int = 8
    pred_n_layers: int = 12
    pred_d_model: int = 512
    pred_n_heads: int = 8
    pred_ff_dim: int = 2048
    pred_ffn_dropout_p: float = 0.1
    pred_attn_dropout_p: float = 0.1
    pred_resid_dropout_p: float = 0.1
    pred_token_dropout_p: float = 0.1
    pred_learn_te: bool = True

    # ================================================================
    # Training Hyperparameters
    # ================================================================
    seed: int = 42

    # Tokenizer training
    tokenizer_learning_rate: float = 2e-4
    tokenizer_epochs: int = 10
    tokenizer_batch_size: int = 64  # per GPU
    tokenizer_accumulation_steps: int = 1
    tokenizer_log_interval: int = 50

    # Predictor training
    predictor_learning_rate: float = 4e-5
    predictor_epochs: int = 30
    predictor_batch_size: int = 50  # per GPU
    predictor_accumulation_steps: int = 1
    predictor_log_interval: int = 100

    # Optimizer
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_weight_decay: float = 0.1
    max_grad_norm: float = 3.0

    # Scheduler
    lr_pct_start: float = 0.03
    lr_div_factor: float = 10.0

    # ================================================================
    # Experiment Tracking & Saving
    # ================================================================
    use_comet: bool = False
    comet_api_key: Optional[str] = None
    comet_project_name: str = "Kronos-Indian-Derivatives"
    comet_workspace: Optional[str] = None
    comet_tag: str = "indian-derivatives-finetune"
    comet_name: str = "finetune-run"

    save_path: str = "./outputs/models"
    tokenizer_save_folder: str = "finetune_tokenizer_indian"
    predictor_save_folder: str = "finetune_predictor_indian"

    # Pretrained model paths (HuggingFace Hub or local)
    pretrained_tokenizer_path: str = "NeoQuasar/Kronos-Tokenizer-base"
    pretrained_predictor_path: str = "NeoQuasar/Kronos-small"

    # ================================================================
    # Distributed Training
    # ================================================================
    num_workers: int = 2
    pin_memory: bool = True

    def __post_init__(self):
        # Derived paths
        self.tokenizer_save_path = os.path.join(self.save_path, self.tokenizer_save_folder)
        self.predictor_save_path = os.path.join(self.save_path, self.predictor_save_folder)

        self.finetuned_tokenizer_path = os.path.join(self.tokenizer_save_path, "checkpoints", "best_model")
        self.finetuned_predictor_path = os.path.join(self.predictor_save_path, "checkpoints", "best_model")

        # Ensure directories exist
        Path(self.tokenizer_save_path).mkdir(parents=True, exist_ok=True)
        Path(self.predictor_save_path).mkdir(parents=True, exist_ok=True)
        Path("./outputs/logs").mkdir(parents=True, exist_ok=True)

    # Compatibility with old dict-based config
    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __contains__(self, key):
        return hasattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def __dict__(self):
        return {k: getattr(self, k) for k in dir(self) if not k.startswith("_")}


# Default config instance
DEFAULT_CONFIG = FinetuneConfig()


def load_config_from_env() -> FinetuneConfig:
    """Load config with environment variable overrides."""
    config = FinetuneConfig()

    # Override from env vars
    env_mappings = {
        "SYMBOLS": ("symbols", lambda x: x.split(",")),
        "DATA_DIR": ("data_dir", str),
        "DATA_SOURCE": ("data_source", str),
        "START_DATE": ("start_date", str),
        "END_DATE": ("end_date", str),
        "INTERVAL": ("interval", str),
        "LOOKBACK_WINDOW": ("lookback_window", int),
        "PREDICT_WINDOW": ("predict_window", int),
        "MAX_CONTEXT": ("max_context", int),
        "SEED": ("seed", int),
        "TOKENIZER_LR": ("tokenizer_learning_rate", float),
        "PREDICTOR_LR": ("predictor_learning_rate", float),
        "TOKENIZER_EPOCHS": ("tokenizer_epochs", int),
        "PREDICTOR_EPOCHS": ("predictor_epochs", int),
        "BATCH_SIZE": ("predictor_batch_size", int),
        "USE_COMET": ("use_comet", lambda x: x.lower() == "true"),
        "COMET_API_KEY": ("comet_api_key", str),
        "COMET_WORKSPACE": ("comet_workspace", str),
        "SAVE_PATH": ("save_path", str),
        "PRETRAINED_TOKENIZER": ("pretrained_tokenizer_path", str),
        "PRETRAINED_PREDICTOR": ("pretrained_predictor_path", str),
    }

    for env_key, (attr, converter) in env_mappings.items():
        env_val = os.environ.get(env_key)
        if env_val is not None:
            try:
                setattr(config, attr, converter(env_val))
            except Exception as e:
                print(f"Warning: Failed to parse {env_key}={env_val}: {e}")

    return config