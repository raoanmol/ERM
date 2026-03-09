import yaml
from typing import Optional
from dataclasses import dataclass


@dataclass
class DataConfig:
    dataset: str
    data_root: str
    batch_size: int
    num_val_samples_per_class: int
    num_workers: int
    image_size: int
    target_attr: str = "Male"


@dataclass
class ModelConfig:
    name: str
    pretrained: bool = True


@dataclass
class TrainingConfig:
    epochs: int = 10
    lr: float = 1e-4
    weight_decay: float = 1e-2


@dataclass
class LoggingConfig:
    log_dir: str = "logs"
    use_wandb: bool = False
    wandb_project: str = "erm-benchmark"
    wandb_entity: Optional[str] = None


@dataclass
class Config:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig
    seed: int = 42
    device: str = "auto"


def load_config(yaml_path: str) -> Config:
    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    data = DataConfig(**raw["data"])
    model = ModelConfig(**raw["model"])
    training = TrainingConfig(**raw.get("training", {}))
    logging = LoggingConfig(**raw.get("logging", {}))

    return Config(
        data=data,
        model=model,
        training=training,
        logging=logging,
        seed=raw.get("seed", 42),
        device=raw.get("device", "auto"),
    )
