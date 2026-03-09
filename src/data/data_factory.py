from typing import Tuple
from torch.utils.data import DataLoader
from src.utils.config import DataConfig
from src.data.bar_dataloader import get_bar_dataloaders
from src.data.celeba_dataloader import get_celeba_dataloaders
from src.data.nico_pp_dataloader import get_nico_pp_dataloaders


NUM_CLASSES = {
    "bar": 6,
    "celeba": 2,
    "nico_pp": 60,
}


def get_dataloaders(
    data_cfg: DataConfig, seed: int
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader, int]:
    dataset = data_cfg.dataset.lower()

    if dataset not in NUM_CLASSES:
        supported = sorted(NUM_CLASSES.keys())
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported datasets: {supported}"
        )

    common_kwargs = {
        "data_root": data_cfg.data_root,
        "batch_size": data_cfg.batch_size,
        "num_val_samples_per_class": data_cfg.num_val_samples_per_class,
        "rndm_seed": seed,
        "num_workers": data_cfg.num_workers,
        "image_size": data_cfg.image_size,
    }

    if dataset == "bar":
        loaders = get_bar_dataloaders(**common_kwargs)
    elif dataset == "celeba":
        loaders = get_celeba_dataloaders(
            target_attr=data_cfg.target_attr, **common_kwargs
        )
    elif dataset == "nico_pp":
        loaders = get_nico_pp_dataloaders(**common_kwargs)

    num_classes = NUM_CLASSES[dataset]
    return (*loaders, num_classes)
