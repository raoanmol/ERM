import shutil
import argparse
import torch
import torch.nn as nn
from dataclasses import asdict
from torch.utils.data import DataLoader, Subset

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.data.data_factory import get_dataloaders
from src.models.model_factory import create_model
from src.utils.logger import Logger
from src.training.trainer import Trainer


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def subset_loader(loader: DataLoader, num_samples: int) -> DataLoader:
    indices = list(range(min(num_samples, len(loader.dataset))))
    subset = Subset(loader.dataset, indices)
    return DataLoader(
        subset,
        batch_size=loader.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test-mode", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.seed is not None:
        config.seed = args.seed

    set_seed(config.seed)

    device = resolve_device(config.device)
    print(f"Using device: {device}")

    train_loader, id_val_loader, ood_val_loader, test_loader, num_classes = (
        get_dataloaders(config.data, seed=config.seed)
    )

    if args.test_mode:
        print("[TEST MODE] Running with subset data and 2 epochs")
        train_loader = subset_loader(train_loader, 20)
        id_val_loader = subset_loader(id_val_loader, 5)
        ood_val_loader = subset_loader(ood_val_loader, 5)
        test_loader = subset_loader(test_loader, 5)
        num_epochs = 2
    else:
        num_epochs = config.training.epochs

    model = create_model(config.model.name, num_classes, config.model.pretrained)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()

    run_name = f"{config.data.dataset}_{config.model.name}_seed{config.seed}"
    logger = Logger(config.logging, run_name, asdict(config))

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        id_val_loader=id_val_loader,
        ood_val_loader=ood_val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        logger=logger,
        checkpoint_dir=logger.log_dir,
        num_classes=num_classes,
        model_name=config.model.name,
    )

    trainer.train(num_epochs)
    trainer.test()
    logger.close()

    if args.test_mode:
        print("[TEST MODE] Run successful. Cleaning up test artifacts...")
        shutil.rmtree(logger.log_dir)
        print(f"[TEST MODE] Removed {logger.log_dir}")

    print("Done.")


if __name__ == "__main__":
    main()
