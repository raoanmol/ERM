import json
import torch
import random
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Tuple, List, Dict
from torch.utils.data import Dataset, DataLoader


class NICOPPDataset(Dataset):
    def __init__(self, image_paths: List[Path], labels: List[int], transform=None):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def collect_all_images(data_root: Path) -> Tuple[List[Path], List[int], Dict[str, int]]:
    mapping_file = data_root / "track_1" / "dg_label_id_mapping.json"
    with open(mapping_file, "r") as f:
        class_to_idx = json.load(f)

    train_dir = data_root / "track_1" / "track_1" / "train"

    all_paths = []
    all_labels = []
    for context_dir in sorted(train_dir.iterdir()):
        if not context_dir.is_dir():
            continue
        for label_dir in sorted(context_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label_name = label_dir.name
            if label_name not in class_to_idx:
                continue
            label_id = class_to_idx[label_name]
            for img_path in sorted(label_dir.glob("*.jpg")):
                all_paths.append(img_path)
                all_labels.append(label_id)

    return all_paths, all_labels, class_to_idx


def create_validation_splits(
    train_paths: List[Path],
    train_labels: List[int],
    test_paths: List[Path],
    test_labels: List[int],
    num_samples_per_class: int,
    seed: int,
    class_to_idx: Dict[str, int],
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    random.seed(seed)
    num_classes = len(class_to_idx)

    # iid
    in_domain_val_paths = []
    in_domain_val_labels = []
    for cls in range(num_classes):
        class_indices = [i for i, l in enumerate(train_labels) if l == cls]
        if len(class_indices) < num_samples_per_class:
            print(
                f"Warning: class {cls} has only {len(class_indices)} images in training set, requested {num_samples_per_class}"
            )
            sampled_indices = class_indices
        else:
            sampled_indices = random.sample(class_indices, num_samples_per_class)
        for i in sampled_indices:
            in_domain_val_paths.append(train_paths[i])
            in_domain_val_labels.append(train_labels[i])

    # ood
    out_of_domain_val_paths = []
    out_of_domain_val_labels = []
    for cls in range(num_classes):
        class_indices = [i for i, l in enumerate(test_labels) if l == cls]
        if len(class_indices) == 0:
            print(f"Warning: class {cls} has no images in test set")
            continue
        sampled_indices = random.choices(class_indices, k=num_samples_per_class)
        for i in sampled_indices:
            out_of_domain_val_paths.append(test_paths[i])
            out_of_domain_val_labels.append(test_labels[i])

    return (
        in_domain_val_paths,
        in_domain_val_labels,
        out_of_domain_val_paths,
        out_of_domain_val_labels,
    )


def get_nico_pp_dataloaders(
    data_root: str,
    batch_size: int,
    num_val_samples_per_class: int,
    rndm_seed: int,
    num_workers: int,
    image_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    data_root = Path(data_root)

    all_paths, all_labels, class_to_idx = collect_all_images(data_root)

    data_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 75/25 train-test split
    combined = list(zip(all_paths, all_labels))
    random.seed(rndm_seed)
    random.shuffle(combined)
    split_idx = int(len(combined) * 0.75)
    train_pairs = combined[:split_idx]
    test_pairs = combined[split_idx:]

    train_paths = [p for p, _ in train_pairs]
    train_labels = [l for _, l in train_pairs]
    test_paths = [p for p, _ in test_pairs]
    test_labels = [l for _, l in test_pairs]

    (
        in_domain_val_paths,
        in_domain_val_labels,
        out_of_domain_val_paths,
        out_of_domain_val_labels,
    ) = create_validation_splits(
        train_paths=train_paths,
        train_labels=train_labels,
        test_paths=test_paths,
        test_labels=test_labels,
        num_samples_per_class=num_val_samples_per_class,
        seed=rndm_seed,
        class_to_idx=class_to_idx,
    )

    in_domain_val_paths_set = set(in_domain_val_paths)
    filtered_train_paths = []
    filtered_train_labels = []
    for p, l in zip(train_paths, train_labels):
        if p not in in_domain_val_paths_set:
            filtered_train_paths.append(p)
            filtered_train_labels.append(l)
    train_paths = filtered_train_paths
    train_labels = filtered_train_labels

    train_dataset = NICOPPDataset(train_paths, train_labels, transform=data_transform)
    in_domain_val_dataset = NICOPPDataset(
        in_domain_val_paths, in_domain_val_labels, transform=data_transform
    )
    out_of_domain_val_dataset = NICOPPDataset(
        out_of_domain_val_paths, out_of_domain_val_labels, transform=data_transform
    )
    test_dataset = NICOPPDataset(test_paths, test_labels, transform=data_transform)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    in_domain_val_loader = DataLoader(
        in_domain_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    out_of_domain_val_loader = DataLoader(
        out_of_domain_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    print("Dataset Statistics:")
    print(f"  Train: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(
        f"  In-Domain Validation: {len(in_domain_val_dataset)} images ({len(in_domain_val_loader)} batches)"
    )
    print(
        f"  Out-of-Domain Validation: {len(out_of_domain_val_dataset)} images ({len(out_of_domain_val_loader)} batches)"
    )
    print(f"  Test: {len(test_dataset)} images ({len(test_loader)} batches)")
    print(f"  Number of classes: {len(class_to_idx)}")

    return train_loader, in_domain_val_loader, out_of_domain_val_loader, test_loader
