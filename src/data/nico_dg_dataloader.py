import torch
import random
from PIL import Image
from pathlib import Path
from typing import Tuple, List
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


EXCLUDED_CONTEXTS = {"dim"}
ALL_CONTEXTS = ["autumn", "dim", "grass", "outdoor", "rock", "water"]
NUM_CLASSES = 60


class NICODGDataset(Dataset):
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


def parse_annotations(
    data_root: Path,
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    annotation_dir = data_root / "DG_Benchmark" / "NICO_DG_Benchmark_annotation"
    image_dir = data_root / "DG_Benchmark" / "NICO_DG_Benchmark"

    contexts = [c for c in ALL_CONTEXTS if c not in EXCLUDED_CONTEXTS]

    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []

    for context in contexts:
        for split in ["train", "test"]:
            annotation_file = annotation_dir / f"{context}_{split}.txt"

            with open(annotation_file, "r") as f:
                for line in f:
                    parts = line.strip().rsplit(maxsplit=1)
                    if len(parts) != 2:
                        continue

                    rel_path = parts[0]
                    class_id = int(parts[1])

                    # Annotation paths start with "NICO_DG/"; strip and
                    # prepend the actual image directory.
                    rel_path = rel_path.replace("NICO_DG/", "", 1)
                    img_path = image_dir / rel_path

                    if split == "train":
                        train_paths.append(img_path)
                        train_labels.append(class_id)
                    else:
                        test_paths.append(img_path)
                        test_labels.append(class_id)

    return train_paths, train_labels, test_paths, test_labels


def create_validation_splits(
    train_paths: List[Path],
    train_labels: List[int],
    test_paths: List[Path],
    test_labels: List[int],
    num_samples_per_class: int,
    seed: int,
) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    random.seed(seed)

    # iid
    in_domain_val_paths = []
    in_domain_val_labels = []
    for cls in range(NUM_CLASSES):
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
    for cls in range(NUM_CLASSES):
        class_indices = [i for i, l in enumerate(test_labels) if l == cls]
        if len(class_indices) == 0:
            print(f"Warning: class {cls} has no images in test set")
            continue
        if len(class_indices) < num_samples_per_class:
            sampled_indices = random.choices(class_indices, k=num_samples_per_class)
        else:
            sampled_indices = random.sample(class_indices, num_samples_per_class)
        for i in sampled_indices:
            out_of_domain_val_paths.append(test_paths[i])
            out_of_domain_val_labels.append(test_labels[i])

    return (
        in_domain_val_paths,
        in_domain_val_labels,
        out_of_domain_val_paths,
        out_of_domain_val_labels,
    )


def get_nico_dg_dataloaders(
    data_root: str,
    batch_size: int,
    num_val_samples_per_class: int,
    rndm_seed: int,
    num_workers: int,
    image_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    data_root = Path(data_root)

    train_paths, train_labels, test_paths, test_labels = parse_annotations(data_root)

    data_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

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

    train_dataset = NICODGDataset(train_paths, train_labels, transform=data_transform)
    in_domain_val_dataset = NICODGDataset(
        in_domain_val_paths, in_domain_val_labels, transform=data_transform
    )
    out_of_domain_val_dataset = NICODGDataset(
        out_of_domain_val_paths, out_of_domain_val_labels, transform=data_transform
    )
    test_dataset = NICODGDataset(test_paths, test_labels, transform=data_transform)

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

    excluded = ", ".join(sorted(EXCLUDED_CONTEXTS))
    print(f"Dataset Statistics (NICO++ DG_Benchmark, excluded contexts: {excluded}):")
    print(f"  Train: {len(train_dataset)} images ({len(train_loader)} batches)")
    print(
        f"  In-Domain Validation: {len(in_domain_val_dataset)} images ({len(in_domain_val_loader)} batches)"
    )
    print(
        f"  Out-of-Domain Validation: {len(out_of_domain_val_dataset)} images ({len(out_of_domain_val_loader)} batches)"
    )
    print(f"  Test: {len(test_dataset)} images ({len(test_loader)} batches)")
    print(f"  Number of classes: {NUM_CLASSES}")

    return train_loader, in_domain_val_loader, out_of_domain_val_loader, test_loader
