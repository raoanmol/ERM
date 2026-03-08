import torch
import random
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Tuple, List, Dict
from torch.utils.data import Dataset, DataLoader


class CelebADataset(Dataset):
    CLASSES = [0, 1]

    def __init__(self, image_paths: List[Path], labels: List[int], transform = None):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def parse_celeba_metadata(data_root: Path, target_attr: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    partition_file = data_root / 'list_eval_partition.txt'
    attr_file = data_root / 'list_attr_celeba.txt'

    partitions = {}
    with open(partition_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                partitions[parts[0]] = int(parts[1])

    attrs = {}
    with open(attr_file, 'r') as f:
        lines = f.readlines()

    header = lines[1].strip().split()
    if target_attr not in header:
        raise ValueError(f"Target attribute '{target_attr}' not found. Available attributes: {header}")
    attr_idx = header.index(target_attr)

    for line in lines[2:]:
        parts = line.strip().split()
        filename = parts[0]
        val = int(parts[1 + attr_idx])
        attrs[filename] = 0 if val == -1 else 1

    return partitions, attrs


def create_validation_splits(train_paths: List[Path], train_labels: List[int], test_paths: List[Path], test_labels: List[int], num_samples_per_class: int, seed: int) -> Tuple[List[Path], List[int], List[Path], List[int]]:
    random.seed(seed)

    # iid
    in_domain_val_paths = []
    in_domain_val_labels = []
    for cls in CelebADataset.CLASSES:
        class_indices = [i for i, l in enumerate(train_labels) if l == cls]
        if len(class_indices) < num_samples_per_class:
            print(f'Warning: class {cls} has only {len(class_indices)} images in training set, requested {num_samples_per_class}')
            sampled_indices = class_indices
        else:
            sampled_indices = random.sample(class_indices, num_samples_per_class)
        for i in sampled_indices:
            in_domain_val_paths.append(train_paths[i])
            in_domain_val_labels.append(train_labels[i])

    # ood
    out_of_domain_val_paths = []
    out_of_domain_val_labels = []
    for cls in CelebADataset.CLASSES:
        class_indices = [i for i, l in enumerate(test_labels) if l == cls]
        if len(class_indices) == 0:
            print(f'Warning: class {cls} has no images in test set')
            continue
        sampled_indices = random.choices(class_indices, k = num_samples_per_class)
        for i in sampled_indices:
            out_of_domain_val_paths.append(test_paths[i])
            out_of_domain_val_labels.append(test_labels[i])

    return in_domain_val_paths, in_domain_val_labels, out_of_domain_val_paths, out_of_domain_val_labels


def get_celeba_dataloaders(data_root: str, target_attr: str, batch_size: int, num_val_samples_per_class: int, rndm_seed: int, num_workers: int, image_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    data_root = Path(data_root)
    img_dir = data_root / 'img_align_celeba'

    partitions, attrs = parse_celeba_metadata(data_root, target_attr)

    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    # 50/50 val imgs to test/train
    val_filenames = [f for f, p in partitions.items() if p == 1]
    random.seed(rndm_seed)
    random.shuffle(val_filenames)
    mid = len(val_filenames) // 2
    val_to_train = set(val_filenames[:mid])
    val_to_test = set(val_filenames[mid:])

    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []

    for filename, partition in partitions.items():
        if filename not in attrs:
            continue
        label = attrs[filename]
        img_path = img_dir / filename

        if partition == 0 or filename in val_to_train:
            train_paths.append(img_path)
            train_labels.append(label)
        elif partition == 2 or filename in val_to_test:
            test_paths.append(img_path)
            test_labels.append(label)

    in_domain_val_paths, in_domain_val_labels, out_of_domain_val_paths, out_of_domain_val_labels = create_validation_splits(train_paths = train_paths, train_labels = train_labels, test_paths = test_paths, test_labels = test_labels, num_samples_per_class = num_val_samples_per_class, seed = rndm_seed)

    in_domain_val_paths_set = set(in_domain_val_paths)
    filtered_train_paths = []
    filtered_train_labels = []
    for p, l in zip(train_paths, train_labels):
        if p not in in_domain_val_paths_set:
            filtered_train_paths.append(p)
            filtered_train_labels.append(l)
    train_paths = filtered_train_paths
    train_labels = filtered_train_labels

    train_dataset = CelebADataset(train_paths, train_labels, transform = data_transform)
    in_domain_val_dataset = CelebADataset(in_domain_val_paths, in_domain_val_labels, transform = data_transform)
    out_of_domain_val_dataset = CelebADataset(out_of_domain_val_paths, out_of_domain_val_labels, transform = data_transform)
    test_dataset = CelebADataset(test_paths, test_labels, transform = data_transform)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = use_pin_memory)
    in_domain_val_loader = DataLoader(in_domain_val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = use_pin_memory)
    out_of_domain_val_loader = DataLoader(out_of_domain_val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = use_pin_memory)

    print(f'Dataset Statistics (target_attr = {target_attr}):')
    print(f'  Train: {len(train_dataset)} images ({len(train_loader)} batches)')
    print(f'  In-Domain Validation: {len(in_domain_val_dataset)} images ({len(in_domain_val_loader)} batches)')
    print(f'  Out-of-Domain Validation: {len(out_of_domain_val_dataset)} images ({len(out_of_domain_val_loader)} batches)')
    print(f'  Test: {len(test_dataset)} images ({len(test_loader)} batches)')
    print(f'  Classes: {CelebADataset.CLASSES}')
    print(f'  Number of classes: {len(CelebADataset.CLASSES)}')

    return train_loader, in_domain_val_loader, out_of_domain_val_loader, test_loader
