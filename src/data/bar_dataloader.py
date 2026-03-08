import torch
import random
from PIL import Image
from pathlib import Path
from torchvision import transforms
from typing import Literal, Tuple, List
from torch.utils.data import Dataset, DataLoader

class BARDataset(Dataset):
    CLASSES = ['climbing', 'diving', 'fishing', 'pole vaulting', 'racing', 'throwing']

    def __init__(self, image_paths: List[Path], transform = None):
        self.image_paths = image_paths
        self.transform = transform

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')

        label_str = get_activity_from_filename(img_path.name)

        if label_str not in self.class_to_idx:
            raise ValueError(f"Unrecognized class '{label_str}' parsed from filename '{img_path.name}'. Expected one of : {self.CLASSES}")

        label = self.class_to_idx[label_str]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_activity_from_filename(filename: str) -> str:
    if filename.startswith('pole vaulting_'):
        return 'pole vaulting'
    else:
        return filename.rsplit('_', 1)[0]


def create_validation_splits(train_dir: Path, test_dir: Path, num_samples_per_class: int, seed: int) -> Tuple[List[Path], List[Path]]:
    random.seed(seed)

    # iid
    train_activity_images = {cls: [] for cls in BARDataset.CLASSES}
    for img_path in train_dir.glob('*.jpg'):
        activity = get_activity_from_filename(img_path.name)
        if activity in train_activity_images:
            train_activity_images[activity].append(img_path)

    in_domain_val_paths = []
    for activity, images in train_activity_images.items():
        if len(images) < num_samples_per_class:
            print(f'Warning: {activity} has only {len(images)} images in the training set, requested {num_samples_per_class}')
            sampled = images
        else:
            sampled = random.sample(images, num_samples_per_class)
        in_domain_val_paths.extend(sampled)

    # ood
    test_activity_images = {cls: [] for cls in BARDataset.CLASSES}
    for img_path in test_dir.glob('*.jpg'):
        activity = get_activity_from_filename(img_path.name)
        if activity in test_activity_images:
            test_activity_images[activity].append(img_path)

    out_of_domain_val_paths = []
    for activity, images in test_activity_images.items():
        if len(images) < num_samples_per_class:
            print(f'Warning: {activity} has only {len(images)} images in the test set, requested {num_samples_per_class}')
            sampled = images
        else:
            sampled = random.sample(images, num_samples_per_class)
        out_of_domain_val_paths.extend(sampled)

    return in_domain_val_paths, out_of_domain_val_paths


def get_bar_dataloaders(data_root: str, batch_size: int, num_val_samples_per_class: int, rndm_seed: int, num_workers: int, image_size: int) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    data_root = Path(data_root)
    train_dir = data_root / 'train'
    test_dir = data_root / 'test'

    data_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
    ])

    in_domain_val_paths, out_of_domain_val_paths = create_validation_splits(
        train_dir = train_dir,
        test_dir = test_dir,
        num_samples_per_class = num_val_samples_per_class,
        seed = rndm_seed
    )
    in_domain_val_paths_set = set(in_domain_val_paths)

    train_paths = []
    for img_path in train_dir.glob('*.jpg'):
        if img_path not in in_domain_val_paths_set:
            train_paths.append(img_path)

    test_paths = list(test_dir.glob('*.jpg'))

    train_dataset = BARDataset(train_paths, transform = data_transform)
    in_domain_val_dataset = BARDataset(in_domain_val_paths, transform = data_transform)
    out_of_domain_val_dataset = BARDataset(out_of_domain_val_paths, transform = data_transform)
    test_dataset = BARDataset(test_paths, transform = data_transform)

    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = use_pin_memory)
    in_domain_val_loader = DataLoader(in_domain_val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = use_pin_memory)
    out_of_domain_val_loader = DataLoader(out_of_domain_val_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = use_pin_memory)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, num_workers = num_workers, pin_memory = use_pin_memory)

    print(f'Dataset Statistics:')
    print(f'  Train: {len(train_dataset)} images ({len(train_loader)} batches)')
    print(f'  In-Domain Validation: {len(in_domain_val_dataset)} images ({len(in_domain_val_loader)} batches)')
    print(f'  Out-of-Domain Validation: {len(out_of_domain_val_dataset)} images ({len(out_of_domain_val_loader)} batches)')
    print(f'  Test: {len(test_dataset)} images ({len(test_loader)} batches)')
    print(f'  Classes: {BARDataset.CLASSES}')
    print(f'  Number of classes: {len(BARDataset.CLASSES)}')

    return train_loader, in_domain_val_loader, out_of_domain_val_loader, test_loader
