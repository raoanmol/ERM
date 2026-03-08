import pytest
import torch
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from data.celeba_dataloader import (
    CelebADataset,
    parse_celeba_metadata,
    create_validation_splits,
    get_celeba_dataloaders,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / 'data' / 'celeba'
TARGET_ATTR = 'Blond_Hair'


# --- CelebADataset ---

@pytest.fixture
def sample_paths_and_labels():
    img_dir = DATA_ROOT / 'img_align_celeba'
    paths = sorted(img_dir.glob('*.jpg'))[:10]
    assert len(paths) > 0, "No images found — is data/celeba/img_align_celeba populated?"
    labels = [i % 2 for i in range(len(paths))]
    return paths, labels


class TestCelebADataset:
    def test_len(self, sample_paths_and_labels):
        paths, labels = sample_paths_and_labels
        ds = CelebADataset(paths, labels)
        assert len(ds) == len(paths)

    def test_getitem_without_transform(self, sample_paths_and_labels):
        paths, labels = sample_paths_and_labels
        ds = CelebADataset(paths, labels)
        image, label = ds[0]
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)
        assert label in CelebADataset.CLASSES

    def test_getitem_with_transform(self, sample_paths_and_labels):
        paths, labels = sample_paths_and_labels
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        ds = CelebADataset(paths, labels, transform=transform)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            CelebADataset([Path('a.jpg')], [0, 1])


# --- parse_celeba_metadata ---

class TestParseCelebaMetadata:
    def test_returns_partitions_and_attrs(self):
        partitions, attrs = parse_celeba_metadata(DATA_ROOT, TARGET_ATTR)
        assert len(partitions) > 0
        assert len(attrs) > 0

    def test_partition_values(self):
        partitions, _ = parse_celeba_metadata(DATA_ROOT, TARGET_ATTR)
        for val in partitions.values():
            assert val in (0, 1, 2)

    def test_attr_values_are_binary(self):
        _, attrs = parse_celeba_metadata(DATA_ROOT, TARGET_ATTR)
        for val in attrs.values():
            assert val in (0, 1)

    def test_invalid_attr_raises(self):
        with pytest.raises(ValueError, match="not found"):
            parse_celeba_metadata(DATA_ROOT, 'NonExistent_Attribute')

    def test_filenames_consistent(self):
        partitions, attrs = parse_celeba_metadata(DATA_ROOT, TARGET_ATTR)
        # Every file in attrs should also be in partitions
        for filename in attrs:
            assert filename in partitions


# --- create_validation_splits ---

class TestCreateValidationSplits:
    @pytest.fixture
    def train_and_test_data(self):
        partitions, attrs = parse_celeba_metadata(DATA_ROOT, TARGET_ATTR)
        img_dir = DATA_ROOT / 'img_align_celeba'
        train_paths, train_labels, test_paths, test_labels = [], [], [], []
        for filename, partition in partitions.items():
            if filename not in attrs:
                continue
            if partition == 0:
                train_paths.append(img_dir / filename)
                train_labels.append(attrs[filename])
            elif partition == 2:
                test_paths.append(img_dir / filename)
                test_labels.append(attrs[filename])
        return train_paths, train_labels, test_paths, test_labels

    def test_returns_correct_count(self, train_and_test_data):
        train_paths, train_labels, test_paths, test_labels = train_and_test_data
        num_per_class = 5
        iid_paths, iid_labels, ood_paths, ood_labels = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, num_per_class, seed=42
        )
        assert len(iid_paths) == num_per_class * len(CelebADataset.CLASSES)
        assert len(iid_labels) == num_per_class * len(CelebADataset.CLASSES)
        assert len(ood_paths) == num_per_class * len(CelebADataset.CLASSES)
        assert len(ood_labels) == num_per_class * len(CelebADataset.CLASSES)

    def test_iid_labels_match_class(self, train_and_test_data):
        train_paths, train_labels, test_paths, test_labels = train_and_test_data
        iid_paths, iid_labels, _, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 5, seed=42
        )
        # First num_per_class should be class 0, next num_per_class should be class 1
        for label in iid_labels[:5]:
            assert label == 0
        for label in iid_labels[5:]:
            assert label == 1

    def test_ood_labels_match_class(self, train_and_test_data):
        train_paths, train_labels, test_paths, test_labels = train_and_test_data
        _, _, ood_paths, ood_labels = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 5, seed=42
        )
        for label in ood_labels[:5]:
            assert label == 0
        for label in ood_labels[5:]:
            assert label == 1

    def test_iid_paths_come_from_train(self, train_and_test_data):
        train_paths, train_labels, test_paths, test_labels = train_and_test_data
        train_set = set(train_paths)
        iid_paths, _, _, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 3, seed=42
        )
        for p in iid_paths:
            assert p in train_set

    def test_ood_paths_come_from_test(self, train_and_test_data):
        train_paths, train_labels, test_paths, test_labels = train_and_test_data
        test_set = set(test_paths)
        _, _, ood_paths, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 3, seed=42
        )
        for p in ood_paths:
            assert p in test_set

    def test_deterministic_with_seed(self, train_and_test_data):
        train_paths, train_labels, test_paths, test_labels = train_and_test_data
        iid_a, _, ood_a, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 5, seed=123
        )
        iid_b, _, ood_b, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 5, seed=123
        )
        assert iid_a == iid_b
        assert ood_a == ood_b

    def test_iid_no_duplicates(self, train_and_test_data):
        train_paths, train_labels, test_paths, test_labels = train_and_test_data
        iid_paths, _, _, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 5, seed=42
        )
        # In-domain uses sample (without replacement), so no duplicates
        assert len(iid_paths) == len(set(iid_paths))


# --- get_celeba_dataloaders (integration) ---

class TestGetCelebaDataloaders:
    def test_returns_four_loaders(self):
        loaders = get_celeba_dataloaders(
            data_root=str(DATA_ROOT),
            target_attr=TARGET_ATTR,
            batch_size=4,
            num_val_samples_per_class=2,
            rndm_seed=42,
            num_workers=0,
            image_size=224,
        )
        assert len(loaders) == 4
        for loader in loaders:
            assert hasattr(loader, '__iter__')

    def test_batch_shape(self):
        train_loader, *_ = get_celeba_dataloaders(
            data_root=str(DATA_ROOT),
            target_attr=TARGET_ATTR,
            batch_size=8,
            num_val_samples_per_class=2,
            rndm_seed=42,
            num_workers=0,
            image_size=224,
        )
        images, labels = next(iter(train_loader))
        assert images.shape == (8, 3, 224, 224)
        assert labels.shape == (8,)

    def test_train_excludes_val(self):
        num_val = 5
        train_loader, id_val_loader, _, _ = get_celeba_dataloaders(
            data_root=str(DATA_ROOT),
            target_attr=TARGET_ATTR,
            batch_size=4,
            num_val_samples_per_class=num_val,
            rndm_seed=42,
            num_workers=0,
            image_size=224,
        )
        train_paths = set(train_loader.dataset.image_paths)
        val_paths = set(id_val_loader.dataset.image_paths)
        assert train_paths.isdisjoint(val_paths)

    def test_labels_are_binary(self):
        loaders = get_celeba_dataloaders(
            data_root=str(DATA_ROOT),
            target_attr=TARGET_ATTR,
            batch_size=16,
            num_val_samples_per_class=2,
            rndm_seed=42,
            num_workers=0,
            image_size=224,
        )
        for loader in loaders:
            _, labels = next(iter(loader))
            for l in labels:
                assert l.item() in (0, 1)

    def test_val_split_adds_to_train_and_test(self):
        partitions, _ = parse_celeba_metadata(DATA_ROOT, TARGET_ATTR)
        num_partition_0 = sum(1 for p in partitions.values() if p == 0)
        num_partition_1 = sum(1 for p in partitions.values() if p == 1)
        num_partition_2 = sum(1 for p in partitions.values() if p == 2)

        num_val = 2
        train_loader, id_val_loader, _, test_loader = get_celeba_dataloaders(
            data_root=str(DATA_ROOT),
            target_attr=TARGET_ATTR,
            batch_size=4,
            num_val_samples_per_class=num_val,
            rndm_seed=42,
            num_workers=0,
            image_size=224,
        )
        total_train_side = len(train_loader.dataset) + len(id_val_loader.dataset)
        total_test_side = len(test_loader.dataset)

        mid = num_partition_1 // 2
        expected_train_side = num_partition_0 + mid
        expected_test_side = num_partition_2 + (num_partition_1 - mid)

        assert total_train_side == expected_train_side
        assert total_test_side == expected_test_side
