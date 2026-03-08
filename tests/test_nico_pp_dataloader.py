import pytest
import torch
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from data.nico_pp_dataloader import (
    NICOPPDataset,
    collect_all_images,
    create_validation_splits,
    get_nico_pp_dataloaders,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / 'data' / 'nico_pp'


# --- NICOPPDataset ---

@pytest.fixture
def sample_paths_and_labels():
    all_paths, all_labels, _ = collect_all_images(DATA_ROOT)
    paths = all_paths[:10]
    labels = all_labels[:10]
    assert len(paths) > 0, "No images found — is data/nico_pp populated?"
    return paths, labels


@pytest.fixture
def class_to_idx():
    _, _, c2i = collect_all_images(DATA_ROOT)
    return c2i


class TestNICOPPDataset:
    def test_len(self, sample_paths_and_labels):
        paths, labels = sample_paths_and_labels
        ds = NICOPPDataset(paths, labels)
        assert len(ds) == len(paths)

    def test_getitem_without_transform(self, sample_paths_and_labels):
        paths, labels = sample_paths_and_labels
        ds = NICOPPDataset(paths, labels)
        image, label = ds[0]
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)

    def test_getitem_with_transform(self, sample_paths_and_labels):
        paths, labels = sample_paths_and_labels
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        ds = NICOPPDataset(paths, labels, transform=transform)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_mismatched_lengths_raises(self):
        with pytest.raises(AssertionError):
            NICOPPDataset([Path('a.jpg')], [0, 1])


# --- collect_all_images ---

class TestCollectAllImages:
    def test_returns_nonempty(self):
        paths, labels, class_to_idx = collect_all_images(DATA_ROOT)
        assert len(paths) > 0
        assert len(labels) > 0
        assert len(class_to_idx) == 60

    def test_paths_and_labels_same_length(self):
        paths, labels, _ = collect_all_images(DATA_ROOT)
        assert len(paths) == len(labels)

    def test_labels_in_valid_range(self):
        _, labels, class_to_idx = collect_all_images(DATA_ROOT)
        valid_ids = set(class_to_idx.values())
        for l in labels:
            assert l in valid_ids

    def test_paths_are_jpg_files(self):
        paths, _, _ = collect_all_images(DATA_ROOT)
        for p in paths[:100]:
            assert p.suffix == '.jpg'


# --- create_validation_splits ---

class TestCreateValidationSplits:
    @pytest.fixture
    def train_test_split(self, class_to_idx):
        all_paths, all_labels, _ = collect_all_images(DATA_ROOT)
        combined = list(zip(all_paths, all_labels))
        random.seed(42)
        random.shuffle(combined)
        split_idx = int(len(combined) * 0.75)
        train_paths = [p for p, _ in combined[:split_idx]]
        train_labels = [l for _, l in combined[:split_idx]]
        test_paths = [p for p, _ in combined[split_idx:]]
        test_labels = [l for _, l in combined[split_idx:]]
        return train_paths, train_labels, test_paths, test_labels, class_to_idx

    def test_returns_correct_count(self, train_test_split):
        train_paths, train_labels, test_paths, test_labels, c2i = train_test_split
        num_per_class = 3
        iid_paths, iid_labels, ood_paths, ood_labels = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, num_per_class, seed=42, class_to_idx=c2i
        )
        assert len(iid_paths) == num_per_class * len(c2i)
        assert len(iid_labels) == num_per_class * len(c2i)
        assert len(ood_paths) == num_per_class * len(c2i)
        assert len(ood_labels) == num_per_class * len(c2i)

    def test_iid_paths_come_from_train(self, train_test_split):
        train_paths, train_labels, test_paths, test_labels, c2i = train_test_split
        train_set = set(train_paths)
        iid_paths, _, _, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 3, seed=42, class_to_idx=c2i
        )
        for p in iid_paths:
            assert p in train_set

    def test_ood_paths_come_from_test(self, train_test_split):
        train_paths, train_labels, test_paths, test_labels, c2i = train_test_split
        test_set = set(test_paths)
        _, _, ood_paths, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 3, seed=42, class_to_idx=c2i
        )
        for p in ood_paths:
            assert p in test_set

    def test_deterministic_with_seed(self, train_test_split):
        train_paths, train_labels, test_paths, test_labels, c2i = train_test_split
        iid_a, _, ood_a, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 3, seed=123, class_to_idx=c2i
        )
        iid_b, _, ood_b, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 3, seed=123, class_to_idx=c2i
        )
        assert iid_a == iid_b
        assert ood_a == ood_b

    def test_iid_no_duplicates(self, train_test_split):
        train_paths, train_labels, test_paths, test_labels, c2i = train_test_split
        iid_paths, _, _, _ = create_validation_splits(
            train_paths, train_labels, test_paths, test_labels, 3, seed=42, class_to_idx=c2i
        )
        assert len(iid_paths) == len(set(iid_paths))


# --- get_nico_pp_dataloaders (integration) ---

class TestGetNicoPPDataloaders:
    def test_returns_four_loaders(self):
        loaders = get_nico_pp_dataloaders(
            data_root=str(DATA_ROOT),
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
        train_loader, *_ = get_nico_pp_dataloaders(
            data_root=str(DATA_ROOT),
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
        train_loader, id_val_loader, _, _ = get_nico_pp_dataloaders(
            data_root=str(DATA_ROOT),
            batch_size=4,
            num_val_samples_per_class=2,
            rndm_seed=42,
            num_workers=0,
            image_size=224,
        )
        train_paths = set(train_loader.dataset.image_paths)
        val_paths = set(id_val_loader.dataset.image_paths)
        assert train_paths.isdisjoint(val_paths)

    def test_75_25_split(self):
        all_paths, _, _ = collect_all_images(DATA_ROOT)
        total = len(all_paths)
        expected_train_side = int(total * 0.75)
        expected_test_side = total - expected_train_side

        num_val = 2
        train_loader, id_val_loader, _, test_loader = get_nico_pp_dataloaders(
            data_root=str(DATA_ROOT),
            batch_size=4,
            num_val_samples_per_class=num_val,
            rndm_seed=42,
            num_workers=0,
            image_size=224,
        )
        total_train_side = len(train_loader.dataset) + len(id_val_loader.dataset)
        total_test_side = len(test_loader.dataset)

        assert total_train_side == expected_train_side
        assert total_test_side == expected_test_side

    def test_labels_in_valid_range(self):
        loaders = get_nico_pp_dataloaders(
            data_root=str(DATA_ROOT),
            batch_size=16,
            num_val_samples_per_class=2,
            rndm_seed=42,
            num_workers=0,
            image_size=224,
        )
        for loader in loaders:
            _, labels = next(iter(loader))
            for l in labels:
                assert 0 <= l.item() < 60
