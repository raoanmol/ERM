import pytest
import torch
import random
from pathlib import Path
from PIL import Image
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))

from data.bar_dataloader import (
    BARDataset,
    get_activity_from_filename,
    create_validation_splits,
    get_bar_dataloaders,
)

DATA_ROOT = Path(__file__).resolve().parent.parent / 'data' / 'bar'
TRAIN_DIR = DATA_ROOT / 'train'
TEST_DIR = DATA_ROOT / 'test'


# --- get_activity_from_filename ---

class TestGetActivityFromFilename:
    @pytest.mark.parametrize("filename,expected", [
        ("climbing_0.jpg", "climbing"),
        ("diving_42.jpg", "diving"),
        ("fishing_100.jpg", "fishing"),
        ("pole vaulting_7.jpg", "pole vaulting"),
        ("racing_999.jpg", "racing"),
        ("throwing_1.jpg", "throwing"),
    ])
    def test_all_classes(self, filename, expected):
        assert get_activity_from_filename(filename) == expected

    def test_pole_vaulting_special_case(self):
        # 'pole vaulting' has a space, so the special-case logic must trigger
        assert get_activity_from_filename("pole vaulting_55.jpg") == "pole vaulting"

    def test_regular_class_uses_rsplit(self):
        assert get_activity_from_filename("climbing_123.jpg") == "climbing"


# --- BARDataset ---

@pytest.fixture
def sample_train_paths():
    paths = list(TRAIN_DIR.glob('*.jpg'))[:10]
    assert len(paths) > 0, "No training images found — is data/bar/train populated?"
    return paths


class TestBARDataset:
    def test_len(self, sample_train_paths):
        ds = BARDataset(sample_train_paths)
        assert len(ds) == len(sample_train_paths)

    def test_getitem_without_transform(self, sample_train_paths):
        ds = BARDataset(sample_train_paths)
        image, label = ds[0]
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)
        assert 0 <= label < len(BARDataset.CLASSES)

    def test_getitem_with_transform(self, sample_train_paths):
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        ds = BARDataset(sample_train_paths, transform=transform)
        image, label = ds[0]
        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 224, 224)
        assert isinstance(label, int)

    def test_invalid_class_raises(self, tmp_path):
        # Create a fake image with an unrecognized class name
        fake_img_path = tmp_path / "badclass_0.jpg"
        Image.new('RGB', (64, 64)).save(fake_img_path)

        ds = BARDataset([fake_img_path])
        with pytest.raises(ValueError, match="Unrecognized class"):
            ds[0]

    def test_class_to_idx_mapping(self):
        ds = BARDataset([])
        for idx, cls in enumerate(BARDataset.CLASSES):
            assert ds.class_to_idx[cls] == idx


# --- create_validation_splits ---

class TestCreateValidationSplits:
    def test_returns_correct_count(self):
        num_per_class = 5
        iid, ood = create_validation_splits(TRAIN_DIR, TEST_DIR, num_per_class, seed=42)
        assert len(iid) == num_per_class * len(BARDataset.CLASSES)
        assert len(ood) == num_per_class * len(BARDataset.CLASSES)

    def test_iid_paths_come_from_train(self):
        iid, _ = create_validation_splits(TRAIN_DIR, TEST_DIR, 3, seed=42)
        for p in iid:
            assert p.parent == TRAIN_DIR

    def test_ood_paths_come_from_test(self):
        _, ood = create_validation_splits(TRAIN_DIR, TEST_DIR, 3, seed=42)
        for p in ood:
            assert p.parent == TEST_DIR

    def test_deterministic_with_seed(self):
        iid_a, ood_a = create_validation_splits(TRAIN_DIR, TEST_DIR, 5, seed=123)
        iid_b, ood_b = create_validation_splits(TRAIN_DIR, TEST_DIR, 5, seed=123)
        assert sorted(iid_a) == sorted(iid_b)
        assert sorted(ood_a) == sorted(ood_b)

    def test_no_iid_ood_overlap(self):
        iid, ood = create_validation_splits(TRAIN_DIR, TEST_DIR, 5, seed=42)
        assert set(iid).isdisjoint(set(ood))


# --- get_bar_dataloaders (integration) ---

class TestGetBarDataloaders:
    def test_returns_four_loaders(self):
        loaders = get_bar_dataloaders(
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
        train_loader, *_ = get_bar_dataloaders(
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
        train_loader, id_val_loader, _, _ = get_bar_dataloaders(
            data_root=str(DATA_ROOT),
            batch_size=4,
            num_val_samples_per_class=5,
            rndm_seed=42,
            num_workers=0,
            image_size=224,
        )
        train_count = len(train_loader.dataset)
        id_val_count = len(id_val_loader.dataset)
        total_train_images = len(list(TRAIN_DIR.glob('*.jpg')))
        assert train_count + id_val_count == total_train_images
