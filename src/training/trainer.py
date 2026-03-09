import os
import json
import torch
import torch.nn as nn
from typing import List, Tuple
from src.utils.logger import Logger
from torch.utils.data import DataLoader, Subset
from src.models.model_factory import create_model
from src.utils.checkpoint import save_checkpoint, load_checkpoint


def _get_image_paths(dataset):
    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        return [base_dataset.image_paths[i] for i in dataset.indices]
    return dataset.image_paths


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        id_val_loader: DataLoader,
        ood_val_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        logger: Logger,
        checkpoint_dir: str,
        num_classes: int,
        model_name: str,
    ):
        self.model = model
        self.train_loader = train_loader
        self.id_val_loader = id_val_loader
        self.ood_val_loader = ood_val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir
        self.num_classes = num_classes
        self.model_name = model_name
        self.best_id_val_acc = -1.0
        self.best_ood_val_acc = -1.0
        self.best_id_path = os.path.join(checkpoint_dir, "best_id_val.pt")
        self.best_ood_path = os.path.join(checkpoint_dir, "best_ood_val.pt")

    def train_one_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
        return correct / total

    @torch.no_grad()
    def evaluate_with_predictions(self, loader: DataLoader) -> Tuple[float, List[dict]]:
        self.model.eval()
        correct = 0
        total = 0
        all_logits = []
        all_preds = []
        all_labels = []
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            all_logits.append(outputs.cpu())
            all_preds.append(predicted.cpu())
            all_labels.append(labels.cpu())
        all_logits = torch.cat(all_logits, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        image_paths = _get_image_paths(loader.dataset)
        predictions = []
        for i in range(len(image_paths)):
            predictions.append(
                {
                    "file_path": str(image_paths[i]),
                    "ground_truth": int(all_labels[i].item()),
                    "predicted_class": int(all_preds[i].item()),
                    "logits": [round(v, 4) for v in all_logits[i].tolist()],
                }
            )
        accuracy = correct / total
        return accuracy, predictions

    def train(self, num_epochs: int) -> None:
        for epoch in range(1, num_epochs + 1):
            train_loss, train_acc = self.train_one_epoch()
            id_val_acc = self.evaluate(self.id_val_loader)
            ood_val_acc = self.evaluate(self.ood_val_loader)
            self.logger.log_epoch(epoch, train_loss, train_acc, id_val_acc, ood_val_acc)
            print(
                f"Epoch {epoch}/{num_epochs} | Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | ID Val Acc: {id_val_acc:.4f} | OOD Val Acc: {ood_val_acc:.4f}"
            )
            if id_val_acc > self.best_id_val_acc:
                self.best_id_val_acc = id_val_acc
                save_checkpoint(self.model, self.best_id_path, epoch, id_val_acc)
            if ood_val_acc > self.best_ood_val_acc:
                self.best_ood_val_acc = ood_val_acc
                save_checkpoint(self.model, self.best_ood_path, epoch, ood_val_acc)

    def test(self) -> Tuple[float, float]:
        original_model = self.model

        id_model = create_model(self.model_name, self.num_classes, pretrained=False)
        load_checkpoint(id_model, self.best_id_path)
        id_model.to(self.device)
        self.model = id_model
        best_id_test_acc, id_predictions = self.evaluate_with_predictions(
            self.test_loader
        )
        print(f"Test Acc (best ID val model): {best_id_test_acc:.4f}")

        id_pred_path = os.path.join(
            self.checkpoint_dir, "test_predictions_best_id_val.json"
        )
        with open(id_pred_path, "w") as f:
            json.dump(id_predictions, f, indent=2)

        ood_model = create_model(self.model_name, self.num_classes, pretrained=False)
        load_checkpoint(ood_model, self.best_ood_path)
        ood_model.to(self.device)
        self.model = ood_model
        best_ood_test_acc, ood_predictions = self.evaluate_with_predictions(
            self.test_loader
        )
        print(f"Test Acc (best OOD val model): {best_ood_test_acc:.4f}")

        ood_pred_path = os.path.join(
            self.checkpoint_dir, "test_predictions_best_ood_val.json"
        )
        with open(ood_pred_path, "w") as f:
            json.dump(ood_predictions, f, indent=2)

        self.model = original_model
        self.logger.log_test(best_id_test_acc, best_ood_test_acc)
        return best_id_test_acc, best_ood_test_acc
