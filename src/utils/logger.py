import os
from src.utils.config import LoggingConfig


class Logger:
    def __init__(self, log_cfg: LoggingConfig, run_name: str, config_dict: dict):
        self.log_dir = os.path.join(log_cfg.log_dir, run_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_file = open(os.path.join(self.log_dir, "metrics.csv"), "w")
        self.csv_file.write("epoch,train_loss,train_acc,id_val_acc,ood_val_acc\n")
        self.csv_file.flush()

        if log_cfg.use_wandb:
            import wandb

            self.wandb_run = wandb.init(
                project=log_cfg.wandb_project,
                entity=log_cfg.wandb_entity,
                name=run_name,
                config=config_dict,
            )
        else:
            self.wandb_run = None

    def log_epoch(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        id_val_acc: float,
        ood_val_acc: float,
    ) -> None:
        self.csv_file.write(
            f"{epoch},{train_loss:.6f},{train_acc:.4f},{id_val_acc:.4f},{ood_val_acc:.4f}\n"
        )
        self.csv_file.flush()

        if self.wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "id_val_acc": id_val_acc,
                    "ood_val_acc": ood_val_acc,
                }
            )

    def log_test(self, best_id_test_acc: float, best_ood_test_acc: float) -> None:
        with open(os.path.join(self.log_dir, "test_results.csv"), "w") as f:
            f.write("model_selection,test_acc\n")
            f.write(f"best_id_val,{best_id_test_acc:.4f}\n")
            f.write(f"best_ood_val,{best_ood_test_acc:.4f}\n")

        if self.wandb_run is not None:
            import wandb

            wandb.log(
                {
                    "test_acc_from_best_id_val": best_id_test_acc,
                    "test_acc_from_best_ood_val": best_ood_test_acc,
                }
            )

    def close(self) -> None:
        self.csv_file.close()
        if self.wandb_run is not None:
            import wandb

            wandb.finish()
