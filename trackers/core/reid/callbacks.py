import os
from typing import Any, Optional

import matplotlib.pyplot as plt


class BaseCallback:
    def on_train_batch_start(self, logs: dict, idx: int):
        pass

    def on_train_batch_end(self, logs: dict, idx: int):
        pass

    def on_train_epoch_end(self, logs: dict, epoch: int):
        pass

    def on_validation_batch_start(self, logs: dict, idx: int):
        pass

    def on_validation_batch_end(self, logs: dict, idx: int):
        pass

    def on_validation_epoch_end(self, logs: dict, epoch: int):
        pass

    def on_checkpoint_save(self, checkpoint_path: str, epoch: int):
        pass

    def on_end(self):
        pass


class TensorboardCallback(BaseCallback):
    def __init__(
        self,
        log_dir: Optional[str] = None,
        comment: str = "",
        purge_step: Optional[Any] = None,
        max_queue: int = 10,
        flush_secs: int = 120,
        filename_suffix: str = "",
    ):
        from torch.utils.tensorboard import SummaryWriter

        self.writer = SummaryWriter(
            log_dir,
            comment=comment,
            filename_suffix=filename_suffix,
            purge_step=purge_step,
            max_queue=max_queue,
            flush_secs=flush_secs,
        )

    def on_train_batch_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, idx)

    def on_train_epoch_end(self, logs: dict, epoch: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, epoch)

    def on_validation_epoch_end(self, logs: dict, epoch: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, epoch)

    def on_end(self):
        self.writer.flush()
        self.writer.close()


class WandbCallback(BaseCallback):
    def __init__(self, config: dict[str, Any]) -> None:
        import wandb

        self.run = wandb.init(config=config) if not wandb.run else wandb.run  # type: ignore

        self.run.define_metric("batch/step")
        self.run.define_metric("batch/train/loss", step_metric="batch/step")

        self.run.define_metric("epoch")
        self.run.define_metric("train/loss", step_metric="epoch")
        self.run.define_metric("validation/loss", step_metric="epoch")

    def on_train_batch_end(self, logs: dict, idx: int):
        logs["batch/step"] = idx
        self.run.log(logs)

    def on_train_epoch_end(self, logs: dict, epoch: int):
        logs["epoch"] = epoch
        self.run.log(logs)

    def on_validation_epoch_end(self, logs: dict, epoch: int):
        logs["epoch"] = epoch
        self.run.log(logs)

    def on_checkpoint_save(self, checkpoint_path: str, epoch: int):
        self.run.log_model(
            path=checkpoint_path,
            name=f"checkpoint_{self.run.id}",
            aliases=[f"epoch-{epoch}", "latest"],
        )

    def on_end(self):
        self.run.finish()


class MatplotlibCallback(BaseCallback):
    def __init__(self, log_dir: str, max_columns: int = 3):
        self.log_dir = log_dir
        self.max_columns = max_columns
        self.train_history: dict[str, list[tuple[int, float]]] = {}
        self.validation_history: dict[str, list[tuple[int, float]]] = {}

    def on_train_batch_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            self.train_history.setdefault(key, []).append((idx, value))

    def on_train_epoch_end(self, logs: dict, epoch: int):
        for key, value in logs.items():
            self.train_history.setdefault(key, []).append((epoch, value))

    def on_validation_epoch_end(self, logs: dict, epoch: int):
        for key, value in logs.items():
            self.validation_history.setdefault(key, []).append((epoch, value))

    def on_end(self):
        all_metrics_data = list(
            set(self.train_history.keys()) | set(self.validation_history.keys())
        )
        if not all_metrics_data:
            return

        batch_metrics_names = sorted(
            [m for m in all_metrics_data if m.startswith("batch/")]
        )

        epoch_metrics_all_keys = sorted(
            [m for m in all_metrics_data if not m.startswith("batch/")]
        )
        epoch_metrics_grouped = {}
        for metric_name in epoch_metrics_all_keys:
            parts = metric_name.split("/")
            base_name = parts[-1]
            epoch_metrics_grouped.setdefault(base_name, []).append(metric_name)

        num_batch_plots = len(batch_metrics_names)
        num_epoch_group_plots = len(epoch_metrics_grouped)
        num_plots = num_batch_plots + num_epoch_group_plots

        if num_plots == 0:
            return

        n_cols = min(self.max_columns, num_plots)
        n_rows = (num_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False
        )
        axes = axes.flatten()

        plot_idx = 0

        for metric_name in batch_metrics_names:
            ax = axes[plot_idx]
            train_data = self.train_history.get(metric_name, [])
            val_data = self.validation_history.get(metric_name, [])

            plotted_anything = False
            if train_data:
                x_train, y_train = zip(*train_data)
                ax.plot(x_train, y_train, label="train", marker="o")
                plotted_anything = True
            if val_data:
                x_val, y_val = zip(*val_data)
                ax.plot(x_val, y_val, label="validation", marker="o", linestyle="--")
                plotted_anything = True

            base_metric_name = metric_name.split("/")[-1]
            ax.set_title(f"Batch {base_metric_name.capitalize()}")
            ax.set_xlabel("batch")
            ax.set_ylabel(base_metric_name)
            if plotted_anything:
                ax.legend()
            ax.grid(True)
            plot_idx += 1

        for base_name, _ in sorted(epoch_metrics_grouped.items()):
            ax = axes[plot_idx]
            plotted_anything_for_group = False

            train_series_data = []
            if f"train/{base_name}" in self.train_history:
                train_series_data = self.train_history[f"train/{base_name}"]
            elif base_name in self.train_history:
                train_series_data = self.train_history[base_name]

            if train_series_data:
                x_pts, y_pts = zip(*train_series_data)
                ax.plot(x_pts, y_pts, label="train", marker="o")
                plotted_anything_for_group = True

            validation_series_data = []
            if f"validation/{base_name}" in self.validation_history:
                validation_series_data = self.validation_history[
                    f"validation/{base_name}"
                ]
            elif base_name in self.validation_history:
                validation_series_data = self.validation_history[base_name]

            if validation_series_data:
                x_pts, y_pts = zip(*validation_series_data)
                ax.plot(x_pts, y_pts, label="validation", marker="o", linestyle="--")
                plotted_anything_for_group = True

            ax.set_title(f"Epoch {base_name.capitalize()}")
            ax.set_xlabel("epoch")
            ax.set_ylabel(base_name)

            if plotted_anything_for_group:
                ax.legend()
            ax.grid(True)
            plot_idx += 1

        for j in range(plot_idx, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        fig.savefig(os.path.join(self.log_dir, "metrics_plot.png"))
        plt.show()
        plt.close(fig)
