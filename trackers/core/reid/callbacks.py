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
    def __init__(self):
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
        metrics = set(self.train_history) | set(self.validation_history)
        for metric in metrics:
            train_data = self.train_history.get(metric, [])
            val_data = self.validation_history.get(metric, [])
            plt.figure()
            if train_data:
                x_train, y_train = zip(*train_data)
                plt.plot(x_train, y_train, label="train", color="blue", marker="o")
            if val_data:
                x_val, y_val = zip(*val_data)
                plt.plot(x_val, y_val, label="validation", color="orange", marker="x")
            plt.title(metric)
            plt.xlabel("batch")
            plt.ylabel(metric)
            plt.legend()
            plt.show()
            plt.close()
