from typing import Any, Optional

import matplotlib.pyplot as plt


class BaseCallback:
    def on_train_batch_start(self, logs: dict, idx: int):
        pass

    def on_train_batch_end(self, logs: dict, idx: int):
        pass

    def on_validation_batch_start(self, logs: dict, idx: int):
        pass

    def on_validation_batch_end(self, logs: dict, idx: int):
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

    def on_validation_batch_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, idx)

    def on_end(self):
        self.writer.flush()
        self.writer.close()


class MatplotlibCallback(BaseCallback):
    def __init__(self):
        self.metrics = {"train": {}, "val": {}}

    def on_train_batch_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            if key not in self.metrics["train"]:
                self.metrics["train"][key] = []
            self.metrics["train"][key].append((idx, value))

    def on_validation_batch_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            if key not in self.metrics["val"]:
                self.metrics["val"][key] = []
            self.metrics["val"][key].append((idx, value))

    def on_end(self):
        for phase in ["train", "val"]:
            for key, values in self.metrics[phase].items():
                if not values:
                    continue
                x, y = zip(*values)
                plt.figure()
                plt.plot(x, y, label=f"{phase}")
                plt.xlabel("Step")
                plt.ylabel(key)
                plt.title(f"{key} ({phase})")
                plt.legend()
                plt.close()


class WandbCallback(BaseCallback):
    def __init__(self, config: dict[str, Any]) -> None:
        import wandb

        self.run = wandb.init(config=config) if not wandb.run else wandb.run  # type: ignore

        self.run.define_metric("train/batch_step")
        self.run.define_metric("train/loss", step_metric="train/batch_step")

        self.run.define_metric("validation/batch_step")
        self.run.define_metric("validation/loss", step_metric="validation/batch_step")

    def on_train_batch_end(self, logs: dict, idx: int):
        logs["train/batch_step"] = idx
        self.run.log(logs)

    def on_validation_batch_end(self, logs: dict, idx: int):
        logs["validation/batch_step"] = idx
        self.run.log(logs)

    def on_checkpoint_save(self, checkpoint_path: str, epoch: int):
        self.run.log_model(
            path=checkpoint_path,
            name=f"checkpoint_{self.run.id}",
            aliases=[f"epoch-{epoch}", "latest"],
        )

    def on_end(self):
        self.run.finish()
