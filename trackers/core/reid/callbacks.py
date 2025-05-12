from typing import Any, Optional

from torch.utils.tensorboard import SummaryWriter

import wandb


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


class WandbCallback(BaseCallback):
    def __init__(self, config: dict[str, Any]) -> None:
        self.run = wandb.init(config=config) if not wandb.run else wandb.run  # type: ignore

    def on_train_batch_end(self, logs: dict, idx: int):
        self.run.log(logs, step=idx)

    def on_validation_batch_end(self, logs: dict, idx: int):
        self.run.log(logs, step=idx)

    def on_checkpoint_save(self, checkpoint_path: str, epoch: int):
        self.run.log_model(
            path=checkpoint_path,
            name=f"checkpoint_{self.run.id}",
            aliases=[f"epoch-{epoch}", "latest"],
        )

    def on_end(self):
        self.run.finish()
