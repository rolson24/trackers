from typing import Any, Optional

from torch.utils.tensorboard import SummaryWriter


class BaseCallback:
    def on_train_step_end(self, logs: dict, idx: int):
        pass

    def on_validation_step_end(self, logs: dict, idx: int):
        pass

    def on_train_val_end(self):
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

    def on_train_step_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, idx)

    def on_validation_step_end(self, logs: dict, idx: int):
        for key, value in logs.items():
            self.writer.add_scalar(key, value, idx)

    def on_train_val_end(self):
        self.writer.flush()
        self.writer.close()
