import os
import logging
import torch
from torch.utils.tensorboard import SummaryWriter


class TensorboardLogger:
    """
    Logging losses to tensorboard
    -Train loss
    -Validation loss
    -Learning rate
    can be logged and visualized in tensorboard
    """

    def __init__(self, log_dir):
        self.writer = SummaryWriter(f"{log_dir}/loss_logs")

    def log_train_loss(self, value, idx):
        """logging train loss"""
        self.writer.add_scalar("train_loss", value, idx)

    def log_val_loss(self, value, idx):
        """logging validation loss"""
        self.writer.add_scalar("val_loss", value, idx)

    def log_lr(self, value, idx):
        """logging learning rate"""
        self.writer.add_scalar("lr", value, idx)


def configure_logger(log_dir, name=__name__):
    """
    Configure the logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("%(levelname)s:%(asctime)s:%(name)s:%(message)s")

    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(f"{log_dir}/logs.log")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream handler (console)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


############ Save and Load models ############
def save_ckp(
    state,
    out_dir: str,
    is_best: bool = False,
    is_last: bool = False,
    special_prefix: str = "",
) -> None:
    """
    saves model checkpoints to outdir
    Use special_prefix to save special checkpoints

    Args:
    state --> This is the state dict (see example)
    is_best --> save the best model
    is_last --> save the last model (can be used to save the model at each epoch)
    special_prefix --> This allows to add any special tags to the saved model

    Example:
    checkpoint = {
    'epoch': epoch + 1,
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict()
    'best_loss': best_loss
    }
    save_ckp(checkpoint, is_best, checkpoint_dir, model_dir)
    """
    if not os.path.exists(os.path.join(out_dir, "ckpts")):
        os.makedirs(os.path.join(out_dir, "ckpts"))

    best_path = f"{out_dir}/ckpts/{special_prefix}_best.pt"
    last_path = f"{out_dir}/ckpts/{special_prefix}_last.pt"

    if is_best:
        torch.save(state, best_path)
    if is_last:
        torch.save(state, last_path)


def load_ckp(checkpoint_fpath: str, model, optimizer):
    """
    Loades saved models from the input path

    Args:
    checkpoint_fpath --> this should be the path to the desired checkpoint.pt file
    model --> Model to load weights to
    optimizer --> optimizer to load optimizer state to

    Example:
    model = MyModel(*args, **kwargs)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    ckp_path = "path/to/checkpoint/checkpoint.pt"
    model, optimizer, start_epoch, best_loss = load_ckp(ckp_path, model, optimizer)
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return model, optimizer, checkpoint["epoch"], checkpoint["best_loss"]
