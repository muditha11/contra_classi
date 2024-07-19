from tqdm import tqdm
from .utils import configure_logger, save_ckp


class Trainer:
    """
    Trainer class definition
    Contains Train and validation loops.
    In addition Evaluator can also be passed in
    """

    def __init__(self, out_dir: str, visualizer):
        self.logger = configure_logger(out_dir, __name__)
        self.logger.info("Initializing trainer")
        self.out_dir = out_dir
        self.visualizer = visualizer

    def train(
        self,
        train_dl,
        val_dl,
        model,
        criterion,
        optimizer,
        scheduler,
        start_epoch,
        best_loss,
        conf,
        evaluator,
    ):
        """Runs the training and validation loops"""

        num_epochs = conf.train.epochs

        for epoch in range(start_epoch, num_epochs):
            ## Train step ##
            self.logger.info(f"Starting epoch {epoch}............")
            model.train()
            cum_loss = 0
            for ind, batch in tqdm(
                enumerate(train_dl), desc="Train", total=len(train_dl)
            ):
                optimizer.zero_grad()
                info = model(batch)
                loss = criterion(info, batch)
                loss.backward()
                optimizer.step()
                cum_loss += loss.detach().cpu()
                self.visualizer.log_train_loss(
                    loss.detach().cpu().item(), epoch * len(train_dl) + ind
                )
            scheduler.step()
            self.visualizer.log_lr(optimizer.param_groups[0]["lr"], epoch)

            ## Validation Step ##
            model.eval()
            val_cum_loss = 0
            for ind, batch in tqdm(enumerate(val_dl), desc="Val", total=len(val_dl)):
                info = model(batch)
                loss = criterion(info, batch)
                val_cum_loss += loss.detach().cpu()
                self.visualizer.log_val_loss(
                    loss.detach().cpu().item(), epoch * len(val_dl) + ind
                )

            self.logger.info(
                f"Epoch:{epoch} | train_loss:{cum_loss/len(train_dl)} | val_loss:{val_cum_loss/len(val_dl)}"
            )
            self.logger.info(f"Learning rate:{optimizer.param_groups[0]['lr']}")

            ## Saving best models ##
            if val_cum_loss / len(val_dl) <= best_loss:
                best_loss = val_cum_loss / len(val_dl)

                checkpoint = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_loss": best_loss,
                }
                save_ckp(
                    checkpoint,
                    self.out_dir,
                    is_best=True,
                    specialPrefix=conf.model.params.base_model,
                )
                model.save_encoder(f"{self.out_dir}/ckpts")
                self.logger.info("Saving best model...........................")

                evaluator(model.encoder, conf, epoch)

        self.logger.info("Training Finished...........")
