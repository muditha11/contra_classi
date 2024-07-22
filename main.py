import os
import argparse
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from omegaconf import OmegaConf, DictConfig

from src.utils import configure_logger, load_ckp, load_module, TensorboardLogger
from src.trainer import Trainer
from src.evaluator import Evaluator


class Pipeline:
    """
    This the full pipeline
    Can include training, testing and evaluation on downstream tasks
    """

    def _init_exp(self) -> None:
        """
        ---Do not change---
        Create the folder structure
        If resuming training folders are not created
        """
        self.exp_name = os.path.join(self.conf.out_dir, self.conf.name)
        self.run = 0
        while os.path.exists(os.path.join(self.exp_name, f"run{self.run}")):
            self.run += 1
        if self.conf.resume:
            self.run = self.conf.run
        self.out_dir = os.path.join(self.exp_name, f"run{self.run}")
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def _init_logs(self) -> None:
        """
        --- Do not change ---
        Initialize the log directories
        Ex. exp_name -- run0 --recipie    (saves the config)
                             --ckpts      (for model weights)
                             --loss_info  (for loss curves and loss info)
                             --outputs    (test,eval results)

        """

        if self.start_epoch == 0:
            self.out_dir = os.path.join(self.exp_name, f"run{self.run}")
            os.makedirs(os.path.join(self.out_dir, "ckpts"))
            os.makedirs(os.path.join(self.out_dir, "outputs"))
            OmegaConf.save(self.conf, f"{self.out_dir}/recipie.yaml")

    def _init_visualizer(self) -> None:
        self.visualizer = TensorboardLogger(self.out_dir)

    def _set_data(self):
        """
        #####################################################
        Create required dataloaders

        Args:
        mock_batch_count --> given as a command line argument

        Sets:
        train_dl
        val_dl --> if val in present in the config file

        #####################################################
        Usage tips:
        Recheck inputs for Dataset class
        """

        # Recheck inputs
        dataset_class = load_module(self.conf.data.target)
        self.train_ds = dataset_class(
            self.conf.data.data_dir, self.conf.data.splits.train
        )
        if self.mock_batch_count != -1:
            self.train_ds = Subset(
                self.train_ds,
                list(
                    range(
                        self.conf.train.loader_params.batch_size * self.mock_batch_count
                    )
                ),
            )

        self.train_dl = DataLoader(self.train_ds, **dict(self.conf.train.loader_params))

        self.val_dl = None
        if "val" in self.conf:
            # Recheck inputs
            self.val_ds = dataset_class(
                self.conf.data.data_dir, self.conf.data.splits.val
            )
            if self.mock_batch_count != -1:
                self.val_ds = Subset(
                    self.val_ds,
                    list(
                        range(
                            self.conf.val.loader_params.batch_size
                            * self.mock_batch_count
                        )
                    ),
                )
            self.val_dl = DataLoader(self.val_ds, **dict(self.conf.val.loader_params))

    def _get_model(self):
        """
        #####################################################
        Create model
        #####################################################
        Usage tips:
        Recheck inputs to the Model class
        Also configure the freezing setup as needed
        """
        # Recheck inputs
        model_class = load_module(self.conf.model.target)
        self.model = model_class(device=self.device, **dict(self.conf.model.params))

        for param in self.model.parameters():
            param.requires_grad = not (self.conf.model.freeze.encoder)

    def _get_train_objs(self):
        """
        Creating --> Loss criterion
                 --> Optimizer
                 --> Schedular
        """
        ## Loss
        loss_class = load_module(self.conf.loss.target)
        self.criterion = loss_class(self.device)

        ## Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), **dict(self.conf.optimizer)
        )

        ## Schedular
        self.schedular = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, **dict(self.conf.lr_schedular.params)
        )
        if self.conf.lr_schedular.warmup > 0:
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                self.optimizer, total_iters=self.conf.lr_schedular.warmup
            )
            self.schedular = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_lr_scheduler, self.schedular],
                milestones=[self.conf.lr_schedular.warmup],
            )

    def check_resume(self):
        """
        Check if resuming the pipeline
        """
        if self.conf.resume:
            self.run = self.conf.run
            ckp_path = os.path.join(
                self.exp_name,
                f"run{self.run}",
                f"ckpts/{self.conf.model.params.base_model}_best.pt",
            )
            self.model, self.optimizer, self.start_epoch, self.best_loss = load_ckp(
                ckp_path, self.model, self.optimizer
            )
            self.logger.info(
                f"Resuming Training from epoch: {self.start_epoch}.........."
            )

        else:
            self.start_epoch = 0
            self.best_loss = torch.inf
            self.logger.info("Starting new training................................")

    def __init__(self, conf: DictConfig, device: int, mock_batch_count: int):
        self.device = device
        self.conf = conf
        self.mock_batch_count = mock_batch_count

        ## Create the folder for the experiment ##
        self._init_exp()

        ## Setup the logger ##
        self.logger = configure_logger(self.out_dir, __name__)
        self.logger.info(
            f"Strating Training job for experiment:{self.conf.name} {self.run}"
        )

        ## Data setup ##
        self._set_data()
        self.logger.info(f"Dataloaders set for the experiment")

        ## Model setup ##
        self._get_model()
        self.logger.info(f"Model set for the experiment")
        self.logger.warning(
            f"Model weights are frozen:{self.conf.model.freeze.encoder}"
        )

        ## Training Objects setup ##
        self._get_train_objs()
        self.logger.info(f"Train objects set for the experiment")

        ## Check if the experiment is a resume of a previous experiment
        self.check_resume()

        ## Logger setup ##
        self._init_logs()
        self._init_visualizer()

        self.trainer = Trainer(self.out_dir, self.visualizer)
        self.evaluator = Evaluator(self.out_dir, device, self.conf)

    def fit(self):
        """fit method"""

        self.trainer.train(
            self.train_dl,
            self.val_dl,
            self.model,
            self.criterion,
            self.optimizer,
            self.schedular,
            self.start_epoch,
            self.best_loss,
            self.conf,
            self.evaluator,
        )

        ## Test tasks ##
        #   self.test()

        ## Downstream tasks ##
        #   self.evaluate()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Getting hyperparameters")

    parser.add_argument("--config", type=str, help="path of the configuration file")

    parser.add_argument("--device", type=str, help="path of the configuration file")

    parser.add_argument(
        "--mb", default=-1, type=int, help="Mock batch count for testing pipeline"
    )

    args = parser.parse_args()

    with open(args.config) as handler:
        conf = OmegaConf.create(yaml.load(handler, yaml.FullLoader))

    DEVICE = eval(args.device)
    mock_batch_count = args.mb
    pipe = Pipeline(conf, DEVICE, mock_batch_count)
    pipe.fit()
