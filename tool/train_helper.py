import collections
import math
import statistics
from datetime import datetime
from enum import Enum
from itertools import repeat

import torch
from ray import tune
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data_process.data_reader.data_enum import DataTypeEnum
from model.base_model import BaseModel
from tool.log_helper import logger, add_log_file, remove_log_file, remove_log_console
from tool.model_helper import save_model
from tool.path_helper import ROOT_DIR

REPORT_STEPS = 10
CHECKPOINT_STEPS = 100
EARLY_STOP_STEPS = 10000


class TrainModeEnum(Enum):
    Single = "Single"  # Train single model at once
    Multi = "Multi"  # Train multi models with ray tune
    Tune = "Tune"  # Tune hyper-parm with ray tune


class TrainHelper:
    def __init__(self, model: BaseModel, train_tag: str, train_mode: TrainModeEnum):

        self.model = model
        self.config = model.config
        self.train_tag = train_tag
        self.train_mode = train_mode

        # init in prepare_training()
        self.model_save_name = None
        self.train_dataset = None
        self.dev_dataset = None
        self.train_dataloader = None
        self.dev_dataloader = None
        self.last_save_step = None
        self.repeat_train_dataloader = None
        self.history_train_losses = None
        self.min_dev_loss = None
        self.tensorboard_writer_train = None
        self.tensorboard_writer_dev = None

        # init in train_model()
        self.pbar = None

    @property
    def enable_console_log(self):
        return self.train_mode == TrainModeEnum.Single

    @property
    def is_tune_hyper(self):
        return self.train_mode == TrainModeEnum.Tune

    def train_model(self):
        self.prepare_training()

        opt = torch.optim.Adam(
            self.model.parameters(),
            self.config.learning_rate,
            weight_decay=self.config.l2_regularization)

        lr_s = lr_scheduler.StepLR(
            opt, step_size=CHECKPOINT_STEPS,
            gamma=self.config.learning_rate_decay)

        self.pbar = self.create_training_bar()
        while self.model.current_training_step < self.config.steps_num:
            self.model.current_training_step += 1

            # train one step
            self.model.train()
            opt.zero_grad()
            batch = next(self.repeat_train_dataloader)
            predict, actual, loss = self.model.predict_iter_i(batch)
            self.history_train_losses.append(loss.item())
            loss.backward()
            opt.step()
            lr_s.step()

            self.history_train_losses.append(loss.item())
            self.pbar.update()

            if not torch.isnan(loss):
                if self.model.current_training_step % REPORT_STEPS == 0:
                    train_loss = statistics.mean(self.history_train_losses)
                    logger.debug(f"Step {self.model.current_training_step} "
                                 f"train loss = {train_loss :5f}")
                    self.tensorboard_writer_train.add_scalar('Loss', train_loss, self.model.current_training_step)

                if self.model.current_training_step % CHECKPOINT_STEPS == 0:
                    early_stop = self.checkpoint()
                    if early_stop:
                        break
            else:
                logger.error(f"Loss is nan! Stop train!")
                break

        self.finish_training()

    def prepare_training(self):
        self.model_save_name = f"{self.model.get_name()}_{datetime.now().strftime('%Y%m%d%H%M%S%f')}"
        if self.train_tag is not None and not self.train_tag.isspace():
            insert_idx = len(self.model.get_name())
            self.model_save_name = self.model_save_name[:insert_idx] + "_" + self.train_tag + self.model_save_name[insert_idx:]

        # Add log file.
        log_path = f"{'tune' if self.is_tune_hyper else 'train'}/{self.model_save_name}"
        add_log_file(logger, f"{log_path}/main.log")

        # tensorboard writer
        self.tensorboard_writer_train = SummaryWriter(ROOT_DIR.joinpath(f"out/log/{log_path}/train"))
        self.tensorboard_writer_dev = SummaryWriter(ROOT_DIR.joinpath(f"out/log/{log_path}/dev"))

        # Don't show console output when use ray[tune]
        if not self.enable_console_log:
            remove_log_console(logger)

        pin_memory = self.config.device not in ["cpu", "CPU"]
        self.train_dataset = self.model.create_dataset(DataTypeEnum.Train)
        self.dev_dataset = self.model.create_dataset(DataTypeEnum.Dev)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=pin_memory)
        self.dev_dataloader = DataLoader(self.dev_dataset, batch_size=self.config.batch_size * 3, shuffle=True, pin_memory=pin_memory)

        # early stop only after one epoch.
        self.last_save_step = len(self.train_dataloader)
        self.repeat_train_dataloader = self.wrap_repeat_dataloader()
        self.history_train_losses = collections.deque([], maxlen=int(len(self.dev_dataset) / self.config.batch_size))
        self.min_dev_loss = float("inf")

        logger.info(f"Training {self.model_save_name}...")
        logger.info(self.config.__dict__)
        logger.info(f"Total parameter numbers = {self.model.get_param_number() / 1000000:.2f}M")
        logger.info(f"Length of epoch = {len(self.train_dataloader)} batches")
        if self.config.steps_num < len(self.train_dataloader):
            self.config.steps_num = math.ceil(len(self.train_dataloader) / CHECKPOINT_STEPS) * CHECKPOINT_STEPS
            logger.warning(f"Step nums is smaller than train data, set it to {self.config.steps_num}")

    def checkpoint(self) -> bool:
        logger.debug("Check Point reached, evaluating model on dev dataset...")

        # close train pbar for eval pbar in eval_model()
        self.pbar.close()
        dev_loss, dev_metric = self.eval_model()
        self.pbar = self.create_training_bar()

        train_loss = statistics.mean(self.history_train_losses)
        logger.info(f"Step {self.model.current_training_step} "
                    f"loss(train/dev) = {train_loss:5f}/{dev_loss:5f} "
                    f"Dev metrics = {dev_metric}")
        self.tensorboard_writer_dev.add_scalar('Loss', dev_loss, self.model.current_training_step)
        self.tensorboard_writer_dev.add_scalar('Metric', dev_metric, self.model.current_training_step)

        # save best model
        if dev_loss <= self.min_dev_loss:
            self.min_dev_loss = dev_loss
            self.last_save_step = max(self.last_save_step, self.model.current_training_step)
            if not self.is_tune_hyper:
                save_model(self.model, self.model_save_name)

        if not self.enable_console_log:
            tune.report(train_loss=train_loss, dev_loss=dev_loss, dev_metric=dev_metric)

        if self.model.current_training_step - self.last_save_step >= EARLY_STOP_STEPS:
            logger.info(f"Early stop due to no performance promotion within {EARLY_STOP_STEPS} checkpoints")
            return True
        else:
            return False

    def eval_model(self):
        self.model.eval()
        predicts, actuals, all_loss = [], [], []
        with torch.no_grad():
            for batch in tqdm(self.dev_dataloader, desc=f"Evaluating {self.model.get_name()}",
                              unit="bt", ncols=100, disable=not self.enable_console_log, leave=False):
                predict, actual, loss = self.model.predict_iter_i(batch)
                predicts.append(predict)
                actuals.append(actual)
                all_loss.append(loss.item())

            loss_f = torch.nn.MSELoss()
            predicts = torch.cat(predicts)
            actuals = torch.cat(actuals)
            metric = loss_f(predicts, actuals).item()
        return statistics.mean(all_loss), metric

    def finish_training(self):
        self.pbar.leave = True
        self.pbar.close()
        logger.info(f"Min dev loss = {self.min_dev_loss:.5f}")
        logger.info("%s trained!" % self.model_save_name)
        remove_log_file(logger)

    def create_training_bar(self):
        pbar = tqdm(initial=self.model.current_training_step, total=self.model.config.steps_num,
                    desc=f"Training {self.model.get_name()}", unit="bt",
                    ncols=100, disable=not self.enable_console_log, leave=False)
        if self.pbar is not None and self.enable_console_log:
            pbar.start_t = self.pbar.start_t
            pbar.last_print_t = self.pbar.last_print_t

        return pbar

    def wrap_repeat_dataloader(self):
        for loader in repeat(self.train_dataloader):
            for data in loader:
                yield data
