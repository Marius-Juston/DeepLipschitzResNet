import datetime
import glob
import logging
import os
import pprint
import socket
import time
from contextlib import nullcontext
from os.path import join, exists

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from models.model import LipschitzNetwork
from training import utils
from training.data.readers import readers_config
from training.utils import print_module_parameters


def local_only(func):
    def wrapper(self, *args, **kwargs):
        if self.local_rank == 0:
            return func(self, *args, **kwargs)
        return None

    return wrapper


class Trainer:
    """A Trainer to train a PyTorch."""

    def __init__(self, config):
        self.config = config

    def _load_state(self):
        # load last checkpoint
        checkpoint_directory = join(self.train_dir, 'checkpoints')

        checkpoints = glob.glob(join(checkpoint_directory, 'model.ckpt-*.pth'))

        if len(checkpoints) == 0:
            logging.warning('No checkpoints found in directory {}'.format(checkpoint_directory))
            return

        get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
        checkpoints = sorted(
            [ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)
        path_last_ckpt = join(self.train_dir, 'checkpoints', checkpoints[-1])
        self.checkpoint = torch.load(path_last_ckpt, map_location=self.model.device, weights_only=False)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.saved_ckpts.add(self.checkpoint['epoch'])
        self.start_epoch = self.checkpoint['epoch']
        self.start_global_step = self.checkpoint['global_step']
        self.config = self.checkpoint['config']

        if callable(getattr(self.scheduler, 'state_dict', None)):
            self.scheduler.load_state_dict(self.checkpoint['scheduler'])

        if self.local_rank == 0:
            logging.info('Loading checkpoint {}'.format(checkpoints[-1]))

    def _save_ckpt(self, step, epoch, final=False, best=False):
        """Save ckpt in train directory."""
        freq_ckpt_epochs = self.config.save_checkpoint_epochs
        if ((epoch % freq_ckpt_epochs == 0 and self.is_master and epoch not in self.saved_ckpts)
                or (final and self.is_master) or best):
            prefix = "model" if not best else "best_model"
            ckpt_name = f"{prefix}.ckpt-{step}.pth"
            ckpt_path = join(self.train_dir, 'checkpoints', ckpt_name)
            if exists(ckpt_path) and not best: return
            self.saved_ckpts.add(epoch)

            state = {
                'epoch': epoch,
                'global_step': step,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scheduler': self.scheduler.state_dict(),
                'config': self.config
            }

            if callable(getattr(self.scheduler, 'state_dict', None)):
                state['scheduler'] = self.scheduler.state_dict()

            logging.info("Saving checkpoint '{}'.".format(ckpt_name))
            torch.save(state, ckpt_path)

    @local_only
    def setup_tensorboard(self):
        self.run_name = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        save_folder = os.path.join(self.train_dir, 'tensorboard', f'{self.config.dataset}_{self.run_name}')

        self.summary_writer = SummaryWriter(log_dir=save_folder)

        self.hparams_dict = {

            "n_conv": self.config.n_conv,
            "n_dense": self.config.n_dense,
            "cin": self.config.w,
            "conv_inner_dim": self.config.conv_inner_dim,
            "dense_inner_dim": self.config.dense_inner_dim,
            "n_classes": self.reader.n_classes,
        }

    @local_only
    def print_module_parameters(self, model):
        print_module_parameters(model)

    def __call__(self):
        """Performs training and evaluation
        """
        cudnn.benchmark = True

        self.train_dir = self.config.train_dir

        self.ngpus = 1

        self.start_approx = 10
        self.end_approx = 40

        # job_env = submitit.JobEnvironment()
        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.num_nodes = int(os.environ['LOCAL_WORLD_SIZE'])
        self.num_tasks = int(os.environ['WORLD_SIZE'])
        self.is_master = bool(self.rank == 0)

        # Setup logging
        utils.setup_logging(self.config, self.rank)

        logging.info(self.rank)
        logging.info(self.local_rank)
        logging.info(self.num_nodes)
        logging.info(self.num_tasks)

        self.message = utils.MessageBuilder()
        # print self.config parameters
        if self.local_rank == 0:
            logging.info(self.config.cmd)
            pp = pprint.PrettyPrinter(indent=2, compact=True)
            logging.info(pp.pformat(vars(self.config)))
        # print infos
        if self.local_rank == 0:
            logging.info(f"PyTorch version: {torch.__version__}.")
            logging.info(f"NCCL Version {torch.cuda.nccl.version()}")
            logging.info(f"Hostname: {socket.gethostname()}.")

        # ditributed settings
        self.world_size = 1
        self.is_distributed = False
        if self.num_nodes > 1 or self.num_tasks > 1:
            self.is_distributed = True
            self.world_size = self.num_nodes * self.ngpus
        if self.num_nodes > 1:
            logging.info(
                f"Distributed Training on {self.num_nodes} nodes")
        elif self.num_nodes == 1 and self.num_tasks > 1:
            logging.info(f"Single node Distributed Training with {self.num_tasks} tasks")
        else:
            assert self.num_nodes == 1 and self.num_tasks == 1
            logging.info("Single node training.")

        if not self.is_distributed:
            self.batch_size = self.config.batch_size * self.ngpus
        else:
            self.batch_size = self.config.batch_size

        if self.is_distributed:
            utils.setup_distributed_training()

        self.global_batch_size = self.batch_size * self.world_size
        logging.info('World Size={} => Total batch size {}'.format(
            self.world_size, self.global_batch_size))

        torch.cuda.set_device(self.local_rank)

        # load dataset
        Reader = readers_config[self.config.dataset]
        self.reader = Reader(self.config, self.batch_size, self.is_distributed, is_training=True)

        # ###############
        self.validation_reader = Reader(self.config, self.batch_size, self.is_distributed, is_training=False)

        self.validator_data_loader, self.validator_sampler = self.validation_reader.load_dataset()
        if self.is_distributed:
            n_validation_files = self.validator_sampler.num_samples
        else:
            n_validation_files = self.validation_reader.n_test_files
        # ###############

        if self.local_rank == 0:
            logging.info(f"Using dataset: {self.config.dataset}")

        self.setup_tensorboard()

        # load model
        self.model = LipschitzNetwork(self.config, self.reader.n_classes, device=f'cuda:{self.local_rank}')

        self.print_module_parameters(self.model)
        nb_parameters = np.sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        logging.info(f'Number of parameters to train: {nb_parameters}')

        if self.config.compile:
            self.model = torch.compile(self.model, mode='max-autotune')

        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        # define set for saved ckpt
        self.saved_ckpts = {0}

        self.data_loader, self.sampler = self.reader.load_dataset()
        if self.sampler is not None:
            assert self.sampler.num_replicas == self.world_size

        if self.is_distributed:
            n_files = self.sampler.num_samples
        else:
            n_files = self.reader.n_train_files

        # define optimizer
        self.optimizer = utils.get_optimizer(self.config, self.model.parameters())

        # define learning rate scheduler
        num_steps = self.config.epochs * (self.reader.n_train_files // self.global_batch_size)
        self.scheduler, self.warmup = utils.get_scheduler(self.optimizer, self.config, num_steps)
        if self.config.warmup_scheduler is not None:
            logging.info(f"Warmup scheduler on {self.config.warmup_scheduler * 100:.0f}% of training")

        # define the loss
        self.criterion = utils.get_loss(self.config)

        if self.local_rank == 0:
            logging.info("Number of train files on worker: {}".format(n_files))
            logging.info("Number of test files on worker: {}".format(n_validation_files))
            logging.info("Start training")

        self.start_epoch = 0
        self.start_global_step = 0

    def load(self):
        checkpoints_path = join(self.train_dir, 'checkpoints')

        if not os.path.exists(checkpoints_path):
            logging.warning(f'The checkpoint path {checkpoints_path} does not exist. Not loading model')
        else:
            logging.warning(f'Loading from path {checkpoints_path}.')
            self._load_state()

    def train(self):
        # training loop
        start_epoch, global_step = self.start_epoch, self.start_global_step
        self.best_checkpoint = None
        self.best_accuracy = 0
        for epoch_id in range(start_epoch, self.config.epochs):
            if self.is_distributed:
                self.sampler.set_epoch(epoch_id)
            for n_batch, data in enumerate(self.data_loader):
                if global_step == self.start_approx and self.is_master:
                    start_time = time.time()
                epoch = (int(global_step) * self.global_batch_size) / self.reader.n_train_files
                self.one_step_training(data, epoch, global_step)
                self._save_ckpt(global_step, epoch_id)
                if global_step == self.end_approx and self.is_master:
                    self._print_approximated_train_time(start_time)
                global_step += 1
            if epoch_id % self.config.validate_every == 0:
                self.validate(self.validator_data_loader, self.validator_sampler, epoch_id)

        self._save_ckpt(global_step, epoch_id, final=True)
        logging.info("Done training -- epoch limit reached.")

    @torch.no_grad()
    def validate(self, dataloader, sampler, epoch):
        if self.is_distributed:
            sampler.set_epoch(epoch)

        logging.info("Starting validation")

        self.model.eval()

        accuracy = self.eval_certified_run(dataloader, epoch)

        logging.info("Finished validation")

        if self.local_rank == 0:
            if accuracy > self.best_accuracy:
                logging.info(f"New best validation accuracy: {accuracy}")
                self.best_accuracy = accuracy

                self.summary_writer.add_hparams(self.hparams_dict, {
                    'hparam/accuracy': accuracy
                }, run_name=self.run_name, global_step=epoch)

        self.model.train()

        return accuracy

    @torch.no_grad()
    def eval_certified_run(self, dataloader, epoch):
        running_accuracy = 0
        running_inputs = 0
        for batch_n, data in enumerate(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(self.local_rank), labels.to(self.local_rank)
            outputs = self.model(inputs)
            predicted = outputs.argmax(axis=1)
            running_accuracy += predicted.eq(labels.data).sum().item()
            running_inputs += inputs.size(0)
        accuracy = running_accuracy / running_inputs

        if self.local_rank == 0:
            self.message.add('epoch', epoch, format='.0f')
            self.message.add('accuracy', accuracy, format='.5f')
            logging.info(self.message.get_message())

            self.summary_writer.add_scalar(f'Validation/Accuracy', accuracy, global_step=epoch)

        return accuracy

    def compute_gradient_norm(self):
        grad_norm = 0.
        for name, p in self.model.named_parameters():
            if p.grad is None: continue
            norm = p.grad.detach().data.norm(2)
            grad_norm += norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        return grad_norm

    def _print_approximated_train_time(self, start_time):
        total_steps = self.reader.n_train_files * self.config.epochs / self.global_batch_size
        total_seconds = total_steps * ((time.time() - start_time) / (self.end_approx - self.start_approx))
        n_days = total_seconds // 86400
        n_hours = (total_seconds % 86400) / 3600
        logging.info(
            'Approximated training time: {:.0f} days and {:.1f} hours'.format(
                n_days, n_hours))

    def _to_print(self, step):
        frequency = self.config.frequency_log_steps
        if frequency is None:
            return False
        return (step % frequency == 0 and self.local_rank == 0) or \
            (step == 1 and self.local_rank == 0)

    def process_gradients(self, step):
        if self.config.gradient_clip_by_norm:
            if step == 0 and self.local_rank == 0:
                logging.info("Clipping Gradient by norm: {}".format(
                    self.config.gradient_clip_by_norm))
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_by_norm)
        elif self.config.gradient_clip_by_value:
            if step == 0 and self.local_rank == 0:
                logging.info("Clipping Gradient by value: {}".format(
                    self.config.gradient_clip_by_value))
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.config.gradient_clip_by_value)

    def one_step_training(self, data, epoch, step):

        self.optimizer.zero_grad(set_to_none=True)

        batch_start_time = time.time()
        images, labels = data
        images, labels = images.to(self.local_rank), labels.to(self.local_rank)

        outputs = self.model(images)
        loss = self.criterion(outputs, labels)

        if not torch.isfinite(loss).all():
            logging.info("Loss is no longer finite, check problem")
            raise RuntimeError('Loss is infinite', loss)

        loss.backward()
        self.process_gradients(step)
        self.optimizer.step()

        if self.scheduler:
            with self.warmup.dampening() if self.warmup else nullcontext():
                self.scheduler.step(step)

        seconds_per_batch = time.time() - batch_start_time
        examples_per_second = self.batch_size / seconds_per_batch
        examples_per_second *= self.world_size

        if step == 0 and self.local_rank == 0:
            logging.info(f'images {images.shape}')
            logging.info(f'labels {labels.shape}')
            logging.info(f'outputs {outputs.shape}')

        if self._to_print(step):
            lr = self.optimizer.param_groups[0]['lr']
            self.message.add("epoch", epoch, format="4.2f")
            self.message.add("step", step, width=5, format=".0f")
            self.message.add("lr", lr, format=".6f")
            self.message.add("loss", loss, format=".4f")
            if self.config.print_grad_norm:
                grad_norm = self.compute_gradient_norm()
                self.message.add("grad", grad_norm, format=".4f")
                self.summary_writer.add_scalar('Training/Grad Norm', grad_norm)
            self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")
            logging.info(self.message.get_message())

            self.summary_writer.add_scalar('Training/LR', lr, global_step=step)
            self.summary_writer.add_scalar('Training/Loss', loss, global_step=step)
            self.summary_writer.add_scalar('Training/FPS', examples_per_second, global_step=step)
