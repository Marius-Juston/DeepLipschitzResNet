import random

import torch

from training.utils import print_module_parameters

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

import argparse
import sys
import warnings

warnings.filterwarnings("ignore")

from os.path import realpath
import glob
import logging
import pprint
import socket
import time
from contextlib import nullcontext
from os.path import join, exists

import numpy as np
from torch.distributed.elastic.multiprocessing.errors import record

from models.model import LipschitzNetwork
from training import utils
from training.data.readers import readers_config

import os

import torch
import torch.nn.functional as F


class Evaluator:
    """Evaluate a Pytorch Model."""

    def __init__(self, config):
        self.config = config

    def load_ckpt(self, ckpt_path=None):
        if ckpt_path is None:
            checkpoints = glob.glob(join(self.config.train_dir, "checkpoints", "model.ckpt-*.pth"))
            get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
            ckpt_name = sorted([ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)[-1]
            ckpt_path = join(self.config.train_dir, "checkpoints", ckpt_name)
        checkpoint = torch.load(ckpt_path)
        new_checkpoint = {}
        for k, v in checkpoint['model_state_dict'].items():
            if 'alpha' not in k:
                new_checkpoint[k] = v
        self.model.load_state_dict(new_checkpoint)
        epoch = checkpoint['epoch']
        return epoch

    def __call__(self):
        """Run evaluation of model or eval under attack"""

        # create a mesage builder for logging
        self.message = utils.MessageBuilder()
        # Setup logging & log the version.
        utils.setup_logging(self.config, 0)

        self.batch_size = self.config.batch_size

        # load reader
        Reader = readers_config[self.config.dataset]
        self.reader = Reader(self.config, self.batch_size, False, is_training=False)
        self.config.means = self.reader.means

        # load model
        self.model = LipschitzNetwork(self.config, self.reader.n_classes, device=self.config.device)

        self.load_ckpt()
        if self.config.mode == "certified":
            for eps in [36, 72, 108, 255]:
                self.eval_certified(eps)
        elif self.config.mode == "attack":
            self.eval_attack()

        logging.info("Done with batched inference.")

    @torch.no_grad()
    def eval_certified(self, eps):
        eps_float = eps / 255
        self.model.eval()
        running_accuracy = 0
        running_certified = 0
        running_inputs = 0
        lip_cst = 1.
        data_loader, _ = self.reader.load_dataset()
        last_weight = self.model.module.model.model[-1].weight
        normalized_weight = F.normalize(last_weight, p=2, dim=1)
        for batch_n, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            outputs = self.model(inputs)
            predicted = outputs.argmax(axis=1)
            correct = outputs.max(1)[1] == labels
            margins, indices = torch.sort(outputs, 1)
            margins = margins[:, -1][:, None] - margins[:, 0:-1]
            for idx in range(margins.shape[0]):
                margins[idx] /= torch.norm(
                    normalized_weight[indices[idx, -1]] - normalized_weight[indices[idx, 0:-1]], dim=1, p=2)
            margins, _ = torch.sort(margins, 1)
            certified = margins[:, 0] > eps_float * lip_cst
            running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
            running_certified += torch.sum(correct & certified).item()
            running_inputs += inputs.size(0)
        accuracy = running_accuracy / running_inputs
        certified = running_certified / running_inputs
        self.message.add('eps', [eps, 255], format='.0f')
        self.message.add('eps', eps_float, format='.5f')
        self.message.add('accuracy', accuracy, format='.5f')
        self.message.add('certified accuracy', certified, format='.5f')
        logging.info(self.message.get_message())
        return accuracy, certified

    def eval_attack(self):
        """Run evaluation under attack."""

        attack = utils.get_attack_eval(
            self.model,
            self.config.attack,
            self.config.eps / 255,
            self.batch_size)

        running_accuracy = 0
        running_accuracy_adv = 0
        running_inputs = 0
        data_loader, _ = self.reader.load_dataset()
        for batch_n, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()
            inputs_adv = attack.perturb(inputs, labels)

            outputs = self.model(inputs)
            outputs_adv = self.model(inputs_adv)
            _, predicted = torch.max(outputs.data, 1)
            _, predicted_adv = torch.max(outputs_adv.data, 1)

            running_accuracy += predicted.eq(labels.data).cpu().sum().numpy()
            running_accuracy_adv += predicted_adv.eq(labels.data).cpu().sum().numpy()
            running_inputs += inputs.size(0)

        accuracy = running_accuracy / running_inputs
        accuracy_adv = running_accuracy_adv / running_inputs
        self.message.add(f'attack: {self.config.attack} - eps', self.config.eps, format='.0f')
        self.message.add('Accuracy', accuracy, format='.5f')
        self.message.add('Accuracy attack', accuracy_adv, format='.5f')
        logging.info(self.message.get_message())


class Trainer:
    """A Trainer to train a PyTorch."""

    def __init__(self, config):
        self.config = config

    def _load_state(self):
        # load last checkpoint
        checkpoints = glob.glob(join(self.train_dir, 'checkpoints', 'model.ckpt-*.pth'))
        get_model_id = lambda x: int(x.strip('.pth').strip('model.ckpt-'))
        checkpoints = sorted(
            [ckpt.split('/')[-1] for ckpt in checkpoints], key=get_model_id)
        path_last_ckpt = join(self.train_dir, 'checkpoints', checkpoints[-1])
        self.checkpoint = torch.load(path_last_ckpt, map_location=self.model.device)
        self.model.load_state_dict(self.checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.saved_ckpts.add(self.checkpoint['epoch'])
        epoch = self.checkpoint['epoch']
        logging.info('Loading checkpoint {}'.format(checkpoints[-1]))

    def _save_ckpt(self, step, epoch, final=False, best=False):
        """Save ckpt in train directory."""
        freq_ckpt_epochs = self.config.save_checkpoint_epochs
        if ((epoch % freq_ckpt_epochs == 0 and epoch not in self.saved_ckpts)
                or final or best):
            prefix = "model" if not best else "best_model"
            ckpt_name = f"{prefix}.ckpt-{step}.pth"
            ckpt_path = join(self.train_dir, 'checkpoints', ckpt_name)
            if exists(ckpt_path) and not best: return
            self.saved_ckpts.add(epoch)
            state = {
                'epoch': epoch,
                'global_step': step,
                'model_state_dict': self.model._orig_mod.state_dict() if self.config.compile else self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                # 'scheduler': self.scheduler.state_dict()
            }
            logging.debug("Saving checkpoint '{}'.".format(ckpt_name))
            torch.save(state, ckpt_path)

    @record
    def __call__(self):
        """Performs training and evaluation
        """
        self.train_dir = self.config.train_dir

        self.start_approx = 10
        self.end_approx = 40

        # Setup logging
        utils.setup_logging(self.config, 0)

        self.message = utils.MessageBuilder()
        # print self.config parameters
        logging.info(self.config.cmd)
        pp = pprint.PrettyPrinter(indent=2, compact=True)
        logging.info(pp.pformat(vars(self.config)))
        # print infos
        logging.info(f"PyTorch version: {torch.__version__}.")
        logging.info(f"NCCL Version {torch.cuda.nccl.version()}")
        logging.info(f"Hostname: {socket.gethostname()}.")

        # ditributed settings
        self.batch_size = self.config.batch_size

        logging.info('Total size {}'.format(self.batch_size))

        # load dataset
        Reader = readers_config[self.config.dataset]
        self.reader = Reader(self.config, self.batch_size, False, is_training=True)
        logging.info(f"Using dataset: {self.config.dataset}")

        # load model
        self.model = LipschitzNetwork(self.config, self.reader.n_classes, device=self.config.device)

        print_module_parameters(self.model)

        if self.config.compile:
            self.model = torch.compile(self.model, mode='max-autotune', dynamic=True)

        nb_parameters = np.sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        logging.info(f'Number of parameters to train: {nb_parameters}')

        # setup distributed process if training is distributed
        # and use DistributedDataParallel for distributed training
        # define set for saved ckpt
        self.saved_ckpts = {0}

        data_loader, sampler = self.reader.load_dataset()
        n_files = self.reader.n_train_files

        # define optimizer
        self.optimizer = utils.get_optimizer(self.config, self.model.parameters())

        # define learning rate scheduler
        num_steps = self.config.epochs * (self.reader.n_train_files // self.batch_size)
        self.scheduler, self.warmup = utils.get_scheduler(self.optimizer, self.config, num_steps)
        if self.config.warmup_scheduler is not None:
            logging.info(f"Warmup scheduler on {self.config.warmup_scheduler * 100:.0f}% of training")

        # define the loss
        self.criterion = utils.get_loss(self.config)

        logging.info("Number of files on worker: {}".format(n_files))
        logging.info("Start training")

        # training loop
        start_epoch, global_step = 0, 0
        self.best_checkpoint = None
        self.best_accuracy = None
        self.best_accuracy = [0., 0.]
        for epoch_id in range(start_epoch, self.config.epochs):
            for n_batch, data in enumerate(data_loader):
                if global_step == self.start_approx:
                    start_time = time.time()
                epoch = (int(global_step) * self.batch_size) / self.reader.n_train_files
                self.one_step_training(data, epoch, global_step)
                self._save_ckpt(global_step, epoch_id)
                if global_step == self.end_approx:
                    self._print_approximated_train_time(start_time)
                global_step += 1

        self._save_ckpt(global_step, epoch_id, final=True)
        logging.info("Done training -- epoch limit reached.")

    def filter_parameters(self):
        conv_params, linear_params = [], []
        for name, params in self.model.named_parameters():
            if 'weight' in name.lower() and params.dim() == 4:
                conv_params.append(params)
            elif 'weight' in name.lower() and params.dim() == 2:
                linear_params.append(params)
            elif 'bias' in name.lower():
                conv_params.append(params)
        return conv_params, linear_params

    def compute_gradient_norm(self):
        grad_norm = 0.
        for name, p in self.model.named_parameters():
            if p.grad is None: continue
            norm = p.grad.detach().data.norm(2)
            grad_norm += norm.item() ** 2
        grad_norm = grad_norm ** 0.5
        return grad_norm

    def _print_approximated_train_time(self, start_time):
        total_steps = self.reader.n_train_files * self.config.epochs / self.batch_size
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
        return (step % frequency == 0) or (step == 1)

    def process_gradients(self, step):
        if self.config.gradient_clip_by_norm:
            if step == 0:
                logging.info("Clipping Gradient by norm: {}".format(
                    self.config.gradient_clip_by_norm))
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.gradient_clip_by_norm)
        elif self.config.gradient_clip_by_value:
            if step == 0:
                logging.info("Clipping Gradient by value: {}".format(
                    self.config.gradient_clip_by_value))
            torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.config.gradient_clip_by_value)

    def one_step_training(self, data, epoch, step):

        self.optimizer.zero_grad()

        batch_start_time = time.time()
        images, labels = data
        images, labels = images.cuda(), labels.cuda()

        if step == 0:
            logging.info(f'images {images.shape}')
            logging.info(f'labels {labels.shape}')

        outputs = self.model(images)
        if step == 0:
            logging.info(f'outputs {outputs.shape}')

        loss = self.criterion(outputs, labels)
        loss.backward()
        self.process_gradients(step)
        self.optimizer.step()
        with self.warmup.dampening() if self.warmup else nullcontext():
            self.scheduler.step(step)

        seconds_per_batch = time.time() - batch_start_time
        examples_per_second = self.batch_size / seconds_per_batch

        if self._to_print(step):
            lr = self.optimizer.param_groups[0]['lr']
            self.message.add("epoch", epoch, format="4.2f")
            self.message.add("step", step, width=5, format=".0f")
            self.message.add("lr", lr, format=".6f")
            self.message.add("loss", loss, format=".4f")
            if self.config.print_grad_norm:
                grad_norm = self.compute_gradient_norm()
                self.message.add("grad", grad_norm, format=".4f")
            self.message.add("imgs/sec", examples_per_second, width=5, format=".0f")
            logging.info(self.message.get_message())


def set_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(config):
    set_seed(config.seed)

    folder = config.train_dir.split('/')[-1]
    if config.mode == 'train':
        trainer = Trainer(config)
        trainer()
    elif config.mode in ['certified', 'attack']:
        evaluate = Evaluator(config)
        evaluate()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or Evaluate Lipschitz Networks.')

    # parameters training or eval
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'certified', 'attack'])
    parser.add_argument("--train_dir", type=str, help="Name of the training directory.")
    parser.add_argument("--data_dir", type=str, help="Name of the data directory.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset to use")

    parser.add_argument("--shift_data", type=bool, default=True, help="Shift dataset with mean.")

    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training.")
    parser.add_argument("--loss", type=str, default="xent", help="Define the loss to use for training.")
    parser.add_argument("--margin", type=float, default=0.7, help="Define margin")
    parser.add_argument("--offset", type=float, default=np.sqrt(2) * 3 / 2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--scheduler", type=str, default="interp")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0, help="Weight decay to use for training.")
    parser.add_argument("--nesterov", action='store_true', default=False)
    parser.add_argument("--warmup_scheduler", type=float, default=0., help="Percentage of training.")
    parser.add_argument("--decay", type=str, help="Milestones for MultiStepLR")
    parser.add_argument("--gamma", type=float, help="Gamma for MultiStepLR")
    parser.add_argument("--gradient_clip_by_norm", type=float, default=None)
    parser.add_argument("--gradient_clip_by_value", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, help="Make the training deterministic.", default=42)
    parser.add_argument("--print_grad_norm", action='store_true', help="Print of the norm of the gradients")
    parser.add_argument("--frequency_log_steps", type=int, default=10, help="Print log for every step.")
    parser.add_argument("--logging_verbosity", type=str, default='INFO', help="Level of verbosity of the logs")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=5, help="Save checkpoint every epoch.")
    parser.add_argument("--compile", type=bool, default=True, help="Compile model")

    # specific parameters for eval
    parser.add_argument("--attack", type=str, choices=['pgd', 'autoattack'], help="Choose the attack.")
    parser.add_argument("--eps", type=float, default=36)

    # parameters of the architectures
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--w", type=int, default=1)
    parser.add_argument("--conv_inner_dim", type=int, default=5)
    parser.add_argument("--dense_inner_dim", type=int, default=-1)
    parser.add_argument("--n_conv", type=int, default=30, help="The depth of the Stable Resnet")
    parser.add_argument("--n_dense", type=int, default=5, help="The number of linear layers.")

    # parse all arguments
    config = parser.parse_args()
    config.cmd = f"python3 {' '.join(sys.argv)}"

    config.ngpus = torch.cuda.device_count()
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def override_args(config, n_conv, n_interior_conv, conv_inner_dim, n_dense, n_interior_dense, w, dense_inner_dim,
                      dense_next_layer):
        config.n_conv = n_conv
        config.n_interior_conv = n_interior_conv
        config.n_dense = n_dense
        config.n_interior_dense = n_interior_dense
        config.w = w
        config.dense_inner_dim = dense_inner_dim
        config.conv_inner_dim = conv_inner_dim

        config.dense_next_layer = dense_next_layer
        return config


    if config.model_name == 'small':
        config = override_args(config, 7, 4, 16, 7, 4, 45, 256, 2048)
    elif config.model_name == 'medium':
        config = override_args(config, 10, 4, 16, 10, 4, 60, 256, 2048)
    elif config.model_name == 'large':
        config = override_args(config, 15, 4, 16, 10, 4, 90, 256, 2048)
    elif config.model_name == 'xlarge':
        config = override_args(config, 20, 4, 16, 15, 4, 120, 256, 2048)
    elif config.model_name is None and \
            not all([config.n_conv, config.n_dense, config.w, config.dense_inner_dim]):
        ValueError("Choose --model-name 'small' 'medium' 'large' 'xlarge'")

    # process argments
    eval_mode = ['certified', 'attack']
    if config.data_dir is None:
        config.data_dir = os.environ.get('DATADIR', None)
    if config.data_dir is None:
        ValueError("the following arguments are required: --data_dir")
    os.makedirs('./trained_models', exist_ok=True)
    path = realpath('./trained_models')
    if config.train_dir is None:
        ValueError("--train_dir must be defined.")
    elif config.mode == 'train' and config.train_dir is not None:
        config.train_dir = f'{path}/{config.train_dir}'
        os.makedirs(config.train_dir, exist_ok=True)
        os.makedirs(f'{config.train_dir}/checkpoints', exist_ok=True)
    elif config.mode in eval_mode and config.train_dir is not None:
        config.train_dir = f'{path}/{config.train_dir}'
    elif config.mode in eval_mode and config.train_dir is None:
        ValueError("--train_dir must be defined.")

    if config.mode == 'attack' and config.attack is None:
        ValueError('With mode=attack, the following arguments are required: --attack')

    main(config)
