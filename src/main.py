import random
import traceback

import torch
from torch.distributed import destroy_process_group
from torch.distributed.elastic.multiprocessing.errors import record

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

import argparse
import os
import sys
import warnings
from os.path import realpath

import numpy as np

from training.evaluate import Evaluator
from training.trainer import Trainer

warnings.filterwarnings("ignore")


def set_seed(seed):
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@record
def main(config):
    set_seed(config.seed)

    try:
        if config.mode == 'train':
            trainer = Trainer(config)
            trainer()
            trainer.load()
            trainer.train()
        elif config.mode in ['certified', 'attack']:
            evaluate = Evaluator(config)
            evaluate()
    except Exception as e:
        traceback.print_exc()
    finally:
        destroy_process_group()

def model_choice(config):
    if config.model_name == 'small':
        config = override_args(config, 20, 64, 7, 45, 1024)
    elif config.model_name == 'medium':
        config = override_args(config, 30, 64, 10, 60, 1024)
    elif config.model_name == 'large':
        config = override_args(config, 50, 64, 10, 90, 1024)
    elif config.model_name == 'xlarge':
        config = override_args(config, 70, 64, 15, 120, 1024)
    elif config.model_name is None and \
            not all([config.n_conv, config.n_dense, config.w, config.dense_inner_dim]):
        ValueError("Choose --model-name 'small' 'medium' 'large' 'xlarge'")

    return config


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train or Evaluate Lipschitz Networks.')

    # parameters training or eval
    parser.add_argument("--mode", type=str, default="train", choices=['train', 'certified', 'attack'])
    parser.add_argument("--train_dir", type=str, help="Name of the training directory.")
    parser.add_argument("--data_dir", type=str, help="Name of the data directory.")
    parser.add_argument("--dataset", type=str, default='cifar10', help="Dataset to use")
    parser.add_argument("--num_workers", type=int, default=20,
                        help="how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process")
    parser.add_argument("--prefetch_factor_scale", type=int, default=2,
                        help="Number of batches loaded in advance by each worker: batch_size * prefetch_factor_scale")

    parser.add_argument("--linear_conv", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Use linear Lipschitz layers rather ResNet layers for better stability")

    parser.add_argument("--shift_data", type=bool, default=True, help="Shift dataset with mean.")

    parser.add_argument("--shift_data", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Shift dataset with mean.")

    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs for training.")
    parser.add_argument("--validate_every", type=int, default=1,
                        help="Number of epochs to run the validation on the test dataset on.")

    parser.add_argument("--loss", type=str, default="xent", help="Define the loss to use for training.")
    parser.add_argument("--margin", type=float, default=0.7, help="Define margin")
    parser.add_argument("--offset", type=float, default=np.sqrt(2) * 3 / 2)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--optimizer", type=str, default='adamw')
    parser.add_argument("--scheduler", type=str, default="interp")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.9)
    parser.add_argument("--wd", type=float, default=0, help="Weight decay to use for training.")
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--warmup_scheduler", type=float, default=0., help="Percentage of training.")
    parser.add_argument("--decay", type=str, help="Milestones for MultiStepLR")
    parser.add_argument("--gamma", type=float, help="Gamma for MultiStepLR")
    parser.add_argument("--gradient_clip_by_norm", type=float, default=1.0)
    parser.add_argument("--gradient_clip_by_value", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, help="Make the training deterministic.", default=42)
    parser.add_argument("--print_grad_norm", action='store_true', help="Print of the norm of the gradients")
    parser.add_argument("--frequency_log_steps", type=int, default=10, help="Print log for every step.")
    parser.add_argument("--logging_verbosity", type=str, default='INFO', help="Level of verbosity of the logs")
    parser.add_argument("--save_checkpoint_epochs", type=int, default=5, help="Save checkpoint every epoch.")
    parser.add_argument("--compile", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Compile model")
    parser.add_argument("--safe_cholesky", type=bool, default=True, action=argparse.BooleanOptionalAction,
                        help="Enable safe cholesky computation for the compute_b function")

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


    def override_args(config, n_conv, conv_inner_dim, n_dense, w, dense_inner_dim):
        config.n_conv = n_conv
        config.n_dense = n_dense
        config.w = w
        config.dense_inner_dim = dense_inner_dim
        config.conv_inner_dim = conv_inner_dim

        return config


    config = model_choice(config)

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
        config.train_dir = f'{path}/{config.train_dir}/{config.dataset}'
        os.makedirs(config.train_dir, exist_ok=True)
        os.makedirs(f'{config.train_dir}/checkpoints', exist_ok=True)
    elif config.mode in eval_mode and config.train_dir is not None:
        config.train_dir = f'{path}/{config.train_dir}/{config.dataset}'
    elif config.mode in eval_mode and config.train_dir is None:
        ValueError("--train_dir must be defined.")

    if config.mode == 'attack' and config.attack is None:
        ValueError('With mode=attack, the following arguments are required: --attack')

    main(config)
