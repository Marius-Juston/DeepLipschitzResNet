import json
import logging
import os
from collections import defaultdict
from os.path import join

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm

from models.model import LipschitzNetwork
from training import utils
from training.data.readers import readers_config


class Evaluator:
    """Evaluate a Pytorch Model."""

    def __init__(self, config):
        raise RuntimeError(
            "Legacy CNN certificates have not been validated after the LDL correction. "
            "Use revision_experiments.py and src/revision_image.py for the audited protocol."
        )

    def load(self):
        checkpoints_path = join(self.config.train_dir, 'checkpoints', str(self.config.seed), 'best_model.pth')

        if not os.path.exists(checkpoints_path):
            logging.error(f'The checkpoint path {checkpoints_path} does not exist. Not loading model')
            raise ValueError(f"No checkpoint to verify has been found")
        else:
            logging.warning(f'Loading from path {checkpoints_path}.')
            self._load_state(checkpoints_path)

    def _load_state(self, checkpoint_directory):
        self.checkpoint = torch.load(checkpoint_directory, weights_only=False)

        new_config = self.checkpoint['config']

        if new_config.compile != self.config.compile:
            new_config.compile = self.config.compile

            if not self.config.compile:
                module = self.checkpoint['model_state_dict']

                module = {
                    k.replace('_orig_mod.', ''): v for k, v in module.items()
                }
            else:
                module = self.checkpoint['model_state_dict']

                module = {
                    k.replace('module.', 'module._orig_mod.'): v for k, v in module.items()
                }

            new_config.compile = self.config.compile
        else:
            module = self.checkpoint['model_state_dict']

        self.model.load_state_dict(module)

        if self.local_rank == 0:
            logging.info('Loading checkpoint {}'.format(checkpoint_directory))
            logging.info('Loading checkpoint {}'.format(checkpoint_directory))

    def __call__(self):
        """Run evaluation of model or eval under attack"""

        self.rank = int(os.environ['RANK'])
        self.local_rank = int(os.environ['LOCAL_RANK'])
        self.num_nodes = int(os.environ['LOCAL_WORLD_SIZE'])
        self.num_tasks = int(os.environ['WORLD_SIZE'])
        self.is_master = bool(self.rank == 0)

        torch.cuda.set_device(self.local_rank)

        self.message = utils.MessageBuilder()
        utils.setup_logging(self.config, self.rank)

        logging.info(f"Rank: {self.rank}")
        logging.info(f"Local rank: {self.local_rank}")
        logging.info(f"Num nodes: {self.num_nodes}")
        logging.info(f"Num tasks: {self.num_tasks}")

        ngpus = 1
        if ngpus:
            self.batch_size = self.config.batch_size * ngpus
        else:
            self.batch_size = self.config.batch_size

        # load reader
        Reader = readers_config[self.config.dataset]
        self.reader = Reader(self.config, self.batch_size, False, is_training=False)
        self.config.means = self.reader.means

        # load model
        self.model = LipschitzNetwork(self.config, self.reader.n_classes, device=f'cuda:{self.local_rank}')

        if self.config.compile:
            self.model = torch.compile(self.model, mode='max-autotune')

        utils.setup_distributed_training()

        self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        self.load()

        if self.config.mode == "certified":
            json_outputs = defaultdict(dict)

            for eps in [36, 72, 108, 255]:
                accuracy, certified = self.eval_certified(eps)

                json_outputs[eps]['accuracy'] = accuracy
                json_outputs[eps]['certified'] = certified

            if self.local_rank == 0:
                with open(os.path.join(self.config.train_dir, f'certified_seed_{self.config.seed}.json'), 'w') as f:
                    json.dump(json_outputs, f)

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

        if self.config.compile:
            module = self.model.module.model[-1]
        else:
            module = self.model.module.model[-1]

        normalized_weight = module.last_weight()

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
            running_accuracy += predicted.eq(labels.data).sum().item()
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
        for batch_n, data in tqdm(enumerate(data_loader), disable=self.local_rank != 0, total=len(data_loader)):
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
