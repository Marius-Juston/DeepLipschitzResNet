"""Reject incompatible legacy checkpoints and unproved activation choices."""
import torch
from torch import nn


def validate_activation(activation):
    """Accept only implemented, componentwise [0,1]-slope activations."""
    if isinstance(activation, (list, nn.ModuleList)):
        for item in activation:
            validate_activation(item)
        return
    if isinstance(activation, (nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Softplus, nn.Identity)):
        return
    if isinstance(activation, nn.LeakyReLU) and 0 <= activation.negative_slope <= 1:
        return
    if isinstance(activation, nn.ELU) and 0 < activation.alpha <= 1:
        return
    raise ValueError(f"No [0,1]-sector certificate is implemented for {activation!r}")


class CertificateVersion:
    """State dictionaries without the corrected version cannot load silently."""
    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        version = state_dict.get(prefix + '_parameterization_version')
        if version is None or int(version) != 2:
            error_msgs.append(prefix + ' requires parameterization version 2; legacy checkpoints need a separate audit')
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)
