import torch

from models.model import LipschitzNetwork

if __name__ == '__main__':
    for i in range(42, 45):
        path = f'./trained_models/small_model/cifar10/checkpoints/{i}/best_model.pth'

        data = torch.load(path, weights_only=False)

        # model_state = data['model_state_dict']
        #
        # model = LipschitzNetwork(data['config'], 10, 'cuda')
        #
        # data = {k.replace('module._orig_mod.', ''):v for k, v in model_state.items()}

        # data['config'].wrong_normalization = True

        # torch.save(data, path)

        # data = torch.load(path, weights_only=False)
        # print(data['config'])

        # model.load_state_dict(data)
        #
        # x = torch.zeros((1, 45, 32, 32), device='cuda')
        # current_input, _, d_inv, prev_alpha, prev_omega_r, prev_constant = model.model[1].layers[0](x)
        #
        # print(len(model_state))



        data['config'].gradient_layer_normalization = False
        data['config'].wrong_normalization = True
        data['config'].activation = 'relu'
        data['config'].log_wandb = False
        #
        torch.save(data, path)
        #

        print(torch.load(path, weights_only=False)['config'].gradient_layer_normalization)