import os.path
from typing import List

import torch

from cnn_train import LipschitzNetwork
from model_lmi import block, E_real, plot_matrix, plot_histogram, plot_singular_extremes

DEFAULT_DPI = 600

OUTPUT_FORMAT = 'png'



def main_real(A: torch.Tensor, B: torch.Tensor, Cs: List[torch.Tensor], Ls: List[torch.Tensor], Lms, dimensions,
              device=None):
    L = len(Ls)

    factory_kwargs = dict(device=device, dtype=A.dtype)

    E0 = E_real(dimensions, None, **factory_kwargs)

    I = torch.eye(dimensions[0], **factory_kwargs)

    M0 = block([[A.mT @ A - I, A.mT @ B], [B.mT @ A, B.mT @ B]])

    M0 = E0.T @ M0 @ E0

    Ms = [M0]

    for i in range(L):
        Ei = E_real(dimensions, i, **factory_kwargs)

        lambda_i = Ls[i]
        Ci = Cs[i]

        CC = torch.zeros((Ci.shape[0], Ci.shape[-2] * 2, Ci.shape[-1] + Ci.shape[-2]), **factory_kwargs)
        CC[..., :Ci.shape[-2], :Ci.shape[-1]] = Ci
        CC[..., -Ci.shape[-2]:, -Ci.shape[-2]:] = torch.eye(Ci.shape[-2], **factory_kwargs)

        L_, m = Lms[i]

        M_sub = block(
            [
                [-2 * L_ * m * lambda_i, (m + L_) * lambda_i],
                [(m + L_) * lambda_i, -2 * lambda_i],
            ]
        ).to(**factory_kwargs)

        Mi = CC.mH @ M_sub @ CC
        expanded_Mi = Ei.T @ Mi @ Ei

        Ms.append(expanded_Mi)

    output_M = Ms[0]

    for i in range(1, len(Ms)):
        output_M += Ms[i]

    M = output_M

    return M


def extract_lmi_constants_deep_resnet_new(model):
    x = torch.zeros((1, model.in_channels, 16, 16), device=model.device, dtype=model.dtype)

    _, a_wf, d_inv = model.A(x)

    Cs = []
    Ls = []
    Lms = []
    dimensions = [
        a_wf.shape[-1],
    ]

    gamma = torch.eye(d_inv.shape[1], device=d_inv.device, dtype=d_inv.dtype).expand(d_inv.shape[0], -1, -1)
    sigma_lower = None

    current_input = x

    prev_layer = model.A

    for layer in model.layers:
        T = gamma @ d_inv

        current_input, c_wf, d_inv = layer(current_input, prev_layer)

        if sigma_lower is None:
            sigma_lower = T
        else:
            sigma_lower = torch.linalg.cholesky(sigma_lower @ sigma_lower.mH + T @ T.mH, upper=False)

        gamma = T @ c_wf.mH

        current_input = model.activation(current_input)

        prev_layer = layer

        Cs.append(c_wf)
        Ls.append(layer.identity_out)
        Lms.append([1, 0])
        dimensions.append(c_wf.shape[1])

    _, b_wf = model.compute_b(current_input, sigma_lower, a_wf, model.layers[-1])

    return a_wf, b_wf, Cs, Ls, Lms, dimensions


def extract_lmi_constants(model):
    return extract_lmi_constants_deep_resnet_new(model)


def main():
    torch.manual_seed(0)

    figs_folder = 'figs/conv'
    os.makedirs(figs_folder, exist_ok=True)

    input_features = 1
    output_features = input_features
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    factory_kwargs = dict(device=device, dtype=torch.float32)

    weights = (32, 32, 32, 32, 32)

    # test_sqrt(device)

    torch.no_grad()

    inputs = (input_features, output_features, weights)

    # model = get_maximum_model(model_fnc, *inputs, device=device)

    model = LipschitzNetwork(device=device)
    # model = DeepLipschitzLinearResNet(*inputs, device=device)

    print(model)

    save_folder = '/home/mjuston2/Documents/DeepLipschitzResNet/runs/general_cnn_trainer_20250923_020626'
    # save_folder = None

    if save_folder is not None:
        data = torch.load(f"{save_folder}/best.pt", map_location=device)
        new_state_dict = {key.replace('_orig_mod.', ''): val for key, val in data.items()}

        model.load_state_dict(new_state_dict)

    model = model.conv[0]

    with torch.no_grad():
        a_weight, b_weight, Cs, Ls, Lms, dimensions = extract_lmi_constants(model)

    M = main_real(a_weight, b_weight, Cs, Ls, Lms, dimensions, device=device)

    eigenvalues, eigenvectors = torch.linalg.eigh(M)

    assert torch.all(eigenvalues < 1e-3), "Not all the eigenvalues of the LMI are negative"

    print(M)

    print(eigenvalues)

    plot_histogram(eigenvalues.flatten(), os.path.join(figs_folder, "eigenvalues_lmi"))
    plot_singular_extremes(a_weight, b_weight, Cs, os.path.join(figs_folder, "singular_weights"))

    max_num = 5

    v = torch.arange(M.shape[1])

    M[:, v,v] = 0.0

    for freq in range(min(M.shape[0], max_num)):
        plot_matrix(M[freq].real, os.path.join(figs_folder, f"full_lmi_{freq}_real"))

        im_sum = M[freq].imag.abs().sum()

        print(im_sum)

        if im_sum > 1e-2:
            plot_matrix(M[freq].imag, os.path.join(figs_folder, f"full_lmi_{freq}_img"))




if __name__ == '__main__':
    main()
