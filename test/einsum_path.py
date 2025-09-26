import numpy as np
import opt_einsum as oe

from torch.backends.opt_einsum import is_available

print("Torch opt-einsum available:", is_available())

# Take a complex string
einsum_string = "fir,fjs,frk,fsk->fij"

n = 1024

dim = 512

x, y = np.random.random((n, dim, dim)), np.random.random((n, dim, dim))

path, path_info = oe.contract_path(einsum_string, x, x, y, y, optimize='optimal')

print(path)
print(path_info)
