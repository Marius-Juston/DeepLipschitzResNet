import numpy as np
from matplotlib import pyplot as plt


def func(a: np.ndarray, b: np.ndarray, g, n: int):
    n1 = 2 * (g + 1) ** 3 * ((a ** 2 * b ** 2 * (g + 1) ** 2) ** n + a ** 2 * b ** 2 * (
                -(a ** 2 * b ** 2 * (g + 1) ** 2) ** n + g * (g + 2) * ((a * b) ** (2 * n) - 1) + (a * b) ** (
                    2 * n)) - (a * b) ** (2 * n))
    d1 = (g + 2) * (a ** 2 * b ** 2 * (g + 1) ** 2 - 1)

    n2 = a * (a + 1) * ((a * b) ** (2 * n) - 1)
    d2 = a ** 2 * b ** 2 - 1

    f = a * (n2 - n1 / d1) / d2

    return f


def main():
    m = 100

    min_val = 10 ** (-11)

    max_val = 10

    epsilon = 2 ** (-11)

    x = np.linspace(min_val, max_val, m)
    y = np.linspace(min_val, max_val, m)
    xv, yv = np.meshgrid(x, y)

    for n in range(4, 5 + 1):
        zs = func(xv, yv, epsilon, n)

        plt.title(f'Log 10 $s_n$ error bound (n = {n})')
        plt.xlabel(r'$\alpha$')
        plt.ylabel(r'$\beta$')
        plt.contourf(x, y, np.log10(zs), levels=50)
        plt.axis('equal')
        plt.colorbar()
        plt.tight_layout()

        plt.savefig('n_' + str(n) + '.png', dpi=600)
        plt.show()


if __name__ == '__main__':
    main()
