# Necessary libraries
from utils import solve, InputData, animate, plot_h0
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    k = 176.1  # permeability coef
    drho = 814.4  # oil-gas pressure difference
    g = 9.80665  # acceleration of gravity
    mu = 18.76  # dynamic viscosity
    phi = 0.266  # effective porosity

    h_0 = 7  # initial height
    W = 150  # half of the drain length
    L = 1171.3  # well length
    dw = 0.089  # well diameter
    zw1 = 5.6
    zw2 = zw1 - dw
    cube_to_kg = 800  # coefficient to translate m^3 to kg
    p1 = 30  # pressure at t1 point
    p2 = 10  # pressure at t3 point
    Bg = 0.00935
    Bo = 1.069
    gamma = 100
    beta = 0.191794288875527

    p0 = drho * g * h_0 / 10 ** 5  # bar
    q0 = 2 * k * p0 * L * h_0 / mu / W / 115.74
    t0 = 115.74 * W ** 2 * mu * phi / k / p0

    inp = InputData(path='./data/1501.dat',
                    cube_to_kg=cube_to_kg, k=k,
                    W=W, L=150,
                    g=g, mu=mu,
                    phi=phi, h_0=h_0,
                    zw1=zw1, zw2=zw2,
                    dw=dw, n_t=100,
                    n_x=100, drho=drho,
                    Bo=Bo, Bg=Bg,
                    beta=beta, gamma=gamma,
                    p1=p1, p2=p2,
                    p0=p0, q0=q0, t0=t0)

    num_of_t = [100, 500, 1000]
    num_of_x = [100, 500, 1000]
    sol_overall = []
    for n_x in num_of_x:
        inp.n_x = n_x
        sol = solve(inp)
        # plot_h0(sol)
        sol_overall.append(sol[:, 0])

    # plt.legend()
    # plt.show()
    plt.plot(np.linspace(0, 169, sol_overall[0].shape[0]), sol_overall[0])
    plt.plot(np.linspace(0, 169, sol_overall[1].shape[0]), sol_overall[1])
    plt.plot(np.linspace(0, 169, sol_overall[2].shape[0]), sol_overall[2])

    plt.show()
