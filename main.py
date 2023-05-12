# Necessary libraries
import numpy as np
from datetime import datetime
from constants import *
from tqdm import tqdm
from scipy.interpolate import interp1d
from math import pi
from scipy.linalg import solve_banded
from utils import tridiagonal_solution
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.integrate import simps
import matplotlib.cm as cm
import time


def diff_coef(z_nm, dt, dx):
    return dt * alpha * (2 * z_nm) ** (1 / 2) / (dx ** 2)


def K(y):
    return (beta - 1) * y / L + 1


def well_flow_rate(y, q0, prev_y, dy):
    delta = np.minimum(np.maximum((np.full(prev_y.shape, zw1) - prev_y), 0) / dw, 1)
    ny = prev_y.shape[0]
    y_length = L * (y / (ny - 1))
    rate_in_point = (1 - delta[y] ** 2) * K(y_length)
    all_ys = np.linspace(0, L, ny)
    rate_overall = [(1 - delta[pt] ** 2) * K(y_len) for pt, y_len in enumerate(all_ys)]
    int_rate_overall = simps(y=rate_overall, dx=dy)
    if int_rate_overall <= 0:
        return 0
    return max(q0 * (rate_in_point / int_rate_overall), 0)


def psi(gnc, dy):
    all_ys = np.linspace(0, L, gnc.shape[0])
    delta = np.minimum(np.maximum((np.full(gnc.shape, zw2) - gnc), 0) / dw, 1)
    gas_diff = [(delta[pt] ** 2) * K(y_len) for pt, y_len in enumerate(all_ys)]
    oil_diff = [(1-delta[pt] ** 2) * K(y_len) for pt, y_len in enumerate(all_ys)]
    gas_rate = simps(y=gas_diff, dx=dy)
    oil_rate = simps(y=oil_diff, dx=dy)
    return gas_rate / oil_rate


def plot(sol):
    t = len(sol)
    n_x = len(sol[0]) - 1
    fig, ax = plt.subplots()
    ax.axis([0, W, (2 * zw2) ** (1 / 2) - 1, (2 * h_0) ** (1 / 2)])
    l, = ax.plot([], [], label='GOC')

    def animate(i):
        l.set_data(np.linspace(0, W, n_x + 1), sol[i])

    ani = animation.FuncAnimation(fig, animate, frames=t)
    plt.plot(np.linspace(0, W, n_x + 1), [zw2 for _ in range(n_x + 1)], label='Well bottom')
    plt.plot(np.linspace(0, W, n_x + 1), [zw1 for _ in range(n_x + 1)], label='Well top')
    plt.legend()
    plt.show()


def solve(path, nt, nx, ny):
    with open(path, 'r') as f:
        dat = f.read().split()
    # save gas and oil values
    days = []
    gas = []
    oil = []
    for i in range(len(dat) // 3):
        days.append(int(dat[i * 3]))
        oil.append(float(dat[i * 3 + 1]) * cube_to_kg)
        gas.append(float(dat[i * 3 + 2]))
    # interpolate oil
    debit = interp1d(days, oil, fill_value='extrapolate')
    Rs = gas[0]/oil[0]
    # to predict gas
    gas_init = gas[0]
    gas_pred = []

    dx = W / nx
    dt = max(days) / nt
    dy = L / ny

    sol = np.zeros((nt, nx, ny))

    # initial condition
    sol[0] = np.full((nx, ny), h_0)
    for t, curr_t in enumerate(tqdm(np.linspace(0, len(oil), nt - 1))):
        for y, curr_y in enumerate(np.linspace(0, L, ny)):
            A = np.zeros((nx, nx))
            b = np.zeros(nx)
            # left boundary condition

            q_o = well_flow_rate(y, debit(curr_t), sol[t][0][:], dy) / (2 * alpha * phi)

            b[0] = -sol[t][1][y] + diff_coef(sol[t][1][y], dt, dx) * (q_o * dx) / (alpha * phi)
            A[0][1] = - (1 - 2 * diff_coef(sol[t][1][y], dt, dx))
            A[0][0] = -2 * diff_coef(sol[t][1][y], dt, dx)

            # finite diff scheme
            for i in range(1, nx - 1):
                b[i] = -sol[t][i][y]
                A[i][i - 1] = diff_coef(sol[t][i][y], dt, dx)
                A[i][i] = -1 - 2 * diff_coef(sol[t][i][y], dt, dx)
                A[i][i + 1] = diff_coef(sol[t][i][y], dt, dx)

            # right boundary condition
            b[nx - 1] = -sol[t][nx - 2][y]
            A[nx - 1][nx - 2] = -(1 - 2 * diff_coef(sol[t][nx - 1][y], dt, dx))
            A[nx - 1][nx - 1] = -2 * diff_coef(sol[t][nx - 1][y], dt, dx)

            x = tridiagonal_solution(A, b)

            # Ab = np.zeros((3, nx))
            # Ab[0, 1:] = A.diagonal(1)
            # Ab[1, :] = A.diagonal()
            # Ab[2, :-1] = A.diagonal(-1)
            # x = solve_banded((1, 1), Ab, b)

            sol[t + 1, :, y] = x

        # overall_volume.append(simps(simps(sol[t] - zw2, axis=0, dx=dx), axis=0, dx=dy))

        # counting gas
        psi_coef = psi(sol[t, 0, :], dy)
        Qg = debit(curr_t)/cube_to_kg * (psi_coef*gamma/Bg + Rs/Bo)
        qg = Bg*(Qg - debit(curr_t)/(Bo*cube_to_kg))
        gas_pred.append(qg)


    plt.plot(gas_pred, label='gas pred')
    plt.plot(gas, label='gas')
    plt.plot(oil, label='oil')
    plt.legend()
    plt.show()
    return (2 * sol) ** (1 / 2), gas_pred


if __name__ == '__main__':
    t1 = time.time()
    path = './data/1501.dat'
    nx = 100
    nt = 169
    ny = 100

    surface, gas = solve(path, nt, nx, ny)

    # mirroring
    cone = np.concatenate([surface[:, ::-1, :], surface], axis=1)

    # def update_plot(frame_number):
    #     ax.clear()
    #     ax.set_zlim((2 * zw2) ** (1 / 2) - 0.5, 7)
    #     z = np.swapaxes(surface[frame_number, :, :], 0, 1)
    #     surface_plot = ax.plot_surface(X, Y, z, cmap="coolwarm", vmin=np.min(np.abs(z)), vmax=np.max(np.abs(z)))
    #     surface_plot.set_facecolor((0, 0, 0, 0))
    #     ax.auto_scale_xyz([X.min(), X.max()], [Y.min(), Y.max()], [(2 * zw2) ** (1 / 2) - 0.5, 7])
    #
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    #
    # X = np.linspace(-150, 150, 2*nx)
    # Y = np.linspace(0, L, ny)
    # X, Y = np.meshgrid(X, Y)
    #
    # z = np.swapaxes(surface[0, :, :], 0, 1)
    # cmap = plt.cm.get_cmap('coolwarm')
    #
    # plot = [ax.plot_surface(X, Y, z, color='0.75', rstride=1, cstride=1, cmap=cmap, vmin=np.min(np.abs(z)),
    #                         vmax=np.max(np.abs(z)))]
    # ax.set_zlim((2 * zw2) ** (1 / 2) - 0.5, 7)
    # ani = animation.FuncAnimation(fig, update_plot, nt, interval=1)
    #
    # plt.show()
    #
    def update_plot(frame_number):
        ax.clear()
        ax.set_zlim((2 * zw2) ** (1 / 2) - 0.5, 7)
        z = np.swapaxes(cone[frame_number, :, :], 0, 1)
        surface_plot = ax.plot_surface(X, Y, z, cmap="coolwarm", vmin=np.min(np.abs(z)), vmax=np.max(np.abs(z)))
        surface_plot.set_facecolor((0, 0, 0, 0))
        ax.auto_scale_xyz([X.min(), X.max()], [Y.min(), Y.max()], [(2 * zw2) ** (1 / 2) - 0.5, 7])
        well_level = np.ones_like(np.array([np.linspace(-W, W, 3), np.linspace(0, L, 3)])) * (2 * zw2) ** (1 / 2)
        well_level_plot = ax.plot_surface(np.linspace(-W, W, 3), np.linspace(0, L, 3), well_level, alpha=0.7, color='grey', label='well_level')
        well_level_plot.set_facecolor((0, 0, 0, 0))


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X = np.linspace(-150, 150, 2*nx)
    Y = np.linspace(0, L, ny)
    X, Y = np.meshgrid(X, Y)

    z = np.swapaxes(cone[0, :, :], 0, 1)
    cmap = plt.cm.get_cmap('coolwarm')

    plot = [ax.plot_surface(X, Y, z, color='0.75', rstride=1, cstride=1, cmap=cmap, vmin=np.min(np.abs(z)),
                            vmax=np.max(np.abs(z)))]
    ax.set_zlim((2 * zw2) ** (1 / 2) - 0.5, 7)
    ani = animation.FuncAnimation(fig, update_plot, nt, interval=1)

    plt.show()

