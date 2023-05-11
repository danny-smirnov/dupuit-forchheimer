# Necessary libraries
import numpy as np
from datetime import datetime
from constants import *
from tqdm import tqdm
from scipy.interpolate import interp1d
from math import pi
from scipy.linalg import solve_banded
from utils import tridiagonal_solution
from numpy.linalg import solve, lstsq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def diff_coef(z_nm, dt, dx):
    return dt * alpha * (2 * z_nm) ** (1 / 2) / (dx ** 2)


def well_flow_rate(y, q0):
    assert 0 <= y <= L, 'Invalid y coordinate'
    x0, y0, x1, y1 = 0, p1, L, p2
    coef = (y1 - y0) / (x1 - x0)
    bias = y1 - coef * x1
    return q0 / L * (coef * y + bias)


def plot(sol):
    t = len(sol)
    n_x = len(sol[0]) - 1
    fig, ax = plt.subplots()
    ax.axis([0, W, zw2 - 1, (2 * h_0) ** (1 / 2)])
    l, = ax.plot([], [], label='GOC')

    def animate(i):
        l.set_data(np.linspace(0, W, n_x + 1), sol[i])

    ani = animation.FuncAnimation(fig, animate, frames=t)
    plt.plot(np.linspace(0, W, n_x + 1), [zw2 for _ in range(n_x + 1)], label='Well bottom')
    plt.plot(np.linspace(0, W, n_x + 1), [zw1 for _ in range(n_x + 1)], label='Well top')
    plt.legend()
    plt.show()


def solve(path, nt, nx, y):
    with open(path, 'r') as f:
        dat = f.read().split()

    # save gas and oil values
    days = []
    gas = []
    oil = []
    for i in range(len(dat) // 3):
        days.append(float(dat[i * 3]))
        oil.append(float(dat[i * 3 + 1]) * cube_to_kg)
        gas.append(float(dat[i * 3 + 2]))

    # interpolate oil
    debit = interp1d(days, oil, fill_value='extrapolate')
    n_x = nx
    n_t = nt
    dx = W / n_x
    dt = int(max(days)) / n_t
    sol = np.zeros((n_t, n_x + 1))

    # initial condition
    sol[0] = np.array([h_0 for _ in range(n_x + 1)])
    for t, curr_t in enumerate(np.linspace(0, len(oil), n_t - 1)):
        A = np.zeros((n_x + 1, n_x + 1))
        b = np.zeros(n_x + 1)

        # left boundary condition
        q_o = well_flow_rate(y, debit(curr_t)) / (2 * alpha * phi)
        # q_o = 0
        b[0] = -sol[t][1] + diff_coef(sol[t][1], dt, dx) * (q_o * dx) / (alpha * phi)
        A[0][1] = - (1 - 2 * diff_coef(sol[t][1], dt, dx))
        A[0][0] = -2 * diff_coef(sol[t][1], dt, dx)

        # finite diff scheme
        for i in range(1, n_x):
            b[i] = -sol[t][i]
            A[i][i - 1] = diff_coef(sol[t][i], dt, dx)
            A[i][i] = -1 - 2 * diff_coef(sol[t][i], dt, dx)
            A[i][i + 1] = diff_coef(sol[t][i], dt, dx)

        # right boundary condition
        b[n_x] = -sol[t][n_x - 1]
        A[n_x][n_x - 1] = -(1 - 2 * diff_coef(sol[t][n_x - 1], dt, dx))
        A[n_x][n_x] = -2 * diff_coef(sol[t][n_x - 1], dt, dx)

        x = tridiagonal_solution(A, b)

        f = lambda v: zw2 ** 2 / 2 if v < zw2 ** 2 / 2 else v
        x = np.vectorize(f)(x)

        sol[t + 1] = x
    return (2 * sol) ** (1 / 2)


if __name__ == '__main__':
    path = './data/1501.dat'
    nt = 100
    nx = 1000
    n_y = 100
    surface = []
    y_steps = np.linspace(0, L, n_y)
    for y in tqdm(y_steps):
        surface.append(solve(path, nt, nx, y))
    surface = np.array(surface)
