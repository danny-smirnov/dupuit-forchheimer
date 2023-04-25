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



def diff_coef(z_nm):
    return dt * alpha * (2 * z_nm) ** (1 / 2) / (dx ** 2)


if __name__ == '__main__':

    # read .dat data
    with open('./data/4085.dat', 'r') as f:
        dat = f.read().split()

    # save gas and oil values
    days = []
    gas = []
    oil = []
    for i in range(len(dat) // 3):
        days.append(float(dat[i * 3]))
        gas.append(float(dat[i * 3 + 1]))
        oil.append(float(dat[i * 3 + 2]))

    # interpolate oil
    debit = interp1d(days, oil, fill_value='extrapolate')
    n_x = 10000
    n_t = 1000

    dx = W / n_x
    dt = days[-1] / (n_t*50)
    print(t0)
    sol = np.zeros((n_t, n_x + 1))

    # initial condition
    sol[0] = np.array([h_0 for _ in range(n_x + 1)])

    for t, curr_t in enumerate(tqdm(np.linspace(0, len(oil), n_t - 1))):
        A = np.zeros((n_x + 1, n_x + 1))
        b = np.zeros(n_x + 1)

        # left boundary condition
        q_o = 2*debit(curr_t) / (q0)
        b[0] = -sol[t][1] + diff_coef(sol[t][1]) * (q_o * dt) / (alpha * phi)
        A[0][0] = -2 * diff_coef(sol[t][1])
        A[0][1] = -(1 - 2 * diff_coef(sol[t][1]))

        # finite diff scheme
        for i in range(1, n_x):
            b[i] = -sol[t][i]
            A[i][i - 1] = diff_coef(sol[t][i])
            A[i][i] = -1 - 2 * diff_coef(sol[t][i])
            A[i][i + 1] = diff_coef(sol[t][i])

        # right boundary condition
        b[n_x] = -sol[t][n_x - 1]
        A[n_x][n_x - 1] = -(1 - 2 * diff_coef(sol[t][n_x - 1]))
        A[n_x][n_x] = -2 * diff_coef(sol[t][n_x - 1])

        x = tridiagonal_solution(A, b)
        f = lambda v: zw2 if v < zw2 else v
        x = np.vectorize(f)(x)

        sol[t + 1] = x


    t = n_t
    x = sol

    fig, ax = plt.subplots()
    ax.axis([0, W, 0, h_0])
    l, = ax.plot([], [], label='GOC')


    def animate(i):
        l.set_data(np.linspace(0, W, n_x + 1), sol[i])


    ani = animation.FuncAnimation(fig, animate, frames=t)
    plt.plot(np.linspace(0, W, n_x + 1), [zw2 for _ in range(n_x+1)], label='Well bottom')
    plt.plot(np.linspace(0, W, n_x + 1), [zw1 for _ in range(n_x+1)], label='Well top')
    plt.legend()
    plt.show()
