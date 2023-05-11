from dataclasses import dataclass
from scipy.interpolate import interp1d
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import animation


def isCorrectArray(a):
    n = len(a)

    for row in range(0, n):
        if (len(a[row]) != n):
            print('Не соответствует размерность')
            return False

    for row in range(1, n - 1):
        if (abs(a[row][row]) < abs(a[row][row - 1]) + abs(a[row][row + 1])):
            print('Не выполнены условия достаточности')
            return False

    if (abs(a[0][0]) < abs(a[0][1])) or (abs(a[n - 1][n - 1]) < abs(a[n - 1][n - 2])):
        print('Не выполнены условия достаточности')
        return False

    for row in range(0, len(a)):
        if (a[row][row] == 0):
            print('Нулевые элементы на главной диагонали')
            return False
    return True


# Процедура нахождения решения 3-х диагональной матрицы
def tridiagonal_solution(a, b):
    if (not isCorrectArray(a)):
        print('Ошибка в исходных данных')
        return -1

    n = len(a)
    x = [0 for k in range(0, n)]  # обнуление вектора решений

    # Прямой ход
    v = [0 for k in range(0, n)]
    u = [0 for k in range(0, n)]
    # для первой 0-й строки
    v[0] = a[0][1] / (-a[0][0])
    u[0] = (- b[0]) / (-a[0][0])
    for i in range(1, n - 1):  # заполняем за исключением 1-й и (n-1)-й строк матрицы
        v[i] = a[i][i + 1] / (-a[i][i] - a[i][i - 1] * v[i - 1])
        u[i] = (a[i][i - 1] * u[i - 1] - b[i]) / (-a[i][i] - a[i][i - 1] * v[i - 1])
    # для последней (n-1)-й строки
    v[n - 1] = 0
    u[n - 1] = (a[n - 1][n - 2] * u[n - 2] - b[n - 1]) / (-a[n - 1][n - 1] - a[n - 1][n - 2] * v[n - 2])

    # Обратный ход
    x[n - 1] = u[n - 1]
    for i in range(n - 1, 0, -1):
        x[i - 1] = v[i - 1] * x[i] + u[i - 1]

    return x


def calculate_alpha(k, drho, mu, phi, denom=1000):
    return k * drho * 9.81 / (mu * phi) / denom


@dataclass
class InputData:
    path: str  # path to .dat file
    cube_to_kg: float  # ratio from cubic meters of oil to kg
    W: float # well drainage length
    h_0: float  # height of GOC
    beta: float  # ratio of pressures
    phi: float  # coefficient of an effective porosity
    zw1: float  # level of the well
    dw: float  # diameter of the well
    k: float  # permeability coefficient
    drho: float  # difference of density
    mu: float  # dinamic viscosity
    Bg: float
    Bo: float
    gamma: float
    L: float
    p1: float
    p2: float
    n_x: int
    n_t: int
    zw2: float  #lower point of the well
    t0: float
    q0: float
    p0: float
    g: float = 9.80665
    @property
    def alpha(self):
        return calculate_alpha(drho=self.drho, k=self.k, mu=self.mu, phi=self.phi)


def diff_coef(z_nm, dt, dx, alpha):
    return dt * alpha * (2 * z_nm) ** (1 / 2) / (dx ** 2)


def well_flow_rate(y, q0, L, p1, p2):
    assert 0 <= y <= L, 'Invalid y coordinate'
    x0, y0, x1, y1 = 0, p1, L, p2
    coef = (y1 - y0) / (x1 - x0)
    bias = y1 - coef * x1
    return q0 / L * (coef * y + bias)


def solve(input_data: InputData):
    with open(f'{input_data.path}', 'r') as f:
        dat = f.read().split()

    days = []
    gas = []
    oil = []
    for i in range(len(dat) // 3):
        days.append(float(dat[i * 3]))
        oil.append(float(dat[i * 3 + 1]) * input_data.cube_to_kg)
        gas.append(float(dat[i * 3 + 2]))

    # interpolate oil
    debit = interp1d(days, oil, fill_value='extrapolate')
    n_x = input_data.n_x
    n_t = input_data.n_t
    dx = input_data.W / n_x
    dt = int(max(days)) / input_data.t0
    sol = np.zeros((n_t, n_x + 1))

    # initial condition
    sol[0] = np.array([input_data.h_0**2/2 for _ in range(n_x + 1)])
    for t, curr_t in enumerate(tqdm(np.linspace(0, len(oil), n_t - 1))):
        A = np.zeros((n_x + 1, n_x + 1))
        b = np.zeros(n_x + 1)

        # left boundary condition
        y = input_data.L / 2

        q_o = well_flow_rate(y, debit(curr_t), input_data.L, dt, dx) / (2 * input_data.alpha * input_data.phi)

        b[0] = -sol[t][1] + diff_coef(sol[t][1], dt, dx, input_data.alpha) * (q_o * dx) / (
                    input_data.alpha * input_data.phi)
        A[0][1] = - (1 - 2 * diff_coef(sol[t][1], dt, dx, input_data.alpha))
        A[0][0] = -2 * diff_coef(sol[t][1], dt, dx, input_data.alpha)

        # finite diff scheme
        for i in range(1, n_x):
            b[i] = -sol[t][i]
            A[i][i - 1] = diff_coef(sol[t][i], dt, dx, input_data.alpha)
            A[i][i] = -1 - 2 * diff_coef(sol[t][i], dt, dx, input_data.alpha)
            A[i][i + 1] = diff_coef(sol[t][i], dt, dx, input_data.alpha)

        # right boundary condition
        b[n_x] = -sol[t][n_x - 1]
        A[n_x][n_x - 1] = -(1 - 2 * diff_coef(sol[t][n_x - 1], dt, dx, input_data.alpha))
        A[n_x][n_x] = -2 * diff_coef(sol[t][n_x - 1], dt, dx, input_data.alpha)

        x = tridiagonal_solution(A, b)

        f = lambda v: input_data.zw2**2/2 if v < input_data.zw2**2/2 else v
        x = np.vectorize(f)(x)

        sol[t + 1] = x

    return (2*sol)**(1/2)


def animate(sol, input_data):
    n_x = sol.shape[1]
    n_t = 20


    fig, ax = plt.subplots()

    ax.axis([0, input_data.W, input_data.zw2-1, input_data.h_0])

    l, = ax.plot([], [])

    def anim_func(i):
        l.set_data(np.linspace(0, input_data.W, n_x), sol[i*int(sol.shape[0]/n_t)])

    ani = animation.FuncAnimation(fig, anim_func, frames=n_t)
    plt.plot(np.linspace(0, input_data.W, n_x), [input_data.zw2 for _ in range(n_x)], label='Well bottom')
    plt.plot(np.linspace(0, input_data.W, n_x), [input_data.zw1 for _ in range(n_x)], label='Well top')
    plt.show()


def plot_h0(sol):
    plt.plot(np.linspace(0, 169, sol.shape[0]), sol[:, 0], label=f'{sol.shape[0]}, {sol.shape[1]}')
