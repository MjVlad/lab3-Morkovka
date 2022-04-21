import random

import matplotlib.pyplot as plt
import numpy as np


def exp(lamda, x):
    return -1 / lamda * np.log(x)


def theor_exp(lamda, t):
    return np.exp(-lamda * t)


def t1():
    N = 100000
    k = 3
    Pk = [0.3, 0.3, 0.4]
    lk = [0.7, 0.4, 0.9]
    out = []
    P_tmp = []
    Nk = [0 for _ in range(k)]

    tmp = 0
    for i in Pk:
        tmp += i
        P_tmp.append(tmp)

    for i in range(N):
        P_rand = random.random()
        for j in range(k):
            if P_rand < P_tmp[j]:
                Nk[j] += 1
                break

    for i in range(k):
        out.append(np.array([exp(lk[i], random.random()) for _ in range(Nk[i])]))

    s = 0
    for i in range(k):
        s += np.sum(out[i])
    s /= N

    size = 100
    R_theor = np.zeros(size)
    t = np.linspace(0, round(s, 3) + 1, size)
    Rt = np.zeros(size)
    for i in range(size):
        for j in out:
            Rt[i] += len(j[j >= t[i]]) / N

    for i in range(k):
        R_theor += theor_exp(lk[i], t) * Pk[i]

    plt.plot(t, R_theor, label="theor")
    plt.plot(t, Rt, label="exp")
    plt.legend()
    plt.show()

    l_theor = (-np.diff(R_theor) / R_theor[:-1] * size) - 1
    l_exp = (-np.diff(Rt) / Rt[:-1] * size) - 1

    plt.plot(t[:-1], l_theor, label="theor")
    plt.plot(t[:-1], l_exp, label="exp")
    plt.legend()
    plt.show()


def t2():
    N_test = 100000
    N = 3
    lk = [0.7, 0.4, 0.9]
    out = np.empty(N_test)

    for i in range(N_test):
        out[i] = (min([exp(lk[_], random.random()) for _ in range(N)]))

    s = np.sum(out) / N_test

    size = 100
    R_theor = np.full_like(np.arange(size, dtype='float64'), 1.)
    t = np.linspace(0, round(s, 3) + 1, size)
    Rt = np.zeros(size)
    for i in range(size):
        Rt[i] += len(out[out >= t[i]]) / N_test

    for i in range(N):
        R_theor *= theor_exp(lk[i], t)

    plt.plot(t, R_theor, label="theor")
    plt.plot(t, Rt, label="exp")
    plt.legend()
    plt.show()

    l_theor = (-np.diff(R_theor) / R_theor[:-1] * size) - 1
    l_exp = (-np.diff(Rt) / Rt[:-1] * size) - 1

    plt.plot(t[:-1], l_theor, label="theor")
    plt.plot(t[:-1], l_exp, label="exp")
    plt.ylim(0, 5)
    plt.legend()
    plt.show()


def t3():
    N_test = 100000
    N = 3
    lk = [0.7, 0.4, 0.9]
    out = np.zeros(N_test)

    for i in range(N_test):
        out[i] = (max([exp(lk[_], random.random()) for _ in range(N)]))

    s = np.sum(out) / N_test

    size = 100
    R_theor = np.full_like(np.arange(size, dtype='float64'), 1.)
    t = np.linspace(0, round(s, 3) + 1, size)
    Rt = np.zeros(size)
    for i in range(size):
        Rt[i] += len(out[out >= t[i]]) / N_test

    for i in range(N):
        R_theor *= 1 - theor_exp(lk[i], t)
    R_theor = 1 - R_theor

    plt.plot(t, R_theor, label="theor")
    plt.plot(t, Rt, label="exp")
    plt.legend()
    plt.show()

    l_theor = (-np.diff(R_theor) / R_theor[:-1] * size)
    l_exp = (-np.diff(Rt) / Rt[:-1] * size)

    plt.plot(t[:-1], l_theor, label="theor")
    plt.plot(t[:-1], l_exp, label="exp")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    t1()
    t2()
    t3()
