import numpy as np
import matplotlib.pyplot as plt

# parameters from the exercise
N = 20
a = 0.5
eps = 1.4
N_cor = 20
Ncf_values = [25, 100, 1000, 10000]


def S(j, x):
    jp = (j + 1) % N
    jm = (j - 1) % N
    return a * x[j] ** 2 / 2 + x[j] * (x[j] - x[jp] - x[jm]) / a


def update(x):
    
    for j in range(N):
        old_x = x[j]
        old_Sj = S(j, x)
        x[j] = old_x + np.random.uniform(-eps, eps)
        dS = S(j, x) - old_Sj
        if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
            x[j] = old_x


def compute_G(x, n):
    g = 0.0
    for j in range(N):
        g += x[(j + n) % N] * x[j]
    return g / N


def MCaverage(Ncf):
    x = np.zeros(N)                      # initialize
    for _ in range(5 * N_cor):           # thermalize (discard initial sweeps)
        update(x)
    G = np.zeros((Ncf, N))
    xmean = 0.0
    for alpha in range(Ncf):
        for _ in range(N_cor):
            update(x)
        for n in range(N):
            G[alpha][n] = compute_G(x, n)
        xmean += x.mean()
    return G.mean(axis=0), xmean / Ncf   # average over configurations


plt.figure()
for Ncf in Ncf_values:
    avg_G, avg_x = MCaverage(Ncf)
    nmax = N // 2
    dE = np.log(np.abs(avg_G[:nmax] / avg_G[1:nmax + 1])) / a
    print(f"Ncf = {Ncf}   <x> = {avg_x:.4f}")
    print("G       =", np.round(avg_G[:nmax], 4))
    print("DeltaE  =", np.round(dE, 4), "\n")
    plt.plot(a * np.arange(nmax), dE, "o-", label=f"Ncf={Ncf}")

plt.axhline(1.0, ls="--", color="k", label="exact = 1")
plt.ylim(0, 3)
plt.xlabel("t = n a")
plt.ylabel(r"$\Delta E(t)$")
plt.legend()
plt.savefig("deltaE_simple.png", dpi=120)
plt.show()
