import numpy as np
import matplotlib.pyplot as plt

# parameters from the exercise
N = 20
a = 0.5
eps = 1.4
N_cor = 20
Ncf_values = [25, 100, 1000, 10000]


def S(j, x):
    """Part of the action that depends on x[j] (periodic, m=1, V=x^2/2)."""
    jp = (j + 1) % N
    jm = (j - 1) % N
    return a * x[j] ** 2 / 2 + x[j] * (x[j] - x[jp] - x[jm]) / a


def update(x):
    """One Metropolis sweep over all sites."""
    for j in range(N):
        old_x = x[j]
        old_Sj = S(j, x)
        x[j] = old_x + np.random.uniform(-eps, eps)
        dS = S(j, x) - old_Sj
        if dS > 0 and np.exp(-dS) < np.random.uniform(0, 1):
            x[j] = old_x


def compute_G(f, n):
    """G_n = (1/N) sum_j f[(j+n)%N] f[j], f = x or x^3 (source = sink)."""
    g = 0.0
    for j in range(N):
        g += f[(j + n) % N] * f[j]
    return g / N


def MCaverage(Ncf):
    x = np.zeros(N)
    for _ in range(5 * N_cor):                 # thermalize
        update(x)
    G1 = np.zeros((Ncf, N))                    # source/sink = x
    G3 = np.zeros((Ncf, N))                    # source/sink = x^3
    for alpha in range(Ncf):
        for _ in range(N_cor):
            update(x)
        x3 = x ** 3
        for n in range(N):
            G1[alpha][n] = compute_G(x, n)
            G3[alpha][n] = compute_G(x3, n)
    return G1.mean(axis=0), G3.mean(axis=0)


def deltaE(G, nmax=N // 2):
    # keep n aligned with t = n*a; use only n < N/2 (periodic symmetry)
    return np.log(np.abs(G[:nmax] / G[1:nmax + 1])) / a


plt.figure()
for Ncf in Ncf_values:
    G1, G3 = MCaverage(Ncf)
    dE1, dE3 = deltaE(G1), deltaE(G3)
    print(f"Ncf = {Ncf}")
    print("  DeltaE (x)   =", np.round(dE1[:7], 3))
    print("  DeltaE (x^3) =", np.round(dE3[:7], 3), "\n")
    t = a * np.arange(len(dE3))
    plt.plot(t, dE3, "o-", label=f"x^3, Ncf={Ncf}")

plt.axhline(1.0, ls="--", color="k", label="exact = 1")
plt.ylim(0, 3)
plt.xlabel("t = n a")
plt.ylabel(r"$\Delta E(t)$  (source = sink = $x^3$)")
plt.legend()
plt.savefig("deltaE_x3.png", dpi=120)
plt.show()
