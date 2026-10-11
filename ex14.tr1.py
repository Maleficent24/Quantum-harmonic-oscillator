import numpy as np
import matplotlib.pyplot as plt

# parameters
N = 20
a = 0.5
eps = 1.4
Ncf = 1000
N_boot = 100
POWER = 1            # 1: source/sink = x (this exercise); 3: x^3
NT = 6               # number of Delta E(t) points to report


def S(j, x):
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


def compute_G(x):
    """All G_n at once for one configuration (periodic)."""
    f = x ** POWER
    return np.array([np.mean(np.roll(f, -n) * f) for n in range(N)])


def generate(N_cor):
    """Return array G[alpha][n], one row per configuration."""
    x = np.zeros(N)
    for _ in range(5 * N_cor):                 # thermalize
        update(x)
    G = np.zeros((Ncf, N))
    for alpha in range(Ncf):
        for _ in range(N_cor):
            update(x)
        G[alpha] = compute_G(x)
    return G


def bin_data(G, binsize):
    nb = len(G) // binsize
    return G[:nb * binsize].reshape(nb, binsize, -1).mean(axis=1)


def bootstrap(G, nboot=N_boot):
    """Bootstrap copies of the ensemble (resample with replacement)."""
    n = len(G)
    return [G[np.random.randint(0, n, n)] for _ in range(nboot)]


def calculate_energy(Gavg):
    return np.log(np.abs(Gavg[:NT] / Gavg[1:NT + 1])) / a


def energy_with_error(G, binsize):
    Gb = bin_data(G, binsize)
    E = np.array([calculate_energy(s.mean(axis=0)) for s in bootstrap(Gb)])
    return calculate_energy(Gb.mean(axis=0)), E.std(axis=0)


# ---- 1) N_cor = 1: several runs, error vs bin size ----
print("N_cor = 1, several independent runs (error on Delta E at t=0, a, 2a)")
runs = [generate(1) for _ in range(4)]
for b in [1, 2, 5, 10, 20, 40]:
    print(f"\nbin size {b}")
    for r, G in enumerate(runs):
        E, dE = energy_with_error(G, b)
        print(f"  run {r}: DeltaE = {np.round(E[:3], 3)}  err = {np.round(dE[:3], 3)}")

# ---- 2) N_cor = 20 for comparison ----
G20 = generate(20)
E20, dE20 = energy_with_error(G20, 1)
print("\nN_cor = 20, no binning:", np.round(E20[:3], 3), "+-", np.round(dE20[:3], 3))

# ---- plot ----
t = a * np.arange(NT)
plt.figure()
E, dE = energy_with_error(runs[0], 1)
plt.errorbar(t - 0.03, E, yerr=dE, fmt="o", capsize=3, label="N_cor=1, no bin")
E, dE = energy_with_error(runs[0], 20)
plt.errorbar(t, E, yerr=dE, fmt="s", capsize=3, label="N_cor=1, bin=20")
plt.errorbar(t + 0.03, E20, yerr=dE20, fmt="^", capsize=3, label="N_cor=20")
plt.axhline(1.0, ls="--", color="k", label="exact = 1")
plt.xlabel("t = n a")
plt.ylabel(r"$\Delta E(t)$")
plt.legend()
plt.savefig("deltaE_bootstrap.png", dpi=120)
plt.show()
