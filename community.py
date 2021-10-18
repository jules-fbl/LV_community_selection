from scipy.integrate import solve_ivp
import numpy as np
import numpy.random as rd
from math import sqrt
from numba import jit
from time import process_time


@jit(nopython=True)
def compute_factor(gamma):
    if abs(gamma) > 1.:
        print("gamma>1 or gamma<1 not allowed")
        raise ValueError
    if gamma != 0.:
        return (1-sqrt(1-gamma**2))/gamma
    else:
        return 0.


@jit(forceobj=True)
def gaussian_matrix(S, mean, std, gamma):
    M = rd.normal(size=(S, S))
    factor = compute_factor(gamma)
    A = (M + factor*M.T)/sqrt(1+factor**2)  # symmetry correlation = gamma
    alpha = mean+std*A
    np.fill_diagonal(alpha, 0)  # null diagonal
    return alpha


def der_LV(t, Ns, rs, Ks, alpha):  # derivative from Lotka-Volterra equation
    return Ns*(rs-rs*Ns/Ks-alpha@Ns)


@jit(nopython=True)
def LV_logit(N_init, rs, Ks, alpha, S, t_max, dt):
    # Integration scheme equivalent to Euler for small dt
    # but ensure positivity and smoothness
    n_time = int(t_max/dt)
    ts = np.linspace(0, t_max, n_time)
    Ns = N_init.copy()
    ys = np.zeros(shape=(S, n_time))
    Ds = rs/Ks
    for i in range(1, n_time):
        K_eff = Ks - alpha @ Ns
        r_eff = K_eff*Ds
        Ns = K_eff*Ns/(Ns+(K_eff-Ns)*np.exp(-r_eff*(dt)))
        ys[:, i] = Ns.copy()
    return ts, ys


def jac(t, Ns, rs, Ks, alpha):  # Jacobian of Lotka-Volterra equation
    return np.diag(rs)-2*np.diag(rs*Ns/Ks) - \
        np.diag(alpha@Ns)-np.diag(Ns)@alpha


class Community():

    def __init__(self, S, gamma, N_min, immig, id='no'):
        # Fixed parameters :
        self.S = S
        self.gamma = gamma
        self.id = id
        self.N_min = N_min
        self.immig = immig
        # Initial conditions :
        self.rs = np.ones(S)
        self.Ks = np.ones(S)
        self.alpha = np.eye(S)
        self.Ns = rd.uniform(size=S)

    def set_gaussian_alpha(self, mu, sigma):
        S = self.S
        self.alpha = gaussian_matrix(S, mu/S, sigma/sqrt(S), self.gamma)
        self.mu_init = mu
        self.sigma_init = sigma

    def set_given_alpha(self, alpha):
        self.alpha = alpha.copy()

    def copy(self):
        comm = Community(self.S, self.gamma, self.N_min, self.immig, self.id)
        comm.alpha = self.alpha.copy()
        comm.Ns = self.Ns.copy()
        comm.Ks = self.Ks.copy()
        comm.mu_init = self.mu_init
        comm.sigma_init = self.sigma_init
        return comm

    @jit(forceobj=True)
    def integrate_LV(self, t_max, method='logit'):
        if method == 'logit':  # Integrate LV with own method
            dt = 0.5  # a mettre autre part
            return LV_logit(self.Ns, self.rs, self.Ks, self.alpha,
                            self.S, t_max, dt)
        else:
            sol = solve_ivp(der_LV, (0, t_max), self.Ns, method=method,
                            args=(self.rs, self.Ks, self.alpha))
            return sol.t, sol.y

    def equilibrate_LV(self, t_max):
        if self.immig:
            self.Ns[np.where(self.Ns < self.N_min)] = 10*self.N_min
        t0 = process_time()
        ts, ys = self.integrate_LV(t_max)
        t1 = process_time()
        self.Ns = ys[:, -1]
        self.Ns[np.where((self.Ns < self.N_min))] = 0
        return t1-t0

    @jit(forceobj=True)
    def mutate(self, epsilon):
        tri_ind = np.triu_indices(self.S, 1)
        mean, std = np.mean(self.alpha[tri_ind]), np.std(self.alpha[tri_ind])
        beta = (self.alpha-mean)/std  # reduced matrix with mean=0 and std=1
        eta = gaussian_matrix(self.S, 0, 1, self.gamma)  # variations
        beta = (beta + epsilon*eta)/sqrt(1+epsilon**2)  # new reduced matrix
        self.alpha = mean + std*beta  # new interaction matrix
        np.fill_diagonal(self.alpha, 0)
        return eta

    @jit(forceobj=True)
    def mutate_no_ms(self, epsilon):
        mean, std = self.mu_init/self.S, self.sigma_init/np.sqrt(self.S)
        beta = (self.alpha-mean)/std  # reduced matrix with mean=0 and std=1
        eta = gaussian_matrix(self.S, 0, 1, self.gamma)  # variations
        beta += epsilon*eta
        beta = (beta-np.mean(beta))/np.std(beta)  # new reduced matrix
        self.alpha = mean + std*beta  # new interaction matrix
        np.fill_diagonal(self.alpha, 0)
        return eta


class Metacommunity():

    def __init__(self, n_comm, S, gamma, N_min, immig):
        self.S = S
        self.n_comm = n_comm
        self.comm_list = np.array([Community(S, gamma, N_min, immig, str(id))
                                   for id in range(n_comm)])

    def set_same_gaussian_alpha(self, mu, sigma):
        self.comm_list[0].set_gaussian_alpha(mu, sigma)
        for i in range(1, self.n_comm):
            self.comm_list[i].set_given_alpha(self.comm_list[0].alpha)
            self.comm_list[i].mu_init = mu
            self.comm_list[i].sigma_init = sigma

    def set_same_random_Ks(self):
        Ks = rd.uniform(size=self.S, low=0.5, high=1.5)
        for comm in self.comm_list:
            comm.Ks = Ks.copy()
        return Ks

    def set_different_gaussian_alpha(self, mu, sigma):
        for comm in self.comm_list:
            comm.set_gaussian_alpha()

    @jit(forceobj=True)
    def equilibrate_all_LV(self, t_max):  # Independend integrations
        tLV_max = 0.
        for comm in self.comm_list:
            tLV = comm.equilibrate_LV(t_max)
            if tLV > tLV_max:
                tLV_max = tLV
        return tLV_max

    def select(self, n_select, score_func, ws):
        if n_select > self.n_comm:
            raise ValueError('n_select > n_community impossible')

        scores = np.array([score_func(comm.Ns, ws) for comm in self.comm_list])
        sorted_index = (-scores).argsort()

        for q in sorted_index[n_select:]:
            i = rd.randint(low=0, high=n_select)
            self.comm_list[q] = self.comm_list[sorted_index[i]].copy()
        # return the list of selected communities
        return sorted_index[:n_select]

    @jit(forceobj=True)
    def mutate_all_no_ms(self, epsilon):
        return [comm.mutate_no_ms(epsilon) for comm in self.comm_list]

    @jit(forceobj=True)
    def mutate_all(self, epsilon):
        return [comm.mutate(epsilon) for comm in self.comm_list]

    def has_exploded(self):
        return np.max([np.max(comm.Ns) for comm in self.comm_list]) > 10**3
