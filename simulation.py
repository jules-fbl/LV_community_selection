import numpy as np
import numpy.random as rd
from math import sqrt
from numba import jit
from community import Metacommunity
import resource
import matplotlib.pyplot as plt
from time import process_time
from tqdm import tqdm

# The differents scores function.
# ws is always a parameter but used only for L.Cnz


def print_status(i, n, mem_init):
    # print((i*100)//n, ' % done')
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print('Memory usage: %s (kb)' %
          (memory-mem_init))


def Shannon_index(Ns, ws):  # diversity of species "continuous" ~entropy
    N_pos = Ns[np.where(Ns > 10**(-15))]
    N_total = np.sum(N_pos)
    Ns = N_pos/N_total
    H = -np.sum(Ns*np.log(Ns))
    if H == np.inf:
        print(Ns, N_pos, N_total, Ns)
    return H


def diversity(Ns, ws): return len(np.where(Ns > 0)[0])/len(Ns)


def random_selection(Ns, ws): return rd.uniform()


def linear_combination(Ns, ws): return np.dot(ws, Ns)/len(Ns)


def linear_combination_relative(Ns, ws):
    ns = Ns/np.sum(Ns)
    return np.dot(ws, ns)/len(ns)


def deviation_relative(Ns, ws):
    ns = Ns/np.sum(Ns)
    b_s = ws/np.sum(ws)
    return -sqrt(np.mean((ns-b_s)**2))


def deviation(Ns, ws):
    return -sqrt(np.mean((Ns-ws)**2))


score_func_dict = {'shannon': Shannon_index,
                   'diversity': diversity,
                   'total_mass': linear_combination,
                   'random_selection': random_selection,
                   'random_combination_pos': linear_combination,
                   'random_combination': linear_combination,
                   'random_combination_relative': linear_combination_relative,
                   'random_deviation': deviation,
                   'random_deviation_relative': deviation_relative,
                   'zero_deviation': deviation}


def gen_ws(score_func_name, S):
    ws = np.zeros(S)
    if score_func_name == 'total_mass':
        ws = np.ones(S)
    elif score_func_name in ['random_combination',
                             'random_combination_relative']:
        ws = rd.uniform(low=-1, high=1, size=S)
        ws -= np.mean(ws)
        ws = ws/sqrt(np.sum(ws**2))
    elif score_func_name == 'random_combination_pos':
        ws = rd.uniform(size=S)
        ws = ws/sqrt(np.sum(ws**2))
    elif score_func_name in ['random_deviation', 'random_deviation_relative']:
        ws = rd.uniform(size=S)
    elif score_func_name == 'zero_deviation':
        ws = np.zeros(S)
    return ws


@jit(forceobj=True)
def get_eig(alpha, S):
    A = np.eye(S) + alpha
    W, V = np.linalg.eig(A)
    if len(V) > 0:
        imin = (W.real).argmin()
        vmin = V[:, imin]
        return W, vmin
    else:
        return [], []


class Dataset():
    def __init__(self, data_to_get, S, ws, Ks):
        self.data_to_get = data_to_get
        self.S = S
        self.ws = ws
        self.Ks = Ks
        self.alpha0 = False
        self.alpha0_star = False
        self.tri_ind = np.triu_indices(S, 1)

        if data_to_get['alpha_eig']:
            self.lamb_list = []
            self.vmin_list = []
        if data_to_get['alpha_star_eig']:
            self.lamb_star_list = []
            self.vmin_star_list = []
        if data_to_get['Ns']:
            self.Ns_list = []
        if data_to_get['mu_sigma']:
            self.mu_list = []
            self.sigma_list = []
        if data_to_get['mutation_direction']:
            self.eta_dir = []
        if self.data_to_get['alpha']:
            self.alpha_list = []
        if self.data_to_get['coeff_abcd']:
            self.a_list = []
            self.b_list = []
            self.c_list = []
            self.d_list = []
        # have to make same with full !

    @jit(forceobj=True)
    def measure(self, metacomm):
        comm = metacomm.comm_list[0]
        if self.data_to_get['alpha_eig']:
            W, vmin = get_eig(comm.alpha, self.S)
            self.lamb_list.append(W.copy())
            self.vmin_list.append(vmin.copy())
        if self.data_to_get['alpha_star_eig']:
            ind_pos = np.where(comm.Ns > 0)[0]
            alpha_star = comm.alpha[np.ix_(ind_pos, ind_pos)]
            W, vmin = get_eig(alpha_star, len(ind_pos))
            self.lamb_star_list.append(W.copy())
            self.vmin_star_list.append(vmin.copy())
        if self.data_to_get['Ns']:
            self.Ns_list.append(comm.Ns)
        if self.data_to_get['mu_sigma']:
            mean, std = np.mean(comm.alpha[self.tri_ind]), np.std(
                comm.alpha[self.tri_ind])
            self.mu_list.append(mean*self.S)
            self.sigma_list.append(std*sqrt(self.S))
        if self.data_to_get['alpha']:
            self.alpha_list.append(comm.alpha.copy())
        if self.data_to_get['coeff_abcd']:
            A = np.eye(self.S) + comm.alpha
            u = np.ones(self.S)/np.sqrt(self.S)
            self.a_list.append(u.T@A@u)
            self.b_list.append(u.T@A@self.ws)
            self.c_list.append((self.ws).T@A@u)
            self.d_list.append((self.ws).T@A@self.ws)

    def measure_init(self, metacomm):
        if self.data_to_get['alpha_init_last']:
            self.alpha_init = metacomm.comm_list[0].alpha.copy()

    def measure_last(self, metacomm):
        if self.data_to_get['alpha_init_last']:
            self.alpha_last = metacomm.comm_list[0].alpha.copy()

    def measure_mutation(self, metacomm, eta):
        if self.data_to_get['mutation_direction']:
            comm = metacomm.comm_list[0]
            ind_pos = np.where(comm.Ns > 0)[0]
            A_star = np.eye(len(ind_pos))+comm.alpha[np.ix_(ind_pos, ind_pos)]
            chi = np.linalg.inv(A_star)
            eta_star = eta[np.ix_(ind_pos, ind_pos)]
            u = chi.T@(self.ws[ind_pos])
            v = chi@np.ones(len(ind_pos))
            # v= N_star
            self.eta_dir.append((u@eta_star@v /
                                 np.sqrt(np.sum(u**2)*np.sum(v**2))).copy())
        else:
            pass

    def save(self, dir):
        np.save(dir+'ws.npy', self.ws)
        np.save(dir+'Ks.npy', self.Ks)
        if self.data_to_get['alpha_eig']:
            np.save(dir+'lambda.npy', self.lamb_list)
            np.save(dir+'vmin.npy', self.vmin_list)
        if self.data_to_get['alpha_star_eig']:
            np.save(dir+'lambda_star.npy', self.lamb_star_list)
            np.save(dir+'vmin_star.npy', self.vmin_star_list)
        if self.data_to_get['Ns']:
            np.save(dir+'Ns.npy', self.Ns_list)
        if self.data_to_get['mu_sigma']:
            np.save(dir+'mu.npy', self.mu_list)
            np.save(dir+'sigma.npy', self.sigma_list)
        if self.data_to_get['mutation_direction']:
            np.save(dir+'eta_dir.npy', self.eta_dir)
        if self.data_to_get['alpha']:
            np.save(dir+'alpha.npy', self.alpha_list)
        if self.data_to_get['coeff_abcd']:
            np.save(dir+'as.npy', self.a_list)
            np.save(dir+'bs.npy', self.b_list)
            np.save(dir+'cs.npy', self.c_list)
            np.save(dir+'ds.npy', self.d_list)
        if self.data_to_get['alpha_init_last']:
            np.save(dir+'alpha_init.npy', self.alpha_init)
            np.save(dir+'alpha_last.npy', self.alpha_last)
        print('Saved in '+dir)

    def plot(self):
        if self.data_to_get['Ns']:
            for ns in np.array(self.Ns_list).T:
                plt.plot(ns)
            plt.title('Ni with generations')
            plt.show()
            plt.plot(np.mean(self.Ns_list, axis=1))
            plt.title('Mean population')
            plt.show()
        if self.data_to_get['alpha_eig']:
            for lamb in np.array(self.lamb_list).T:
                plt.plot(lamb.real)
            plt.title('eigenvalues of A')
            plt.show()
        if self.data_to_get['mu_sigma']:
            plt.plot(self.mu_list)
            plt.title('mu : mean of interactions')
            plt.show()
            plt.plot(self.sigma_list)
            plt.title('sigma')
            plt.show()
        if self.data_to_get['coeff_abcd']:
            plt.plot(self.a_list, label='a')
            plt.plot(self.b_list, label='b')
            plt.plot(self.c_list, label='c')
            plt.plot(self.d_list, label='d')
            plt.legend()
            plt.show()


class Simu_artificial_select():

    def __init__(self, n_comm, S, gamma, N_min, immig, mu, sigma, n_gen,
                 epsilon, t_max, n_select, score_func_name, data_to_get,
                 random_Ks, fixed_ms):
        self.mem_init = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.metacomm = Metacommunity(n_comm, S, gamma, N_min, immig)
        self.metacomm.set_same_gaussian_alpha(mu, sigma)
        self.n_gen = n_gen
        self.epsilon = epsilon
        self.t_max = t_max
        self.n_select = n_select
        self.fixed_ms = fixed_ms
        try:
            self.score_func = score_func_dict[score_func_name]
        except KeyError:
            print(score_func_name + " not implemented yet")
        self.ws = gen_ws(score_func_name, S)
        if random_Ks:
            Ks = self.metacomm.set_same_random_Ks()
        else:
            Ks = np.ones(S)
        self.runned = False
        self.dataset = Dataset(data_to_get, S, self.ws, Ks)
        self.tLVs = []  # for debug only

    def run(self):
        if self.runned:
            print("Simulation already runned !")
            raise ValueError
        t_mute = 0
        t_LV = 0
        t_measure = 0
        t_select = 0
        self.dataset.measure_init(self.metacomm)
        for i in tqdm(range(self.n_gen)):
            if (i*10) % self.n_gen == 0:
                print_status(i, self.n_gen, self.mem_init)
            if i == 0:
                t_max = self.t_max*10
            else:
                t_max = self.t_max
            # mutations
            t1 = process_time()
            if self.fixed_ms:
                etas = self.metacomm.mutate_all_no_ms(self.epsilon)
            else:
                etas = self.metacomm.mutate_all(self.epsilon)
            # LV
            t2 = process_time()
            tLV_max = self.metacomm.equilibrate_all_LV(t_max)
            # selection
            t3 = process_time()
            selected = self.metacomm.select(
                self.n_select, self.score_func, self.ws)
            eta_selected = etas[selected[0]]
            # measure
            t4 = process_time()
            self.dataset.measure(self.metacomm)
            self.tLVs.append(tLV_max)  # for debug only, clean after
            # not super clean, have to change it one day
            self.dataset.measure_mutation(self.metacomm, eta_selected)
            # Time measure
            t5 = process_time()
            t_mute += t2-t1
            t_LV += t3-t2
            t_measure += t4-t3
            t_select += t5-t4
            # last iteraction
            if i == self.n_gen-1:
                self.dataset.measure_last(self.metacomm)
            # check divergence
            if self.metacomm.has_exploded():
                print("Simulation stopped at generation ", i)
                self.dataset.measure_last(self.metacomm)
                break
        t_total = t_mute + t_LV + t_measure + t_select
        print('T mutations = ', round(t_mute*100/t_total, 1), '%')
        print('T LV = ', round(t_LV*100/t_total, 1), '%')
        print('T selection = ', round(t_measure*100/t_total, 1), '%')
        print('T measure = ', round(t_select*100/t_total, 1), '%')
        print('T total =', round(t_total, 3))

        self.runned = True
        return self.dataset
