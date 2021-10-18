from simulation import Simu_artificial_select

# score_func_name : {'shannon', 'diversity', 'total_mass', 'random_selection',
# 'random_combination_pos', 'random_combination', 'random_deviation'}

n_comm = 10
S = 500
gamma = 0
N_min = 10**(-20)
immig = False
random_Ks = True
fixed_ms = False  # If true impose mu and sigma fixed (to a certain extand)
mu = 2.5
sigma = 0.5
n_gen = 5000
epsilon = 0.02
t_max = 300
n_select = 1
score_func_name = 'total_mass'
data_to_get = {'alpha_eig': True,
               'alpha_star_eig': False, 'Ns': True, 'mu_sigma': True,
               'mutation_direction': False, 'alpha': True,
               'coeff_abcd': False, 'alpha_init_last': False}

# dir = '../data/fixed_ms/'
# dir = '../tests/'
dir = '../data/alpha_full_Ks/'  # Le dossier doit déjà exister !!! à changer !

simu = Simu_artificial_select(n_comm, S, gamma, N_min, immig, mu, sigma,
                              n_gen, epsilon, t_max, n_select, score_func_name,
                              data_to_get, random_Ks, fixed_ms)


data = simu.run()
print("simu ok")
data.save(dir)
print("save ok")
data.plot()
