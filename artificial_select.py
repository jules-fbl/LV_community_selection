from simulation import Simu_artificial_select
import os

# possibles values for score_func_name :
# {'shannon', 'diversity', 'total_mass', 'random_selection',
# 'random_combination_pos', 'random_combination', 'random_deviation'}


def mkdir(dir):
    try:
        os.mkdir(dir)
    except Exception:
        pass


args = {'n_comm': 10, 'S': 200, 'gamma': 0, 'N_min': 10**(-20),
        'immig': False, 'mu': 20, 'sigma': 1, 'n_gen': 5000,
        'epsilon': 0.05, 't_max': 500,
        'n_select': 1, 'score_func_name': 'total_mass',
        'data_to_get': {'alpha_eig': True,
                        'alpha_star_eig': False, 'Ns': True, 'mu_sigma': True,
                        'mutation_direction': False, 'alpha': True,
                        'coeff_abcd': False, 'alpha_init_last': False},
        'random_Ks': True, 'fixed_ms': False}

dir = '../data/big_mu/'  # name of the directory to save
mkdir(dir)
f = open(dir+"args.txt", "w")
f.write(str(args))  # Save the arguments of th simulation
f.close()

simu = Simu_artificial_select(**args)

data = simu.run()
print("simu ok")
data.save(dir)
print("save ok")
data.plot()
