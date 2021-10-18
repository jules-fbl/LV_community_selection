import fire
import os
from simulation import Simu_artificial_select


def mkdir(dir):
    try:
        os.mkdir(dir)
    except Exception:
        pass


def run_simu(n_comm=9, S=100, gamma=0, N_min=10**(-20), immig=False, mu=3,
             sigma=0.1, n_gen=1000, epsilon=0.1, t_max=300, n_select=1,
             score_func_name='total_mass', get_eig=True, get_alpha=False,
             random_Ks=False, fixed_ms=False, alpha_il=False):

    data_to_get = {'alpha_eig': get_eig,
                   'alpha_star_eig': get_eig, 'Ns': True, 'mu_sigma': True,
                   'mutation_direction': False, 'alpha': get_alpha,
                   'coeff_abcd': (score_func_name == 'random_combination'),
                   'alpha_init_last': alpha_il}

    over_write = True

    # create directory to save data
    dir = "../data/%s/" % score_func_name
    mkdir(dir)

    dir += "gamma%a/" % gamma
    mkdir(dir)

    dir += "S%a/" % S
    mkdir(dir)

    dir += "eps%a_mu%a_sig%a/" % (epsilon, mu, sigma)
    mkdir(dir)

    dir += "ngen%a_T%a/" % (n_gen, t_max)
    mkdir(dir)

    dir += "ncomm%a-%a" % (n_comm, n_select)
    if immig:
        dir += '_immig'
    if random_Ks:
        dir += '_randKs'
    if fixed_ms:
        dir += '_fixms'
    dir += '/'

    try:
        os.mkdir(dir)
    except Exception:
        print("experiment already done with same params")
        if over_write:
            pass
        else:
            raise
    print(dir)

    simu = Simu_artificial_select(n_comm, S, gamma, N_min, immig, mu, sigma,
                                  n_gen, epsilon, t_max,
                                  n_select, score_func_name, data_to_get,
                                  random_Ks, fixed_ms)

    data = simu.run()
    data.save(dir)


if __name__ == '__main__':
    fire.Fire(run_simu)
