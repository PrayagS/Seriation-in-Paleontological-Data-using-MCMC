import time
import subprocess
import shlex
import os
from multiprocessing import Pool
import multiprocessing
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import random
import matplotlib.pyplot as plt
import seaborn as sns

f = open('Dataset/g10s10.txt', mode='r')


def generate_random_seed():
    seed = subprocess.check_output(
        'od -vAn -N1 -tu1 < /dev/urandom', shell=True, text=True)
    return seed


def run_chain(chain_index, old_seeds):
    seed = generate_random_seed()
    while (True):
        seed = generate_random_seed()
        if seed not in old_seeds:
            old_seeds.append(seed)
            break
        else:
            pass
    os.environ['GSL_RNG_SEED'] = seed.strip()
    subprocess.call('./mcmc ' + str(chain_index) +
                    ' < Dataset/g10s10.txt', shell=True, stdout=subprocess.DEVNULL)


def run_all_chains():
    manager = multiprocessing.Manager()
    old_seeds = manager.list()
    start = time.perf_counter()

    pool = Pool(processes=4)
    x = [(i, old_seeds) for i in range(100)]
    chains = pool.starmap(run_chain, x)
    pool.terminate()

    finish = time.perf_counter()

    print(round(finish - start, 2))


def choose_chains():
    neg_exp_loglik = []
    for chain_dir in os.listdir('Chains'):
        x = pd.read_csv('Chains/' + chain_dir + '/exp_data.csv')
        x = x.exp_loglik.to_list()
        neg_exp_loglik.append(x[0])

    min_neg_exp_loglik = min(neg_exp_loglik)

    y = []
    std_dev = np.std(neg_exp_loglik)
    # print(min_neg_exp_loglik)
    for x in neg_exp_loglik:
        if x > min_neg_exp_loglik - std_dev and x < min_neg_exp_loglik + std_dev:
            y.append(x)
    y.sort()

    chains_to_select = 9
    chosen_chains = []
    for z in y[:chains_to_select]:
        for chain_dir in os.listdir('Chains'):
            x = pd.read_csv('Chains/' + chain_dir + '/exp_data.csv')
            x = x.exp_loglik.to_list()
            if x[0] == z:
                chosen_chains.append(int(chain_dir.split('_')[1]))

    chosen_chains.sort()
    # print(chosen_chains)
    return chosen_chains


def xcorr(x, y, normed=True, detrend=False, maxlags=10):
    # Cross correlation of two signals of equal length
    # Returns the coefficients when normed=True
    # Returns inner products when normed=False
    # Usage: lags, c = xcorr(x,y,maxlags=len(x)-1)
    # Optional detrending e.g. mlab.detrend_mean

    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    if detrend:
        import matplotlib.mlab as mlab
        x = mlab.detrend_mean(np.asarray(x))  # can set your preferences here
        y = mlab.detrend_mean(np.asarray(y))

    c = np.correlate(x, y, mode='full')

    if normed:
        # this is the transformation function
        n = np.sqrt(np.dot(x, x) * np.dot(y, y))
        c = np.true_divide(c, n)

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    lags = np.arange(-maxlags, maxlags + 1)
    c = c[Nx - 1 - maxlags:Nx + maxlags]
    return lags, c
    # _, c = xcorr(pi_sites, pi_sites, maxlags=1)
    # print(c)


def compute_exp_cd(chains):
    c_sum = 0
    c_sum_chain = 0
    d_sum = 0
    d_sum_chain = 0
    for chain in chains:
        chain_index = "%02d" % chain
        f = open('Chains/chain_' + chain_index + '/chain_data.csv')
        c_sum = 0
        d_sum = 0
        for line in f.readlines():
            c_sum += float(line.split(',')[3].split(' ')[0].strip())
            d_sum += float(line.split(',')[4].split(' ')[0].strip())
        c_sum_chain += (c_sum / 1000)
        d_sum_chain += (d_sum / 1000)

    return c_sum_chain / 9, d_sum_chain / 9


def compute_exp_ages(chains):
    coeff_sum = 0
    coeff_sum_chain = 0
    for chain in chains:
        chain_index = "%02d" % chain
        f1 = open('Chains/chain_' + chain_index + '/chain_data.csv')
        coeff_sum = 0
        for line in f1.readlines():
            pi_chain = [int(i.strip())
                        for i in line.split(',')[2].split(' ')[:124]]
            # _, coeff = xcorr(pi_chain, np.arange(0, 124), maxlags=1)
            coeff = pearsonr(pi_chain, np.arange(0, 124))
            coeff_sum += coeff[0]
        coeff_sum_chain += (coeff_sum / 1000)

    return coeff_sum_chain / 9


def compute_pair_order_matrix(chains):
    po_matrix = np.zeros(shape=(124, 124))
    po_matrix_chain = np.zeros(shape=(124, 124))
    for chain in chains:
        chain_index = "%02d" % chain
        f = open('Chains/chain_' + chain_index + '/chain_data.csv')
        for line in f.readlines():
            pi_sites_chain = [int(i.strip())
                              for i in line.split(',')[2].split(' ')[:124]]
            generate_po_matrix(pi_sites_chain, po_matrix_chain)
        po_matrix_chain /= 1000
        po_matrix += po_matrix_chain
    po_matrix /= 9
    plot_po_matrix(po_matrix)


def generate_po_matrix(pi_sites, po_matrix):
    for i in range(124):
        for j in range(124):
            if i == j:
                po_matrix[i][j] += -1
            else:
                po_matrix[i][j] += int(pi_sites[i] < pi_sites[j])


def plot_po_matrix(po_matrix):
    sns.set()
    ax = sns.heatmap(po_matrix, vmin=0, vmax=1, cmap='Greys')
    plt.gca().invert_yaxis()
    plt.show()


if __name__ == "__main__":
    # run_all_chains()
    chains = choose_chains()
    compute_pair_order_matrix(chains)
