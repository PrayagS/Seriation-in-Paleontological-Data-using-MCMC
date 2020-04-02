import time
import subprocess
import shlex
import os
from multiprocessing import Pool
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import random

f = open('Dataset/g10s10.txt', mode='r')


def run_chain(chain_index):
    # seed = subprocess.check_output(
    #     'od -vAn -N1 -tu1 < /dev/urandom', shell=True, text=True)
    r = Random()
    r.seed(chain_index)
    os.environ['GSL_RNG_SEED'] = seed
    subprocess.call('./mcmc ' + str(chain_index) +
                    ' < Dataset/g10s10.txt', shell=True, stdout=subprocess.DEVNULL)


def run_all_chains():
    start = time.perf_counter()

    pool = Pool(processes=4)
    x = [i for i in range(100)]
    chains = pool.map(run_chain, x)
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
    print(y[:chains_to_select])
    chosen_chains = []
    for z in y[:chains_to_select]:
        for chain_dir in os.listdir('Chains'):
            x = pd.read_csv('Chains/' + chain_dir + '/exp_data.csv')
            x = x.exp_loglik.to_list()
            if x[0] == z:
                chosen_chains.append(int(chain_dir.split('_')[1]))
                print(x[0], z, int(chain_dir.split('_')[1]))

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


if __name__ == "__main__":
    run_all_chains()
    chains = choose_chains()
    print(len(chains))
    # f = open('Chains/chain_00/chain_data.csv')
    # for line in f.readlines():
    #     print(line.split(',')[2].split())
