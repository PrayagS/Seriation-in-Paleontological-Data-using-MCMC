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


def generate_random_seed():
    """
    Generates a random seed for the next chain from /dev/urandom
    :return seed - 1 byte signed integer
    """
    seed = subprocess.check_output(
        'od -vAn -N1 -tu1 < /dev/urandom', shell=True, text=True)
    return seed


def run_chain(chain_index, old_seeds, dataset):
    """
    :param chain_index: index number of the current chain
    :param old_seeds: list containing seeds already used
    :param dataset: dataset file to sample data from
    """
    f = open(str(dataset), mode='r')
    seed = generate_random_seed()
    # Generate seed until you find an unique one
    while (True):
        seed = generate_random_seed()
        if seed not in old_seeds:
            old_seeds.append(seed)
            break
        else:
            pass
    # Set the seed for the next chain
    os.environ['GSL_RNG_SEED'] = seed.strip()
    # Run the sampling routine
    subprocess.call('./mcmc ' + str(chain_index) +
                    ' < ' + str(dataset), shell=True, stdout=subprocess.DEVNULL)


def run_all_chains(dataset):
    """
    :param dataset: dataset file to sample data from
    Runs all 100 chains where 4 of them run in parallel
    """

    # Multiprocessing to run multiple chains in parallel
    manager = multiprocessing.Manager()
    old_seeds = manager.list()
    start = time.perf_counter()

    # Limit the number of parallel chains running to the number of CPU cores
    pool = Pool(processes=8)
    x = [(i, old_seeds, dataset) for i in range(100)]
    chains = pool.starmap(run_chain, x)
    pool.terminate()

    finish = time.perf_counter()

    print(round(finish - start, 2))


def choose_chains(chains_selected):
    """
    Choose few number of chains which are within one standard deviation
    of the best chain into account from the 100 total chains
    """
    neg_exp_loglik = []
    for chain_dir in os.listdir('Chains'):
        x = pd.read_csv('Chains/' + chain_dir + '/exp_data.csv')
        x = x.exp_loglik.to_list()
        neg_exp_loglik.append(x[0])

    min_neg_exp_loglik = min(neg_exp_loglik)

    y = []
    std_dev = np.std(neg_exp_loglik)
    for x in neg_exp_loglik:
        if x > min_neg_exp_loglik - std_dev and x < min_neg_exp_loglik + std_dev:
            y.append(x)
    y.sort()

    chosen_chains = []
    for z in y[:chains_selected]:
        for chain_dir in os.listdir('Chains'):
            x = pd.read_csv('Chains/' + chain_dir + '/exp_data.csv')
            x = x.exp_loglik.to_list()
            if x[0] == z:
                chosen_chains.append(int(chain_dir.split('_')[1]))

    chosen_chains.sort()
    return chosen_chains


def compute_exp_cd(chains, chains_selected):
    """
    :param chains: the chosen chains to analyse samples from
    :return exp_c: Expected probability of false positive,
            exp_d: Expected probability of false negative
    Computes the expected value error probabilities
    """
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

    exp_c = c_sum_chain / chains_selected
    exp_d = d_sum_chain / chains_selected
    return exp_c, exp_d


def compute_exp_ages(chains, chains_selected, sites):
    """
    :param chains: the chosen chains to analyse samples from
    :param chains_selected: No. of chains selected from the 100 total chains
    :param sites: No. of sites in the dataset
    :return exp_corr_MN: Expected correlation of the order with the order of MN ages
    Computes the expected correlation of the predicted seriation
    """
    coeff_sum = 0
    coeff_sum_chain = 0
    for chain in chains:
        chain_index = "%02d" % chain
        f1 = open('Chains/chain_' + chain_index + '/chain_data.csv')
        coeff_sum = 0
        for line in f1.readlines():
            pi_chain = [int(i.strip())
                        for i in line.split(',')[2].split(' ')[:sites]]
            # _, coeff = xcorr(pi_chain, np.arange(0, sites), maxlags=1)
            coeff = pearsonr(pi_chain, np.arange(0, sites))
            coeff_sum += coeff[0]
        coeff_sum_chain += (coeff_sum / 1000)

    exp_corr_MN = coeff_sum_chain / chains_selected
    return exp_corr_MN


def compute_pair_order_matrix(chains, chains_selected, sites):
    """
    :param chains: the chosen chains to analyse samples from
    :param chains_selected: No. of chains selected from the 100 total chains
    :param sites: No. of sites in the dataset
    :return po_matrix: Pair order matrix of the sites
    Computes the pair order matrix of the sites which denotes relative ordering of the sites
    """
    po_matrix = np.zeros(shape=(sites, sites))
    po_matrix_chain = np.zeros(shape=(sites, sites))
    for chain in chains:
        chain_index = "%02d" % chain
        f = open('Chains/chain_' + chain_index + '/chain_data.csv')
        for line in f.readlines():
            pi_sites_chain = [int(i.strip())
                              for i in line.split(',')[2].split(' ')[:sites]]
            generate_po_matrix(pi_sites_chain, po_matrix_chain, sites)
        po_matrix_chain /= 1000
        po_matrix += po_matrix_chain
    po_matrix /= chains_selected
    return po_matrix


def generate_po_matrix(pi_sites, po_matrix, sites):
    """
    :param pi_sites: pi vector for a given sample
    :param sites: No. of sites in the dataset
    Updates the po_matrix for each sample
    """
    for i in range(sites):
        for j in range(sites):
            if i == j:
                po_matrix[i][j] += -1
            else:
                po_matrix[i][j] += int(pi_sites[i] < pi_sites[j])


def plot_po_matrix(po_matrix):
    """
    :param po_matrix: Pair order matrix of the sites
    Plots the pair order matrix of the sites which denotes relative ordering of the sites
    """
    sns.set()
    ax = sns.heatmap(po_matrix, vmin=0, vmax=1, cmap='Greys')
    plt.gca().invert_yaxis()
    plt.show()


def construct_matrix(dataset, sites, taxa):
    """
    :param dataset: dataset file to sample data from
    :param sites: No. of sites in the dataset
    :param taxa: No. of taxa in the dataset
    Computes the occurence matrix X from a given data file
    """
    f = open(str(dataset))
    X = np.zeros(shape=(sites, taxa))
    line = f.readline()
    for i in range(sites):
        line = f.readline()
        for j in range(taxa):
            X[i][j] = line.split(" ")[j]
    return X


def plot_data_matrix(X):
    """
    :param X: Occurence matrix X
    Plots the occurence matrix X
    """
    plt.imshow(X, cmap='Greys')
    plt.gca().invert_yaxis()
    plt.show()


def compute_exp_pi(chains, sites, chains_selected):
    """
    :param chains: the chosen chains to analyse samples from
    :param chains_selected: No. of chains selected from the 100 total chains
    :param sites: No. of sites in the dataset
    :return exp_pi: Expected ordering of the sites
    Computes the expected ordering of the sites
    """
    pi_sum = np.zeros(shape=(sites,))
    pi_sum_chain = np.zeros(shape=(sites,))
    for chain in chains:
        chain_index = "%02d" % chain
        f1 = open('Chains/chain_' + chain_index + '/chain_data.csv')
        pi_sum = 0
        for line in f1.readlines():
            pi_chain = [int(i.strip())
                        for i in line.split(',')[2].split(' ')[:sites]]
            # _, coeff = xcorr(pi_chain, np.arange(0, sites), maxlags=1)
            pi_sum_chain += pi_chain
        pi_sum_chain /= 1000
        pi_sum += pi_sum_chain
    exp_pi = pi_sum / chains_selected
    return exp_pi


def compute_exp_a(chains, chains_selected, taxa):
    """
    :param chains: the chosen chains to analyse samples from
    :param chains_selected: No. of chains selected from the 100 total chains
    :param taxa: No. of taxa in the dataset
    :return exp_a: Expected birth sites of each taxa
    Computes the expected birth sites of each taxa to then order them based on that
    """
    a_sum = np.zeros(shape=(taxa,))
    a_sum_chain = np.zeros(shape=(taxa,))
    for chain in chains:
        chain_index = "%02d" % chain
        f = open('Chains/chain_' + chain_index + '/chain_data.csv')
        pi_sum = 0
        for line in f.readlines():
            a_chain = [int(i.strip())
                       for i in line.split(',')[0].split(' ')[:taxa]]
            a_sum_chain += a_chain
        a_sum_chain /= 1000
        a_sum += a_sum_chain
    exp_a = a_sum / chains_selected
    return exp_a


def plot_new_data_matrix(chains, chains_selected, dataset, sites, taxa):
    """
    :param chains: the chosen chains to analyse samples from
    :param sites: No. of sites in the dataset
    :param taxa: No. of taxa in the dataset
    Orders the sites based on exp_pi and taxa based on exp_a and then plot the new occurence matrix
    """
    # Shuffle the rows i.e. sites
    exp_pi = compute_exp_pi(chains, sites, chains_selected)
    rpi = np.argsort(exp_pi)
    X = construct_matrix(dataset, sites, taxa)
    idx = np.empty_like(rpi)
    idx[rpi] = np.arange(len(rpi))
    X = X[idx, :]

    # Shuffle the columns i.e. taxa
    exp_a = compute_exp_a(chains, chains_selected, taxa)
    ra = np.argsort(exp_a)

    Y = np.zeros(shape=(sites, taxa))
    i = 0
    for taxon in ra:
        Y[:, i] = X[:, taxon]
        i += 1
    plot_data_matrix(Y)


def plot_taxa_occurence_probability_matrix(chains, chains_selected, sites, taxa):
    """
    :param chains: the chosen chains to analyse samples from
    :param chains_selected: No. of chains selected from the 100 total chains
    :param sites: No. of sites in the dataset
    :param taxa: No. of taxa in the dataset
    :return taxa_occurence_probability_matrix: Matrix X such that X_ij = Pr(taxa i alive at site j)
    Plots the probability of occurence of each taxa on each given site
    """

    # Construct matrix X such that X_ij = Pr(taxa i alive at site j)
    X_sum_chain = np.zeros(shape=(sites, taxa))
    X_sum = np.zeros(shape=(sites, taxa))
    for chain in chains:
        chain_index = "%02d" % chain
        f = open('Chains/chain_' + chain_index + '/chain_data.csv')
        for line in f.readlines():
            a_chain = [int(i.strip())
                       for i in line.split(',')[0].split(' ')[:taxa]]
            b_chain = [int(i.strip())
                       for i in line.split(',')[1].split(' ')[:taxa]]
            for i, a in enumerate(a_chain):
                for j in range(sites):
                    X_sum_chain[j][i] += int(j >= a and j <= b_chain[i])
        X_sum_chain /= 1000
        X_sum += X_sum_chain
    X_sum /= chains_selected

    # Shuffle the sites in matrix according to exp_pi
    exp_pi = compute_exp_pi(chains, sites, chains_selected)
    rpi = np.argsort(exp_pi)
    idx = np.empty_like(rpi)
    idx[rpi] = np.arange(len(rpi))
    X_sum = X_sum[idx, :]

    # Shuffle the taxa in matrix according to exp_a
    exp_a = compute_exp_a(chains, chains_selected, taxa)
    ra = np.argsort(exp_a)

    taxa_occurence_probability_matrix = np.zeros(shape=(sites, taxa))
    i = 0
    for taxon in ra:
        taxa_occurence_probability_matrix[:, i] = X_sum[:, taxon]
        i += 1

    return taxa_occurence_probability_matrix


def plot_false_taxa_occurence_probability(chains, chains_selected, sites, taxa):
    """
    :param chains: the chosen chains to analyse samples from
    :param chains_selected: No. of chains selected from the 100 total chains
    :param sites: No. of sites in the dataset
    :param taxa: No. of taxa in the dataset
    :return false_taxa_occurence_probability_matrix: Matrix X such that X_ij = Pr(taxa i not alive at site j)
    Plots the probability of false occurence of each taxa on each given site
    """

    # Construct matrix X such that X_ij = Pr(taxa i alive at site j)
    X_sum_chain = np.zeros(shape=(sites, taxa))
    X_sum = np.zeros(shape=(sites, taxa))
    for chain in chains:
        chain_index = "%02d" % chain
        f = open('Chains/chain_' + chain_index + '/chain_data.csv')
        for line in f.readlines():
            a_chain = [int(i.strip())
                       for i in line.split(',')[0].split(' ')[:taxa]]
            b_chain = [int(i.strip())
                       for i in line.split(',')[1].split(' ')[:taxa]]
            for i, a in enumerate(a_chain):
                for j in range(sites):
                    X_sum_chain[j][i] += int(j < a or j > b_chain[i])
        X_sum_chain /= 1000
        X_sum += X_sum_chain
    X_sum /= chains_selected

    # Shuffle the sites in matrix according to exp_pi
    exp_pi = compute_exp_pi(chains, sites, chains_selected)
    rpi = np.argsort(exp_pi)
    idx = np.empty_like(rpi)
    idx[rpi] = np.arange(len(rpi))
    X_sum = X_sum[idx, :]

    # Shuffle the taxa in matrix according to exp_a
    exp_a = compute_exp_a(chains, chains_selected, taxa)
    ra = np.argsort(exp_a)

    false_taxa_occurence_probability_matrix = np.zeros(shape=(sites, taxa))
    i = 0
    for taxon in ra:
        false_taxa_occurence_probability_matrix[:, i] = X_sum[:, taxon]
        i += 1

    return false_taxa_occurence_probability_matrix


def plot_false_ones_probability(chains, chains_selected, dataset, sites, taxa):
    """
    :param chains: the chosen chains to analyse samples from
    :param chains_selected: No. of chains selected from the 100 total chains
    :param sites: No. of sites in the dataset
    :param taxa: No. of taxa in the dataset
    :return false_ones_probability_matrix: Matrix Y such that Y_ij = Pr(X_ij=1 is false)
    Plots the probability of an occurence in the occurence matrix to be false
    """
    X = construct_matrix(dataset, sites, taxa)

    X_sum_chain = np.zeros(shape=(sites, taxa))
    X_sum = np.zeros(shape=(sites, taxa))
    for chain in chains:
        chain_index = "%02d" % chain
        f = open('Chains/chain_' + chain_index + '/chain_data.csv')
        for line in f.readlines():
            a_chain = [int(i.strip())
                       for i in line.split(',')[0].split(' ')[:taxa]]
            b_chain = [int(i.strip())
                       for i in line.split(',')[1].split(' ')[:taxa]]
            for i, a in enumerate(a_chain):
                for j in range(sites):
                    if X[j][i] == 1:
                        if j >= a and j <= b_chain[i]:
                            X_sum_chain[j][i] += 0
                        else:
                            X_sum_chain[j][i] += 1
        X_sum_chain /= 1000
        X_sum += X_sum_chain
    X_sum /= chains_selected

    exp_pi = compute_exp_pi(chains, sites, chains_selected)
    rpi = np.argsort(exp_pi)
    idx = np.empty_like(rpi)
    idx[rpi] = np.arange(len(rpi))
    X_sum = X_sum[idx, :]

    exp_a = compute_exp_a(chains, chains_selected, taxa)
    ra = np.argsort(exp_a)

    false_ones_probability_matrix = np.zeros(shape=(124, 139))
    i = 0
    for taxon in ra:
        false_ones_probability_matrix[:, i] = X_sum[:, taxon]
        i += 1
    return false_ones_probability_matrix


def compute_exp_data():
    dataset_files = [
        {'name': 'Dataset/g10s10.txt', 'nt': 10, 'ns': 10,
            'sites': 124, 'taxa': 139, 'chains': 8},
        {'name': 'Dataset/g5s5.txt', 'nt': 5, 'ns': 5,
            'sites': 273, 'taxa': 202, 'chains': 2},
        {'name': 'Dataset/g10s2.txt', 'nt': 10, 'ns': 2,
            'sites': 501, 'taxa': 139, 'chains': 2},
        {'name': 'Dataset/g2s2.txt', 'nt': 2, 'ns': 2,
            'sites': 526, 'taxa': 296, 'chains': 4},
    ]
    df = pd.DataFrame(dataset_files, columns=[
                      'nt', 'ns', 'sites', 'taxa', 'chains'])
    exp_c_arr = np.zeros(shape=(4,), dtype='float32')
    exp_d_arr = np.zeros(shape=(4,), dtype='float32')
    exp_corr_arr = np.zeros(shape=(4,), dtype='float32')
    for i, file in enumerate(dataset_files):
        chains_selected = file['chains']
        sites = file['sites']
        taxa = file['taxa']
        run_all_chains(file['name'])
        chosen_chains = choose_chains(chains_selected)
        exp_c, exp_d = compute_exp_cd(chosen_chains, chains_selected)
        print(i, exp_c, exp_d)
        exp_c_arr[i] = exp_c
        exp_d_arr[i] = exp_d
        exp_corr = compute_exp_ages(chosen_chains, chains_selected, sites)
        print(i, exp_corr)
        exp_corr_arr[i] = exp_corr
    df = df.assign(Ec=pd.Series(exp_c_arr).values)
    df = df.assign(Ed=pd.Series(exp_d_arr).values)
    df = df.assign(Ecorr=pd.Series(exp_corr_arr).values)
    df.to_csv('exp_data.csv', index=False)
    df = pd.read_csv('exp_data.csv')
    print(df.head())


if __name__ == "__main__":
    # run_all_chains('Dataset/g10s10.txt')
    # chains = choose_chains(9)

    # po_matrix = compute_pair_order_matrix(chains, 8, 124)
    # plot_po_matrix(po_matrix)

    # plot_new_data_matrix(chains, 8, 124, 139)

    # taxa_occurence_probability_matrix = plot_taxa_occurence_probability_matrix(
    # chains, 8, 124, 139)
    # plot_po_matrix(taxa_occurence_probability_matrix)

    # false_ones_probability_matrix = plot_false_ones_probability(
    #     chains, 9, 124, 139)
    # plot_po_matrix(false_ones_probability_matrix)

    # compute_exp_data()
    run_all_chains('Dataset/g2s2.txt')
    chosen_chains = choose_chains(4)
    exp_c, exp_d = compute_exp_cd(chosen_chains, 4)
    print(exp_c, exp_d)
    exp_c_arr[i] = exp_c
    exp_d_arr[i] = exp_d
    exp_corr = compute_exp_ages(chosen_chains, 4, 526)
    print(exp_corr)
