import numpy as np
cimport numpy as np
import math
import logging
from model import *

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logfile_handler = logging.FileHandler('mcmc.log', mode='w')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S')
logfile_handler.setFormatter(formatter)
logger.addHandler(logfile_handler)


def read_no_of_sites_taxa(file):
    str = file.readline()
    x1, x2 = map(int, str.split())
    cdef int N = x1
    cdef int M = x2
    return N, M


def construct_matrix(file, model):

    N = model.N
    M = model.M

    data_matrix_shape = (N, M)
    cdef np.ndarray X = np.zeros(data_matrix_shape)
    hard_sites_vector_shape = (N, 1)
    cdef np.ndarray H = np.zeros(hard_sites_vector_shape)
    cdef int Nh = 0
    for i in range(N):
        str = file.readline()
        for j in range(M):
            X[i][j] = str.split(" ")[j]
        try:
            if(str.split(" ")[M] == '*'):
                H[i][0] = 1
                Nh = Nh + 1
        except IndexError:
            pass
    return X, H, Nh


def compute_lifespan(model):

    lifespan_matrix_shape = (model.M, 1)
    cdef np.ndarray a = np.zeros(lifespan_matrix_shape)
    cdef np.ndarray b = np.zeros(lifespan_matrix_shape)

    """
    Zero-indexing in lifespan matrices. Reference code uses one-indexing for death site vector
    """

    for m in range(model.M):
        n = 0
        while (n < model.N and not model.X[model.rpi[n]][m]):
            n += 1
        if (n is model.N):
            print("Zero column at", m, ", continuing.")
            a[m] = 0
            b[m] = model.N
        else:
            a[m] = n
            n = model.N - 1
            while(n >= (a[m] - 1) and not model.X[model.rpi[n]][m]):
                n -= 1
            b[m] = n + 1
    return a, b


def init_log_prob(model):

    cdef int M = model.M

    log_prob_matrix_shape = (M, 1)
    cdef np.ndarray c = np.zeros(log_prob_matrix_shape)
    cdef np.ndarray d = np.zeros(log_prob_matrix_shape)
    for m in range(M):
        c[m] = math.log(0.01, math.e)
        d[m] = math.log(0.3, math.e)
    return c, d


def compute_trfa_count(model):

    cdef int N = model.N
    cdef int M = model.M
    cdef np.ndarray X = model.X
    cdef np.ndarray pi = model.pi
    #rpi = model.rpi
    cdef np.ndarray a = model.a
    cdef np.ndarray b = model.b

    n, m = [0] * 2
    cdef int tr0a = 0
    cdef int tr1a = 0
    cdef int fa0a = 0
    cdef int fa1a = 0
    count_matrix_shape = (M, 1)
    cdef np.ndarray tr0 = np.zeros(count_matrix_shape)
    cdef np.ndarray tr1 = np.zeros(count_matrix_shape)
    cdef np.ndarray fa0 = np.zeros(count_matrix_shape)
    cdef np.ndarray fa1 = np.zeros(count_matrix_shape)
    for m in range(M):
        t0, f0, t1, f1 = [0] * 4
        for n in range(N):
            if (a[m] <= pi[n] and pi[n] < b[m]):
                if (X[n][m]):
                    t1 += 1
                else:
                    f0 += 1
            else:
                if (X[n][m]):
                    f1 += 1
                else:
                    t0 += 1

        tr0a += t0
        tr1a += t1
        fa0a += f0
        fa1a += f1

        tr0[m] = t0
        tr1[m] = t1
        fa0[m] = f0
        fa1[m] = f1

    return tr0, tr1, fa0, fa1, tr0a, tr1a, fa0a, fa1a


def compute_loglike(model):

    cdef int M = model.M
    cdef np.ndarray c = model.c
    cdef np.ndarray d = model.d
    cdef np.ndarray tr0 = model.tr0
    cdef np.ndarray tr1 = model.tr1
    cdef np.ndarray fa0 = model.fa0
    cdef np.ndarray fa1 = model.fa1

    cdef double loglike = 0
    for m in range(M):
        cur_c = int(c[m])
        cur_d = int(d[m])
        temp_sum = int(tr0[m]) * math.log(1 - math.exp(cur_c),
                                          math.e) + int(fa0[m]) * (cur_d) + int(fa1[m]) * (cur_c) + int(tr1[m]) * math.log(1 - math.exp(cur_d), math.e)
        loglike += temp_sum
    return loglike


def read_model(file):

    mcmc_model = model()

    N, M = read_no_of_sites_taxa(file)
    mcmc_model.set_sites(N, M)

    X, H, Nh = construct_matrix(file, mcmc_model)
    mcmc_model.set_matrix(X, H, Nh)

    pi = np.arange(0, mcmc_model.N, 1)
    rpi = np.argsort(pi)
    mcmc_model.set_pi(pi, rpi)

    a, b = compute_lifespan(mcmc_model)
    mcmc_model.set_lifespan(a, b)

    c, d = init_log_prob(mcmc_model)
    mcmc_model.set_cd(c, d)

    tr0, tr1, fa0, fa1, tr0a, tr1a, fa0a, fa1a = compute_trfa_count(
        mcmc_model)
    mcmc_model.set_trfe(tr0, tr1, fa0, fa1, tr0a, tr1a, fa0a, fa1a)

    loglike = compute_loglike(mcmc_model)
    mcmc_model.set_loglike(loglike)
    
    return mcmc_model
