import numpy as np
import math
from model import model


def read_no_of_sites_taxa(f):
    str = f.readline()
    N, M = map(int, str.split())
    return N, M


def construct_matrix(f, N, M):
    data_matrix_shape = (N, M)
    X = np.zeros(data_matrix_shape)
    hard_sites_vector_shape = (N, 1)
    H = np.zeros(hard_sites_vector_shape)
    Nh = 0
    for i in range(N):
        str = f.readline()
        for j in range(M):
            X[i][j] = str.split(" ")[j]
        try:
            if(str.split(" ")[M] == '*'):
                H[i][0] = 1
                Nh = Nh + 1
        except IndexError:
            pass
    return X, H, Nh


def compute_lifespan(f, X, N, M, rpi):
    lifespan_matrix_shape = (M, 1)
    a = np.zeros(lifespan_matrix_shape)
    b = np.zeros(lifespan_matrix_shape)

    """
    Zero-indexing in lifespan matrices. Reference code uses one-indexing for death site vector
    """

    for m in range(M):
        n = 0
        while (n < N and not X[rpi[n]][m]):
            n += 1
        if (n is N):
            print("Zero column at", m, ", continuing.")
            a[m] = 0
            b[m] = N
        else:
            a[m] = n
            n = N-1
            while(n >= (a[m]-1) and not X[rpi[n]][m]):
                n -= 1
            b[m] = n+1
    return a, b


def init_log_prob(M):
    log_prob_matrix_shape = (M, 1)
    c = np.zeros(log_prob_matrix_shape)
    d = np.zeros(log_prob_matrix_shape)
    for m in range(M):
        c[m] = math.log(0.01, math.e)
        d[m] = math.log(0.3, math.e)
    return c, d


def compute_trfa_count(X, N, M, pi, rpi, a, b):
    n, m = [0]*2
    tr0a, tr1a, fa0a, fa1a = [0]*4
    count_matrix_shape = (M, 1)
    tr0 = np.zeros(count_matrix_shape)
    tr1 = np.zeros(count_matrix_shape)
    fa0 = np.zeros(count_matrix_shape)
    fa1 = np.zeros(count_matrix_shape)
    for m in range(M):
        t0, f0, t1, f1 = [0]*4
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


def compute_loglike(M, c, d, tr0, tr1, fa0, fa1):
    loglike = 0.0
    for m in range(M):
        cur_c = c[m]
        cur_d = d[m]
        loglike += tr0[m]*math.log(1-math.exp(cur_c),
                                   math.e) + fa0[m]*(cur_d) + fa1
        [m]*(cur_c) + tr1[m]*math.log(1-math.exp(cur_d), math.e)
        print(loglike)
    return loglike


if __name__ == "__main__":
    f = open("Dataset/g10s10.txt", 'r')
    mcmc_model = model()
    N, M = read_no_of_sites_taxa(f)
    mcmc_model.N = N
    mcmc_model.M = M

    X, H, Nh = construct_matrix(f, mcmc_model.N, mcmc_model.M)
    mcmc_model.X = X
    mcmc_model.H = H
    mcmc_model.Nh = Nh

    mcmc_model.pi = np.arange(0, mcmc_model.N, 1)
    mcmc_model.rpi = np.argsort(mcmc_model.pi)

    a, b = compute_lifespan(f, X, N, M, mcmc_model.rpi)
    mcmc_model.a = a
    mcmc_model.b = b

    c, d = init_log_prob(M)
    mcmc_model.c = c
    mcmc_model.d = d

    tr0, tr1, fa0, fa1, tr0a, tr1a, fa0a, fa1a = compute_trfa_count(
        X, N, M, mcmc_model.pi, mcmc_model.rpi, a, b)
    mcmc_model.tr0 = tr0
    mcmc_model.tr1 = tr1
    mcmc_model.fa0 = fa0
    mcmc_model.fa1 = fa1
    mcmc_model.tr0a = tr0a
    mcmc_model.tr1a = tr1a
    mcmc_model.fa0a = fa0a
    mcmc_model.fa1a = fa1a

    mcmc_model.loglike = compute_loglike(M, c, d, tr0, tr1, fa0, fa1)
    # print(mcmc_model.loglike)

    f.close()
