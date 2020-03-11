import numpy as np
import logging
import random
import math
from read_model import compute_loglike

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logfile_handler = logging.FileHandler('mcmc.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logfile_handler.setFormatter(formatter)
logger.addHandler(logfile_handler)

LOGEPSILON = -32.236191301916641

# CHANGED

# iteration of samples; each block of 10 samples

# mcmc_sample


def sample(model, r):
    count, cc, cd, cab, cpi1, cpi20, cpi21, cpi3 = [0]*8

    p = []
    for i in range(10):
        cc += samplec(model)
        cd += sampled(model)
        cab += sampleab(model, r)

        cpi21 += samplepi2(model, 1)
        for j in range(5):
            cpi1 += samplepi1(model)
            cpi20 += samplepi2(model, 0)
            cpi3 += samplepi3(model, p)

    count += 1

    return cc + cd + cab + cpi1 + cpi20 + cpi21 + cpi3


# mcmc_samplec

def samplec(model, r):
    # Updates c with MH step. Returns number of accepted samples
    y = model.c[0]
    samplebeta(y, model.fa1a, model.tr0a, math.log(
        0.001, math.e), math.log(0.1, math.e))
    for m in range(model.M):
        model.c[m] = y
    return 1


# mcmc_samplebeta


def samplebeta(x, a, b, low, high, r):
    y = r.betavariate(1.0 + a, 1.0 + b)
    if y > 0.0:
        y = math.log(y)
        if(low <= y and y <= high):
            x = y

    return x


def sampled(model):
    # Updates d with MH step. Returns number of accepted samples
    y = model.d[0]
    samplebeta(y, model.fa0a, model.tr1a, math.log(
        0.2, math.e), math.log(0.8, math.e))
    for m in range(model.M):
        model.d[m] = y
    return 1


def sampleab(model, r):
    count = 0
    t0, f0, t1, f1 = [0]*4
    cur_c, cur_d = [0.0]*2
    v = []

    matrix_shape = (model.N+1, 1)
    q = np.zeros(matrix_shape)
    dt0 = np.zeros(matrix_shape)
    df0 = np.zeros(matrix_shape)
    dt1 = np.zeros(matrix_shape)
    df1 = np.zeros(matrix_shape)

    for m in range(model.M):
        v = model.X[:, m]
        permute(v, model.rpi, model.N)

        t = model.a[m]
        t0 = model.tr0[m]
        f0 = model.fa0[m]
        t1 = model.tr1[m]
        f1 = model.fa1[m]

        cur_c = model.c[m]
        cur_d = model.d[m]

        t0, f0, t1, f1 = aux(
            model, v, model.b[m], cur_c, cur_d, t, t0, f0, t1, f1, q, dt0, df0, dt1, df1, r)
        if t != model.a[m]:
            model.a[m] = t
            count += 1

        v.reverse()

        t = model.N - model.b[m]
        t0, f0, t1, f1 = aux(
            model, v, model.N - model.a[m], cur_c, cur_d, t, t0, f0, t1, f1, q, dt0, df0, dt1, df1, r)
        model.tr0[m] = t0
        model.fa0[m] = f0
        model.tr1[m] = t1
        model.f1[m] = f1
        if(t != model.N - model.b[m]):
            moodel.b[m] = model.N - t
            count += 1

        model.tr0a = 0
        model.fa0a = 0
        model.tr1a = 0
        model.fa1a = 0

        for m in range(model.M):
            model.tr0a += model.tr0[m]
            model.fa0a += model.fa0[m]
            model.tr1a += model.tr1[m]
            model.fa1a += model.fa1[m]

        model.loglike = compute_loglike(model)

        return count


def aux(model, x, b, c, d, a, t0, f0, t1, f1, q, dt0, df0, dt1, df1, r):

    q[a] = 0.0

    cc = math.log(1.0 - math.exp(c))
    dd = math.log(1.0 - math.exp(d))

    for i in reversed(range(a)):
        if(x[i]):
            dt0[i] = dt0[i+1]
            df0[i] = df0[i+1]
            dt1[i] = dt1[i+1] + 1
            df1[i] = df1[i+1] - 1
        else:
            dt0[i] = dt0[i+1] - 1
            df0[i] = df0[i+1] + 1
            dt1[i] = dt1[i+1]
            df1[i] = df1[i+1]

    i = a+1
    while(i <= b):
        if(x[i-1]):
            dt0[i] = dt0[i-1]
            df0[i] = df0[i-1]
            dt1[i] = dt1[i-1] - 1
            df1[i] = df1[i-1] + 1
        else:
            dt0[i] = dt0[i-1] + 1
            df0[i] = df0[i-1] - 1
            dt1[i] = dt1[i-1]
            df1[i] = df1[i-1]
        i += 1

    for i in range(b):
        q[i] = dt0[i] * cc + df0[i] * d + dt1[i] * dd + df1[i] * c

    a = randompick(logtop(q, b+1), b+1, r)
    t0 += dt0[a]
    f0 += df0[a]
    t1 += dt1[a]
    f1 += df1[a]

    return t0, f0, t1, f1


def randompick(p, n, r):
    # Pick one sample of length n from vector p
    i = 0
    x = r.uniform(0, 1) - p[0]
    while(x > 0.0 and i < n-1):
        i += 1
        x -= p[i]
    return i


def logtop(p, n):
    z = p[0]
    for i in range(1, n):
        if(p[i] > z):
            z = p[i]
    x = 0.0
    for i in range(n):
        y = math.exp(math.max(LOGEPSILON, p[i]-z))
        p[i] = y
        x += y
    for i in range(n):
        p[i] = p[i] / x
    return p


def permute(A, P, n):

    # For each element of P
    for i in range(n):
        next = i

        # Check if it is already
        # considered in cycle
        while (P[next] >= 0):

            # Swap the current element according
            # to the permutation in P
            t = A[i]
            A[i] = A[P[next]]
            A[P[next]] = t

            temp = P[next]

            # Subtract n from an entry in P
            # to make it negative which indicates
            # the corresponding move
            # has been performed
            P[next] -= n
            next = temp


def samplepi2(model, swp):
    if swp != 0:
        i = random.randrange(0, model.N)
        j = random.randrange(0, model.N-1)
        if j >= i:
            j += 1
        else:
            n = i
            i = j
            j = n
    else:
        i = random.randrange(0, model.N-1)
        j = i+1

    m = 0
    for n in range(i, j+1):
        m += model.H[model.rpi[n]]
        if m > 1:
            return 0

    inc1 = random.randrange(0, 2)
    inc2 = random.randrange(0, 2)

    delta = 0.0

    for m in range(model.M):
        dt0, df0, dt1, df1 = [0]*4
        cur_a = model.a[m]
        cur_b = model.b[m]
        ain = ininterval(cur_a, i, j+1, inc1, inc2)
        binn = ininterval(cur_b, i, j+1, inc1, inc2)
        if ain and !binn:
            for n in range(i, a):
                if model.X[model.rpi[n]][m]:
                    dt1 += 1
                    df1 -= 1
                else:
                    dt0 -= 1
                    df0 += 1
            for n in range(a, j+1):
                if model.X[model.rpi[n]][m]:
                    dt1 -= 1
                    df1 += 1
                else:
                    dt0 += 1
                    df0 -= 1
        elif (!ain and binn):
            for n in range(i, b):
                if model.X[model.rpi[n]][m]:
                    dt1 -= 1
                    df1 += 1
                else:
                    dt0 += 1
                    df0 -= 1
            for n in range(b, j+1):
                if model.X[model.rpi[n]][m]:
                    dt1 += 1
                    df1 -= 1
                else:
                    dt0 -= 1
                    df0 += 1

        cur_c = model.c[m]
        cur_d = model.d[m]

        delta += dt0 * math.log(1-math.exp(cur_c), math.e) * \
            df0*d + dt1*math.log(1.0 - math.exp(d), math.e) + df1*c
    while(True):
        k = random.random()
        if k != 0.0:
            break
    if delta >= 0.0 or delta > log(k):
        for m in range(model.M):
            cur_a = model.a[m]
            cur_b = model.b[m]
            ain = ininterval(cur_a, i, j+1, inc1, inc2)
            binn = ininterval(cur_b, i, j+1, inc1, inc2)
            if ain and !binn:
                model.a[m] = i+j+1-cur_a
            elif !ain and binn:
                model.b[m] = i+j+1-cur_b
            elif ain and binn:
                model.b[m] = i+j+1 - a
                model.a[m] = i+j+1 - b

        for n in range(i, (i+1)/2 + 1):
            m = mode.rpi[n]
            model.rpi[n] = model.rpi[i+j-n]
            model.rpi[i+j-n] = m

        model.pi = np.argsort(model.rpi)
        model.loglike += delta
        model.tr0, model.tr1, model.fa0, model.fa1, model.tr0a, model.tr1a, model.fa0a, model.fa1a = compute_trfa_count(
            model)
        return 1
    return 0


def ininterval(i, a, b, inc1, inc2):
    if a > b:
        rr = a
        a = b
        b = rr
    if inc1:
        rr = (a <= i)
    else:
        rr = (a < i)
    if rr:
        if inc2:
            rr = (i <= b)
        else:
            rr = (i < b)

    return rr
