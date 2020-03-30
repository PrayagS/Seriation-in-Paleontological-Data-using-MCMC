import numpy as np
cimport numpy as np
import logging
import random
import math
from time import sleep
from read_model import compute_loglike, compute_trfa_count

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logfile_handler = logging.FileHandler('mcmc.log', mode='w')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S')
logfile_handler.setFormatter(formatter)
logger.addHandler(logfile_handler)

LOGEPSILON = -32.236191301916641

# CHANGED

# iteration of samples; each block of 10 samples

# mcmc_sample


def sample(model, r):
    cdef int count = 0
    cdef int cc = 0
    cdef int cd = 0
    cdef int cab = 0
    cdef int cpi1 = 0
    cdef int cpi20 = 0
    cdef int cpi21 = 0
    cdef int cpi3 = 0

    cdef list p = [0 for i in range(model.N)]
    for i in range(10):
        cc += samplec(model, r)
        cd += sampled(model, r)
        cab += sampleab(model, r)

        cpi21 += samplepi2(model, 1, r)
        for j in range(5):
            cpi1 += samplepi1(model, r)
            cpi20 += samplepi2(model, 0, r)
            cpi3 += samplepi3(model, p, r)

    count += 1

    return cc + cd + cab + cpi1 + cpi20 + cpi21 + cpi3


# mcmc_samplec

def samplec(model, r):
    # Updates c with MH step. Returns number of accepted samples
    cdef double y = np.double(model.c[0])
    samplebeta(y, model.fa1a, model.tr0a, math.log(
        0.001, math.e), math.log(0.1, math.e), r)
    for m in range(model.M):
        model.c[m] = y
    return 1


# mcmc_samplebeta


def samplebeta(double x, int a, int b, double low, double high, r):
    cdef double y = r.betavariate(1.0 + a, 1.0 + b)
    if y > 0.0:
        y = math.log(y)
        if(low <= y and y <= high):
            x = y

    return x


def sampled(model, r):
    # Updates d with MH step. Returns number of accepted samples
    cdef double y = np.double(model.d[0])
    samplebeta(y, model.fa0a, model.tr1a, math.log(
        0.2, math.e), math.log(0.8, math.e), r)
    for m in range(model.M):
        model.d[m] = y
    return 1


def sampleab(model, r):
    cdef int count = 0
    cdef int t0 = 0
    cdef int f0 = 0
    cdef int t1 = 0
    cdef int f1 = 0
    cdef double cur_c = 0
    cdef double cur_d = 0
    cdef np.ndarray v

    matrix_shape = (model.N + 1, 1)
    cdef np.ndarray q = np.zeros(matrix_shape)
    cdef np.ndarray dt0 = np.zeros(matrix_shape)
    cdef np.ndarray df0 = np.zeros(matrix_shape)
    cdef np.ndarray dt1 = np.zeros(matrix_shape)
    cdef np.ndarray df1 = np.zeros(matrix_shape)

    # sleep(1)
    cdef int t = 0 
    for m in range(model.M):
        v = model.X[:, m]
        permute(v, model.rpi, model.N)

        t = int(model.a[m])
        t0 = int(model.tr0[m])
        f0 = int(model.fa0[m])
        t1 = int(model.tr1[m])
        f1 = int(model.fa1[m])

        cur_c = np.double(model.c[m])
        cur_d = np.double(model.d[m])

        # sleep(1)
        t0, f0, t1, f1 = aux(
            model, v, int(model.b[m]), cur_c, cur_d, t, t0, f0, t1, f1, q, dt0, df0, dt1, df1, r)
        if t != model.a[m]:
            model.a[m] = t
            count += 1

        v = np.flip(v, 0)

        t = int(model.N - model.b[m])
        t0, f0, t1, f1 = aux(
            model, v, int(model.N - model.a[m]), cur_c, cur_d, t, t0, f0, t1, f1, q, dt0, df0, dt1, df1, r)
        model.tr0[m] = t0
        model.fa0[m] = f0
        model.tr1[m] = t1
        model.fa1[m] = f1
        if(t != model.N - model.b[m]):
            model.b[m] = model.N - t
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


def aux(model, np.ndarray x, int b, double c, double d, int a, int t0, int f0, int t1, int f1, np.ndarray q, np.ndarray dt0, np.ndarray df0, np.ndarray dt1, np.ndarray df1, r):

    q[a] = 0.0
    cdef double cc = math.log(1.0 - math.exp(c), math.e)
    cdef double dd = math.log(1.0 - math.exp(d), math.e)

    cdef int i = 0
    for i in reversed(range(a)):
        if(x[i]):
            dt0[i] = dt0[i + 1]
            df0[i] = df0[i + 1]
            dt1[i] = dt1[i + 1] + 1
            df1[i] = df1[i + 1] - 1
        else:
            dt0[i] = dt0[i + 1] - 1
            df0[i] = df0[i + 1] + 1
            dt1[i] = dt1[i + 1]
            df1[i] = df1[i + 1]

    # print(a + 1, b)
    # sleep(1)
    i = a + 1
    while(i <= b):
        if(x[i - 1]):
            dt0[i] = dt0[i - 1]
            df0[i] = df0[i - 1]
            dt1[i] = dt1[i - 1] - 1
            df1[i] = df1[i - 1] + 1
        else:
            dt0[i] = dt0[i - 1] + 1
            df0[i] = df0[i - 1] - 1
            dt1[i] = dt1[i - 1]
            df1[i] = df1[i - 1]
        i += 1

    for i in range(b + 1):
        q[i] = dt0[i] * cc + df0[i] * d + dt1[i] * dd + df1[i] * c

    a = randompick(logtop(q, b + 1), b + 1, r)
    t0 += dt0[a]
    f0 += df0[a]
    t1 += dt1[a]
    f1 += df1[a]

    return t0, f0, t1, f1


def randompick(np.ndarray p, int n, r):
    # Pick one sample of length n from vector p
    cdef int i = 0
    cdef double x = r.uniform(0, 1) - p[0]
    while(x > 0.0 and i < n - 1):
        i += 1
        x -= p[i]
    return i


def logtop(np.ndarray p, int n):
    z = np.double(p[0])
    for i in range(1, n):
        if(p[i] > z):
            z = p[i]
    cdef double x = 0.0
    cdef double y = 0.0
    for i in range(n):
        y = math.exp(max(LOGEPSILON, p[i] - z))
        p[i] = y
        x += y
    for i in range(n):
        p[i] = p[i] / x
    return p


def permute(np.ndarray A, np.ndarray P, int n):

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


def samplepi2(model, int swap, r):
    cdef int i = 0
    cdef int j = 0

    if not swap:
        i = r.randrange(0, model.N)
        j = r.randrange(0, model.N - 1)
        if j >= i:
            j += 1
        else:
            n = i
            i = j
            j = n
    else:
        i = r.randrange(0, model.N - 1)
        j = i + 1

    m = 0
    for n in range(i, j + 1):
        m += int(model.H[model.rpi[n]])
        if m > 1:
            return 0

    cdef int inc1 = r.randrange(0, 2)
    cdef int inc2 = r.randrange(0, 2)

    cdef double delta = 0.0
    cdef int dt0 = 0
    cdef int df0 = 0
    cdef int dt1 = 0
    cdef int df1 = 0
    cdef double cur_c = 0.0
    cdef double cur_d = 0.0
    for m in range(model.M):
        dt0, df0, dt1, df1 = [0] * 4
        cur_a = int(model.a[m])
        cur_b = int(model.b[m])
        a_in = ininterval(cur_a, i, j + 1, inc1, inc2)
        b_in = ininterval(cur_b, i, j + 1, inc1, inc2)
        if a_in and not b_in:
            for n in range(i, cur_a):
                if model.X[model.rpi[n]][m]:
                    dt1 += 1
                    df1 -= 1
                else:
                    dt0 -= 1
                    df0 += 1
            for n in range(cur_a, j + 1):
                if model.X[model.rpi[n]][m]:
                    dt1 -= 1
                    df1 += 1
                else:
                    dt0 += 1
                    df0 -= 1
        elif (not a_in and b_in):
            for n in range(i, cur_b):
                if model.X[model.rpi[n]][m]:
                    dt1 -= 1
                    df1 += 1
                else:
                    dt0 += 1
                    df0 -= 1
            for n in range(cur_b, j + 1):
                if model.X[model.rpi[n]][m]:
                    dt1 += 1
                    df1 -= 1
                else:
                    dt0 -= 1
                    df0 += 1

        cur_c = np.double(model.c[m])
        cur_d = np.double(model.d[m])

        delta += dt0 * math.log(1 - math.exp(cur_c), math.e) + \
            df0 * cur_d + dt1 * \
            math.log(1.0 - math.exp(cur_d), math.e) + df1 * cur_c
    cdef double k = 0.0
    while(True):
        k = r.random()
        if k != 0.0:
            break
    # logger.debug(k)
    if delta >= 0.0 or delta > math.log(k, math.e):
        for m in range(model.M):
            cur_a = int(model.a[m])
            cur_b = int(model.b[m])
            a_in = ininterval(cur_a, i, j + 1, inc1, inc2)
            b_in = ininterval(cur_b, i, j + 1, inc1, inc2)
            if a_in and not b_in:
                model.a[m] = i + j + 1 - cur_a
            elif not a_in and b_in:
                model.b[m] = i + j + 1 - cur_b
            elif a_in and b_in:
                model.b[m] = i + j + 1 - cur_a
                model.a[m] = i + j + 1 - cur_b

        for n in range(i, int((i + j) / 2 + 1)):
            m = model.rpi[n]
            model.rpi[n] = model.rpi[i + j - n]
            model.rpi[i + j - n] = m

        model.pi = np.argsort(model.rpi)
        model.loglike += delta
        model.tr0, model.tr1, model.fa0, model.fa1, model.tr0a, model.tr1a, model.fa0a, model.fa1a = compute_trfa_count(
            model)
        # compute_trfa_count(model)
        return 1
    return 0


def ininterval(int i, int a, int b, int inc1, int inc2):
    rr = None
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


def samplepi1(model, r):
    cdef int i = r.randrange(model.N)
    cdef int j = r.randrange(model.N - 1)
    if (j >= i):
        j += 1
    if (i < j):
        ii = i
        jj = j
    else:
        ii = j
        jj = i

    if (model.H[model.rpi[i]]):
        m = 0
        for n in range(ii, jj + 1):
            m += int(model.H[model.rpi[n]])
            if (m > 1):
                return 0

    cdef np.ndarray v = model.X[model.rpi[i], :]
    cdef double delta = 0.0
    cdef int dt0 = 0
    cdef int df0 = 0
    cdef int dt1 = 0
    cdef int df1 = 0
    if (i < j):
        for m in range(model.M):
            a = int(model.a[m])
            b = int(model.b[m])
            a_in = (ii < a and a <= jj + 1)
            b_in = (ii < b and b <= jj + 1)
            if (a_in and not b_in):
                if (v[m]):
                    dt1 += 1
                    df1 -= 1
                else:
                    dt0 -= 1
                    df0 += 1
            elif (not a_in and b_in):
                if (v[m]):
                    dt1 -= 1
                    df1 += 1
                else:
                    dt0 += 1
                    df0 -= 1
            c = np.double(model.c[m])
            d = np.double(model.d[m])
            delta += dt0 * math.log(1 - math.exp(c), math.e) + df0 * \
                d + dt1 * math.log(1 - math.exp(d), math.e) + df1 * c
    else:
        for m in range(model.M):
            a = int(model.a[m])
            b = int(model.b[m])
            a_in = (ii < a and a <= jj)
            b_in = (ii < b and b <= jj)
            if (not a_in and b_in):
                if (v[m]):
                    dt1 += 1
                    df1 -= 1
                else:
                    dt0 -= 1
                    df0 += 1
            elif (a_in and not b_in):
                if (v[m]):
                    dt1 -= 1
                    df1 += 1
                else:
                    dt0 += 1
                    df0 -= 1
            c = np.double(model.c[m])
            d = np.double(model.d[m])
            delta += dt0 * math.log(1 - math.exp(c), math.e) + df0 * \
                d + dt1 * math.log(1 - math.exp(d), math.e) + df1 * c

    cdef double k = 0.0
    while(True):
        k = r.random()
        if k != 0.0:
            break
    if (delta >= 0.0 or delta > math.log(k, math.e)):
        if(i < j):
            for m in range(model.M):
                a = int(model.a[m])
                b = int(model.b[m])
                if (ii < a and a <= jj + 1):
                    model.a[m] = a - 1
                if (ii < b and b <= jj + 1):
                    model.b[m] = b - 1
            t = int(model.rpi[i])
            for n in range(i, j):
                model.rpi[n] = model.rpi[n + 1]
            model.rpi[j] = t
        else:
            for m in range(model.M):
                a = int(model.a[m])
                b = int(model.b[m])
                if (ii <= a and a <= jj):
                    model.a[m] = a + 1
                if (ii <= b and b <= jj):
                    model.b[m] = b + 1
            t = int(model.rpi[i])
            for n in range(i, j):
                model.rpi[n] = model.rpi[n - 1]
            model.rpi[j] = t

        model.pi = np.argsort(model.rpi)
        model.loglike += delta
        compute_trfa_count(model)
        return 1
    return 0


def samplepi3(model, list p, r):
    if (model.N - model.Nh < 2):
        return 0
    cdef int n = r.randrange(0, model.N - model.Nh)
    cdef int m = r.randrange(0, model.N - model.Nh - 1)
    if (n <= m):
        i = n
        j = m + 1
    else:
        i = m
        j = n

    n = 0
    while(n <= i):
        if (model.H[model.rpi[n]]):
            i += 1
            j += 1
        n += 1
    while (n <= j):
        if (model.H[model.rpi[n]]):
            j += 1
        n += 1
    n = i
    m = j
    while (n <= m):
        if (model.H[model.rpi[n]]):
            p[n] = n
            n += 1
        elif (model.H[model.rpi[m]]):
            p[m] = m
            m -= 1
        else:
            p[n] = m
            p[m] = n
            n += 1
            m -= 1

    cdef int nn = 0
    cdef int na = 0
    cdef int nb = 0
    cdef int inc1 = r.randrange(0, 2)
    cdef int inc2 = r.randrange(0, 2)
    cdef double  delta = 0.0
    cdef int dt0 = 0
    cdef int df0 = 0
    cdef int dt1 = 0
    cdef int df1 = 0

    for m in range(model.M):
        dt0, df0, dt1, df1 = [0] * 4
        a = int(model.a[m])
        b = int(model.b[m])
        a_in = ininterval(a, i, j + 1, inc1, inc2)
        b_in = ininterval(b, i, j + 1, inc1, inc2)
        if (a_in and not b_in):
            na = i + j + 1 - a
            nb = b
        elif (not a_in and b_in):
            na = a
            nb = i + j + 1 - b
        elif (a_in and b_in):
            na = i + j + 1 - b
            nb = i + j + 1 - a
        else:
            na = a
            nb = b

        for n in range(i, j + 1):
            nn = p[n]
            wasalive = (a <= n and n < b)
            isalive = (na <= nn and nn < nb)
            if (wasalive and not isalive):
                if (model.X[model.rpi[n], m]):
                    dt1 -= 1
                    df1 += 1
                else:
                    df0 -= 1
                    dt0 += 1
            elif (not wasalive and isalive):
                if (model.X[model.rpi[n], m]):
                    dt1 += 1
                    df1 -= 1
                else:
                    df0 += 1
                    dt0 -= 1

        c = np.double(model.c[m])
        d = np.double(model.d[m])
        delta += dt0 * math.log(1 - math.exp(c), math.e) + df0 * \
            d + dt1 * math.log(1 - math.exp(d), math.e) + df1 * c

    cdef double k = 0.0
    while(True):
        k = r.random()
        if k != 0.0:
            break

    if (delta >= 0.0 or delta > math.log(k, math.e)):
        for m in range(model.M):
            a = int(model.a[m])
            b = int(model.b[m])
            a_in = ininterval(a, i, j + 1, inc1, inc2)
            b_in = ininterval(b, i, j + 1, inc1, inc2)
            if (a_in and not b_in):
                model.a[m] = i + j + 1 - a
            elif (not a_in and b_in):
                model.b[m] = i + j + 1 - b
            elif (a_in and b_in):
                model.b[m] = i + j + 1 - a
                model.a[m] = 1 + j + 1 - b

        for n in range(i, j + 1):
            a = p[n]
            p[n] = model.rpi[n]

        for n in range(i, j + 1):
            model.rpi[n] = p[n]

        model.pi = np.argsort(model.rpi)
        model.loglike += delta
        compute_trfa_count(model)
        return 1
    return 0
