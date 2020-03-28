import numpy as np
import logging
import math
from model import model
from read_model import compute_trfa_count, compute_loglike, compute_lifespan

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logfile_handler = logging.FileHandler('mcmc.log', mode='w')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S')
logfile_handler.setFormatter(formatter)
logger.addHandler(logfile_handler)


def mcmc_randomize(model, r):

    if (model.Nh == 0):  # Zero hard sites
        r.shuffle(model.pi)
        model.rpi = np.argsort(model.pi)
        # Recompute parameters
        model.tr0, model.tr1, model.fa0, model.fa1, model.tr0a, model.tr1a, model.fa0a, model.fa1a = compute_trfa_count(
            model)
        model.loglike = compute_loglike(model)
        return
    elif(model.Nh == model.N):  # All sites are hard sites
        return

    all_sites = [i for i in range(model.N)]
    hard_sites = []  # contains Nh sites randomly chosen from all sites
    hard_sites = r.sample(all_sites, k=model.Nh)
    list.sort(hard_sites)

    # remove hard sites from all_sites
    j, k = [0] * 2
    for i in range(model.N):
        if j == model.Nh:
            all_sites[k] = i
            k += 1
        else:
            if (i == hard_sites[j]):
                j += 1
            else:
                all_sites[k] = i
                k += 1

    # shuffle remaining sites
    temp_arr = [all_sites[i] for i in range(model.N - model.Nh)]
    r.shuffle(temp_arr)
    for i in range(model.N - model.Nh):
        all_sites[i] = temp_arr[i]

    # logger.debug(model.pi)
    j, k = [0] * 2
    for i in range(model.N):
        if model.H[i]:
            model.pi[i] = hard_sites[j]
            j += 1
        else:
            model.pi[i] = all_sites[k]
            k += 1

    if(check_valid_permutation(model.pi, model) is False):
        logger.error("Invalid permutation generated")

    model.rpi = np.argsort(model.pi)
    model.a, model.b = compute_lifespan(model)
    model.tr0, model.tr1, model.fa0, model.fa1, model.tr0a, model.tr1a, model.fa0a, model.fa1a = compute_trfa_count(
        model)
    model.loglike = compute_loglike(model)


def check_valid_permutation(arr, model):
    hash_arr = [0 for i in range(max(arr) + 1)]
    for i in range(len(arr)):
        if arr[i] >= model.N:
            logger.error(arr[i])
            return False
        hash_arr[arr[i]] += 1
    for i in range(len(hash_arr)):
        if hash_arr[i] > 1:
            logger.error(i)
            return False
    return True


def check_consistency(model):
    flag = True

    # Check lifespans
    for m in range(model.M):
        cur_a = model.a[m]
        cur_b = model.b[m]
        if(not(cur_a >= 0 and cur_b >= cur_a and model.N >= cur_b)):
            logger.error(
                "Consistency error a = %d, b = %d, m = %d", cur_a, cur_b, m)
            flag = False

    # Check order of pi
    if(not(check_valid_permutation(model.pi, model))):
        flag = False
        logger.error(model.pi)
        logger.error("Inconsistent Permutation generated")

    # check for rpi
    p = np.argsort(model.pi)
    for n in range(model.N):
        if (model.rpi[n] != p[n]):
            logger.error("Inverse permutation is invalid")
            flag = False
            break

    m = -1
    i = 0
    for n in range(model.N):
        if(model.H[n]):
            i += 1
            if(m >= 0 and model.pi[n] < m):
                logger.error(
                    "Hard sites order is not maintained n = %d, pi[n] = %d, m = %d", n, model.pi[n], m)
                flag = False
            m = model.pi[n]

    if (i != model.Nh):
        logger.error(
            "Incorrect Number of hard sites i = %d, Nh = %d", i, model.Nh)
        flag = False

    t0 = model.tr0a
    f0 = model.fa0a
    t1 = model.tr1a
    f1 = model.fa1a
    loglike = model.loglike
    model.tr0, model.tr1, model.fa0, model.fa1, model.tr0a, model.tr1a, model.fa0a, model.fa1a = compute_trfa_count(
        model)
    model.loglike = compute_loglike(model)
    delta = loglike - model.loglike
    if (delta < float(0)):
        delta = -delta

    if (t0 != (model.tr0a) or f0 != (model.fa0a) or t1 != (model.tr1a) or f1 != (model.fa1a) or delta > math.e ** (-8)):
        logger.error("Inconsistent parameters t0 = %d %d, f0 %d %d, t1 %d %d, f1 %d %d, logl %f %f",
                     t0, model.tr0a, f0, model.fa0a, t1, model.tr1a, f1, model.fa1a, loglike, model.loglike)
        flag = False

    return flag
