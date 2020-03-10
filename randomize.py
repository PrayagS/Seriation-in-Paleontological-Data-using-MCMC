import numpy as np
import logging
from model import model
from read_model import compute_trfa_count, compute_loglike

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logfile_handler = logging.FileHandler('mcmc.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logfile_handler.setFormatter(formatter)
logger.addHandler(logfile_handler)


def mcmc_randomize(model, r):

    if (model.Nh == 0):  # Zero hard sites
        r.shuffle(model.pi)
        logger.debug(model.pi)
        model.rpi = np.argsort(model.pi)
        # Recompute parameters
        compute_trfa_count(model)
        model.loglike = compute_loglike(model)
        return
    elif(model.Nh == model.N):  # All sites are hard sites
        return

    all_sites = [i for i in range(model.N)]
    hard_sites = []  # contains Nh sites randomly chosen from all sites
    hard_sites = r.sample(all_sites, k=model.Nh)
    list.sort(hard_sites)
    hard_sites = [12, 20, 21, 27, 35, 37, 43, 58, 65, 73, 99]
    logger.debug(hard_sites)

    # remove hard sites from all_sites
    j, k = [0]*2
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
    logger.debug(temp_arr)
    for i in range(model.N - model.Nh):
        all_sites[i] = temp_arr[i]
    logger.debug(all_sites)
