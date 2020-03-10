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
