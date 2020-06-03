import logging
import numpy as np
cimport numpy as np
from random import Random
from model import model
from read_model import read_model
from randomize import mcmc_randomize, check_consistency
from sampling import sample

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logfile_handler = logging.FileHandler('mcmc.log', mode='w')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s', datefmt='%H:%M:%S')
logfile_handler.setFormatter(formatter)
logger.addHandler(logfile_handler)

def run():
    logger.debug('STARTED')
    f = open("../Dataset/g10s10.txt", 'r')
    mcmc_model = read_model(f)
    f.close()
    cdef int seed = 42  # Seed for the RNG
    r = Random()
    r.seed(seed)
    mcmc_randomize(mcmc_model, r)
    check_consistency(mcmc_model)

    for i in range(100):
        print(i)
        sample(mcmc_model, r)

    for i in range(100):
        print('GG',i)
        sample(mcmc_model, r)
        logger.debug(mcmc_model.a)
        logger.debug(mcmc_model.b)
        logger.debug(mcmc_model.pi)
        logger.debug(mcmc_model.c)
        logger.debug(mcmc_model.d)
        logger.debug(mcmc_model.loglike)
