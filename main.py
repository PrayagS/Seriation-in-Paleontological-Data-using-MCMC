import logging
from random import Random
from model import model
from read_model import read_model
from randomize import mcmc_randomize, check_consistency
from sampling import sample

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

logfile_handler = logging.FileHandler('mcmc.log')
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logfile_handler.setFormatter(formatter)
logger.addHandler(logfile_handler)

if __name__ == "__main__":
    logger.debug('STARTED')
    f = open("Dataset/g10s10.txt", 'r')
    mcmc_model = read_model(f)
    f.close()
    seed = 42  # Seed for the RNG
    r = Random()
    r.seed(seed)
    mcmc_randomize(mcmc_model, r)
    check_consistency(mcmc_model)

    for i in range(1000):
        sample(mcmc_model, r)

    for i in range(1000):
        sample(mcmc_model, r)
        logger.debug(mcmc_model.a)
        logger.debug(mcmc_model.b)
        logger.debug(mcmc_model.pi)
        logger.debug(mcmc_model.c)
        logger.debug(mcmc_model.d)
        logger.debug(mcmc_model.loglike)
