import logging
from model import model
from read_model import read_model

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
