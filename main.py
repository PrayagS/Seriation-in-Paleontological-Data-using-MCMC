from model import model
from read_model import read_model

if __name__ == "__main__":
    f = open("Dataset/g10s10.txt", 'r')
    mcmc_model = read_model(f)
    f.close()
