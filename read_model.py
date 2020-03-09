from model import model


def read_no_of_sites_taxa(f):
    str = f.readline()
    N, M = map(int, str.split())
    return N, M


def construct_matrix(f, N, M):
    # for i in range(N):
    str = f.readline()
    for j in range(M):
        print(str.split(" ")[j])


if __name__ == "__main__":
    f = open("Dataset/g10s10.txt", 'r')
    mcmc_model = model()
    N, M = read_no_of_sites_taxa(f)
    print(N, M)
    mcmc_model.N = N
    mcmc_model.M = M

    construct_matrix(f, mcmc_model.N, mcmc_model.M)
    f.close()
