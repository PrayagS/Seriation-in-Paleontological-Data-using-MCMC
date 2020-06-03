import numpy as np
cimport numpy as np

class model():

    def __init__(self):
        # self.N = N
        # self.M = M
        # self.X = X
        # self.a = a
        # self.b = b
        # self.pi = pi
        # self.rpi = rpi
        # self.H = H
        # self.c = c
        # self.d = d
        # self.loglik = loglik
        # self.tr0 = tr0
        # self.tr1 = tr1
        # self.fa0 = fa0
        # self.fa1 = fa1
        # self.tr0a = tr0a
        # self.tr1a = tr1a
        # self.fa0a = fa0a
        # self.fa1a = fa1a
        # self.Nh = Nh
        pass

    def set_sites(self, int N, int M):
        self.N = N
        self.M = M

    def set_matrix(self, np.ndarray X, np.ndarray H, int Nh):
        self.X = X
        self.H = H
        self.Nh = Nh

    def set_pi(self, np.ndarray pi, np.ndarray rpi):
        self.pi = pi
        self.rpi = rpi

    def set_lifespan(self, np.ndarray a, np.ndarray b):
        self.a = a
        self.b = b

    def set_cd(self, np.ndarray c, np.ndarray d):
        self.c = c
        self.d = d

    def set_trfe(self, np.ndarray tr0, np.ndarray tr1, np.ndarray fa0, np.ndarray fa1, int tr0a, int tr1a, int fa0a, int fa1a):
        self.tr0 = tr0
        self.tr1 = tr1
        self.fa0 = fa0
        self.fa1 = fa1
        self.tr0a = tr0a
        self.tr1a = tr1a
        self.fa0a = fa0a
        self.fa1a = fa1a

    def set_loglike(self, double loglike):
        self.loglike = loglike