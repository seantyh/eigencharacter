import pickle
import numpy as np

class SvdUtils:
    def __init__(self, svd_path, k):
        with svd_path.open("rb") as fin:
            svd = pickle.load(fin)
        self.U = svd[0][:, :k]
        self.sigma = svd[1][:k]
        self.Vt = svd[2][:k, :]
    
    def project_svd(self, im_vec):
        sigma = self.sigma
        U = self.U
        A = np.dot(np.diag(1/sigma), np.transpose(U))
        proj = np.dot(A, im_vec)
        return proj

    def recon_svd(self, coeff_mat):
        sigma = self.sigma
        U = self.U
        A = np.dot(U, np.diag(sigma))
        recon_mat = np.dot(A, coeff_mat)
        return recon_mat
    