import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class SvdDataset(Dataset):
    def __init__(self, emb_wv, coeff_vectors, coeff_itos, k=50):        
        self.itos = {i:s for i, s in coeff_itos.items() if s in emb_wv}  
        self.lex_ids = list(self.itos.keys())
        self.coeffs = self.normalize(coeff_vectors)
        self.emb = self.normalize_wv(emb_wv)
        self.k = k
    
    def __len__(self):
        return len(self.lex_ids)
    
    def __getitem__(self, idx):
        char_idx = self.lex_ids[idx]        
        char_x = self.itos[char_idx]
        return self.coeffs[char_idx, :self.k], self.emb[char_x]
    
    def normalize(self, X):
        minX = X.min()
        maxX = X.max()
        return (((X - minX) / (maxX - minX)) - 0.5) * 2

    def normalize_wv(self, wv):
        wv.vectors = self.normalize(wv.vectors)
        return wv
    
def load_svd_data(emb_wv, coeff_vector, coeff_itos, 
        ptrain, pdev, batch_size=100, k=50):  

    svd_dataset = SvdDataset(emb_wv, coeff_vector, coeff_itos, k)
    idx_list = np.arange(len(svd_dataset))
    np.random.seed(12345)
    np.random.shuffle(idx_list)

    n_sample = len(idx_list)
    train_idxs = idx_list[:int(n_sample*ptrain)]    
    dev_idxs = idx_list[int(n_sample*ptrain):]

    train_loader = DataLoader(svd_dataset, 
        sampler=SubsetRandomSampler(train_idxs),
        batch_size=batch_size)
    dev_loader = DataLoader(svd_dataset, 
        sampler=SubsetRandomSampler(dev_idxs),
        batch_size=batch_size)
    
    return train_loader, dev_loader


