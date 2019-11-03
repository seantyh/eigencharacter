import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class CharDataset(Dataset):
    def __init__(self, M):
        self.M = M
    
    def __len__(self):
        return self.M.shape[1]

    def __getitem__(self, idx):
        return self.M[:,idx]

def load_char_data(M, ptrain, pdev, ptest, batch_size=64):
    idx_list = np.arange(M.shape[1])
    np.random.shuffle(idx_list)
    n_sample = M.shape[1]
    train_idxs = idx_list[:int(n_sample*ptrain)]    
    dev_idxs = idx_list[int(n_sample*ptrain):int(n_sample*(ptrain+pdev))]
    # train_idxs = idx_list[:1000]
    # dev_idxs = idx_list[1000:1100]
    test_idxs = idx_list[int(n_sample*(ptrain+pdev)):]    
    data = CharDataset(M)
    train_loader = DataLoader(data, 
        sampler=SubsetRandomSampler(train_idxs),
        batch_size=batch_size)
    dev_loader = DataLoader(data, 
        sampler=SubsetRandomSampler(dev_idxs),
        batch_size=batch_size)
    test_loader = DataLoader(data, 
        sampler=SubsetRandomSampler(test_idxs),
        batch_size=batch_size)

    return train_loader, dev_loader, test_loader