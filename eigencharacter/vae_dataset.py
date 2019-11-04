import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class VaeDataset(Dataset):
    def __init__(self, emb_wv, lexicon, zcoeffs, M_char2id):
        assert all((x in emb_wv) for x in lexicon)
        self.emb = self.normalize_wv(emb_wv)
        self.lexicon = lexicon
        self.zcoeffs = self.normalize(zcoeffs)
        self.M_char2id = M_char2id

    def __len__(self):
        return len(self.lexicon)

    def __getitem__(self, idx):
        char_x = self.lexicon[idx]
        M_idx = self.M_char2id[char_x]
        return self.zcoeffs[M_idx,:], self.emb[char_x]

    def normalize(self, X):
        minX = X.min()
        maxX = X.max()
        return (((X - minX) / (maxX - minX)) - 0.5) * 2

    def normalize_wv(self, wv):
        wv.vectors = self.normalize(wv.vectors)
        return wv

def load_vae_data(emb_wv, lexicon, zcoeffs, ptrain, pdev, batch_size=100):
    M_char2id = {char: idx for idx, char in enumerate(lexicon)}
    lexicon_sub = [x for x in lexicon if x in emb_wv]
    vae_dataset = VaeDataset(emb_wv, lexicon_sub, zcoeffs, M_char2id)
    idx_list = np.arange(len(lexicon_sub))
    np.random.seed(12345)
    np.random.shuffle(idx_list)

    n_sample = len(idx_list)
    train_idxs = idx_list[:int(n_sample*ptrain)]
    dev_idxs = idx_list[int(n_sample*ptrain):]

    train_loader = DataLoader(vae_dataset,
        sampler=SubsetRandomSampler(train_idxs),
        batch_size=batch_size)
    dev_loader = DataLoader(vae_dataset,
        sampler=SubsetRandomSampler(dev_idxs),
        batch_size=batch_size)

    return train_loader, dev_loader


