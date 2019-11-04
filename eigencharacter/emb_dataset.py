import numpy as np
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

class EmbeddingDataset(Dataset):
    def __init__(self, emb_wv, lexicon, charM, M_char2id):
        assert all((x in emb_wv) for x in lexicon)
        self.emb = emb_wv
        self.lexicon = lexicon        
        self.charM = charM
        self.M_char2id = M_char2id
    
    def __len__(self):
        return len(self.lexicon)
    
    def __getitem__(self, idx):        
        char_x = self.lexicon[idx]
        M_idx = self.M_char2id[char_x]
        return self.charM[:,M_idx], self.emb[char_x]
    
def load_embedding_data(emb_wv, lexicon, charM, ptrain, pdev, batch_size=100):
    M_char2id = {char: idx for idx, char in enumerate(lexicon)}
    lexicon_sub = [x for x in lexicon if x in emb_wv]    
    emb_dataset = EmbeddingDataset(emb_wv, lexicon_sub, charM, M_char2id)
    idx_list = np.arange(len(lexicon_sub))
    np.random.shuffle(idx_list)

    n_sample = len(idx_list)
    train_idxs = idx_list[:int(n_sample*ptrain)]    
    dev_idxs = idx_list[int(n_sample*ptrain):]

    train_loader = DataLoader(emb_dataset, 
        sampler=SubsetRandomSampler(train_idxs),
        batch_size=batch_size)
    dev_loader = DataLoader(emb_dataset, 
        sampler=SubsetRandomSampler(dev_idxs),
        batch_size=batch_size)
    
    return train_loader, dev_loader


