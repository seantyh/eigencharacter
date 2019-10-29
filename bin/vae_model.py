import pickle
import eigencharacter as ec
from eigencharacter.neural import VAE
from eigencharacter.char_dataset import load_char_data
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, in_dim), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD

def normalize(x):
    return (x/255).type(torch.float)

def train(epoch, train_loader):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = normalize(data.to(device))
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            n_test = data.shape[0]
            data = normalize(data.to(device))
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                print("recon_batch.shape", recon_batch.shape)
                comparison = torch.cat([data.view(n_test, 1, 75, 64)[:n],
                                      recon_batch.view(n_test, 1, 75, 64)[:n]])                
                save_image( comparison.cpu(),
                         'results/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":
    n_epoch = 50
    log_interval = 100
    batch_size = 64
    m_path = ec.get_resource_path('', 'character_M.pkl')
    with open(m_path, "rb") as fin:
        M = pickle.load(fin)

    in_dim = M.shape[0]
    print(f"input dimension: {in_dim}")
    device = torch.device('cpu')
    model = VAE(in_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    train_loader, valid_loader, test_loader = load_char_data(M, 0.8, 0.1, 0.1, batch_size)

    for epoch in range(1, n_epoch + 1):
        train(epoch, train_loader)
        test(epoch, test_loader)
        with torch.no_grad():
            sample = torch.randn(64, 50).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, 1, 75, 64),
                       'results/sample_' + str(epoch) + '.png')