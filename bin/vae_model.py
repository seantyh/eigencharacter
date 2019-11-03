import os
import pickle
import eigencharacter as ec
from eigencharacter.neural import VAE
from eigencharacter.char_dataset import load_char_data
import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import argparse
from tqdm.autonotebook import tqdm

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

def train(epoch, train_loader, output_dir=""):
    model.train()
    train_loss = 0
    n_sample = len(train_loader.sampler.indices)
    for batch_idx, data in enumerate(train_loader):        
        data = normalize(data.to(device))
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()                

    epoch_loss = train_loss / n_sample
    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), n_sample,
                100. * batch_idx / len(train_loader),
                epoch_loss))
    writer.add_scalar('train/loss', epoch_loss, epoch)

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, epoch_loss))


def test(epoch, test_loader, output_dir=""):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            n_test = data.shape[0]
            data = normalize(data.to(device))
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0 and epoch % log_interval == 0:
                n = min(data.size(0), 8)
                print("recon_batch.shape", recon_batch.shape)
                comparison = torch.cat([data.view(n_test, 1, 75, 64)[:n],
                                      recon_batch.view(n_test, 1, 75, 64)[:n]])                
                save_image(comparison.cpu(),
                         f'{output_dir}/reconstruction_epoch' + str(epoch) + '.png', nrow=n)

    epoch_loss = test_loss / len(train_loader.dataset)
    writer.add_scalar('test/loss', test_loss, epoch)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='eigencharacter VAE trainer')
    parser.add_argument('exp_name', help='Experiment for this run')
    parser.add_argument('fc_node', type=int, help='number of nodes in fully-connected')
    parser.add_argument('vae_node', type=int, help='number of nodes in VAE')
    parser.add_argument('--n_epoch', type=int, default=50, help='number of nodes in VAE')

    args = parser.parse_args()

    n_epoch = args.n_epoch
    log_interval = 10
    batch_size = 100
    m_path = ec.get_resource_path('', 'character_M.pkl')
    with open(m_path, "rb") as fin:
        M = pickle.load(fin)

    in_dim = M.shape[0]
    print(f"input dimension: {in_dim}")

    device = torch.device('cuda')
    model = VAE(in_dim, args.fc_node, args.vae_node).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)    

    model_dir = ec.get_cache_path('EC_vae', '')
    exp_dir = f'{model_dir}/{args.exp_name}'
    
    if os.path.exists(exp_dir):
        print(f"{exp_dir} already existed, remove the directory and run again")
        exit()

    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f'{exp_dir}/results', exist_ok=True)
    writer = SummaryWriter(logdir=exp_dir)
    model_path = f"{exp_dir}/vae_model_{args.exp_name}.pth"
        
    train_loader, dev_loader, test_loader = load_char_data(M, 0.8, 0.2, 0, batch_size)

    for epoch in tqdm(range(1, n_epoch + 1)):
        train(epoch, train_loader)
        test(epoch, dev_loader, f'{model_dir}/{args.exp_name}/results')
        with torch.no_grad():
            sample = torch.randn(64, args.vae_node).to(device)
            sample = model.decode(sample).cpu()
            if epoch % log_interval == 0:
                save_image(sample.view(64, 1, 75, 64),
                       f'{model_dir}/{args.exp_name}/sample_' + str(epoch) + '.png')
    torch.save(model.state_dict(), model_path)
    writer.add_text("model/info", 
        f"FC: {args.fc_node}/VAE: {args.vae_node} for {args.n_epoch} epoch")
    writer.export_scalars_to_json(f'{model_dir}/{args.exp_name}/scalars_log.json')
    writer.close()
