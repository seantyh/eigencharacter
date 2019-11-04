
import argparse
import torch
import torch.nn as nn
import eigencharacter as ec
import pickle
from eigencharacter.neural import MF_Model
from tensorboardX import SummaryWriter
from tqdm.autonotebook import tqdm
import logging
import json

logger = logging.getLogger()
logger.setLevel("INFO")

MEAN2FORM = 'mean2form'
FORM2MEAN = 'form2mean'

def preprocess(X, Y):
    X = X.type(torch.float).to(device)
    Y = Y.type(torch.float).to(device)

    return X, Y

def data_dispatch(batch_form, batch_mean):
    if data_mode == MEAN2FORM:
        X = batch_mean
        Y = batch_form
    elif data_mode == FORM2MEAN:
        X = batch_form
        Y = batch_mean
    return X, Y

def train(epoch, model, train_loader):
    model.train()
    optim.zero_grad()
    train_loss = 0
    for batch_f, batch_m in train_loader:
        X, Y = data_dispatch(batch_f, batch_m)
        X, Y = preprocess(X, Y)
        Y_pred = model(X)
        loss = loss_fn(Y_pred, Y)
        loss.backward()
        optim.step()
        train_loss += loss.item()
    n_sample = len(train_loader.sampler.indices)
    train_loss = train_loss / n_sample
    writer.add_scalar('train/loss', train_loss, epoch)

def evaluate(epoch, model, test_loader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_f, batch_m in test_loader:
            X, Y = data_dispatch(batch_f, batch_m)
            X, Y = preprocess(X, Y)
            Y_pred = model(X)
            loss = loss_fn(Y_pred, Y)
            test_loss += loss.item()
    n_sample = len(test_loader.sampler.indices)
    test_loss = test_loss / n_sample
    writer.add_scalar('test/loss', test_loss, epoch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str, help="Name of experiment")
    parser.add_argument("--form", help="svd/vae", choices=["svd", "vae"], default="vae")
    parser.add_argument("--form_dim", type=int, help="dimension of form vector", 
                        choices=[10,50,100], default=50)
    parser.add_argument("--data_mode", help="mean2form or form2mean", 
                        choices=['mean2form', 'form2mean'], default='mean2form')
    parser.add_argument("--n_epoch", type=int, help="number of epoch", default=10)
    parser.add_argument("--hidden_dim", type=int, default=100)

    args = parser.parse_args()
    model_dir = ec.get_cache_path("MF_Model", "")
    if not model_dir.exists():
        ec.install_cache_dir(args.exp_name)

    exp_dir = model_dir / args.exp_name
    if exp_dir.exists():
        logger.error(f"{model_dir} already existed, remove before continuing")
        exit()

    n_epoch = args.n_epoch

    # Data preparation
    logger.info("Loading embeddings")
    emb_path = ec.get_resource_path("", "gensim_kv_fasttext_tc.pkl")
    with emb_path.open("rb") as fin:
        emb = pickle.load(fin)

    logger.info("Loading character lexicon")
    chfreq_path = ec.get_resource_path("", "as_chFreq.pickle")
    with chfreq_path.open("rb") as fin:
        chfreq = pickle.load(fin)
        chars = sorted(chfreq.keys(), key=chfreq.get, reverse=True)
        freq_chars = chars[:5000]

    if args.form == "vae":
        logger.info(f"Loading VAE vectors: {args.form_dim}")
        vae_prefix = {10: "a", 50: "b", 100: "c"}[args.form_dim]
        zcoeff_path = ec.get_cache_path("VAE_zcoeff", f"vae_{vae_prefix}_zcoeff.pkl")
        with zcoeff_path.open("rb") as fin:
            zcoeffs = pickle.load(fin)
        train_loader, test_loader = \
            ec.load_vae_data(emb, freq_chars, zcoeffs, 0.8, 0.2)

    elif args.form == "svd":
        logger.info(f"Loading SVD vectors: {args.form_dim}")
        cv_path = ec.get_resource_path('', 'charac_coeff.pkl')
        with open(cv_path, "rb") as fin:
            cv_itos, cv_stoi, cv_vectors = pickle.load(fin)
        train_loader, test_loader = \
            ec.load_svd_data(emb, cv_vectors, cv_itos, 0.8, 0.2, k=args.form_dim)
    else:
        logger.error(f"form {args.form} not supported")
        exit()

    # Model initialization
    logger.info("Initializing model")
    device = torch.device('cuda')
    emb_dim = emb.vectors.shape[1]
    form_dim = args.form_dim

    data_mode = args.data_mode
    if data_mode == MEAN2FORM:
        model = MF_Model(emb_dim, form_dim, args.hidden_dim).to(device)
    elif data_mode == FORM2MEAN:
        model = MF_Model(form_dim, emb_dim, args.hidden_dim).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(exp_dir)

    # Train loop
    for epoch in tqdm(range(1, n_epoch + 1)):
        train(epoch, model, train_loader)
        evaluate(epoch, model, test_loader)

    model_path = exp_dir / f"MF_Model-{args.exp_name}.pth"
    torch.save(model.state_dict(), model_path)
    writer.add_text("model/info", 
        f"MF_Model/{args.data_mode.upper()}/{args.form}-{args.form_dim}/H{args.hidden_dim}")
    writer.close()

    param_path = exp_dir / f"MF_Model-{args.exp_name}.json"
    with param_path.open("w") as fout:
        json.dump({
            "mode": args.data_mode,
            "form": args.form,
            "form_dim": form_dim,
            "emb_dim": emb_dim,
            "hidden_dim": args.hidden_dim
        }, fout, indent=4)




