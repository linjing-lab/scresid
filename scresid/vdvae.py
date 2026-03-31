from .residual import EfficientEncoder, EfficientDecoder
from .utils import AnnDataSet
import torch, numpy, scanpy, tqdm, ot, sklearn

class SCRESID(torch.nn.Module):
    def __init__(self, input_dim=6000, latent_dim=100, hidden_dim=1000, noise_rate=0.1, kl_weight=5e-4, device=None):
        super(SCRESID, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.noise_rate = noise_rate
        self.kl_weight = kl_weight
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        # Encoder layers
        self.encoder = EfficientEncoder(
            input_dim, hidden_dim, latent_dim
        )

        # Decoder layers
        self.decoder = EfficientDecoder(
            latent_dim, hidden_dim, input_dim
        )

    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = torch.chunk(h, 2, dim=1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward(self, x):
        noise = torch.randn_like(x)
        x_noisy = x + noise * self.noise_rate
        z, mu, logvar = self.encode(x_noisy)
        x_hat = self.decoder(z)

        std = torch.exp(logvar / 2)
        loss_kl = torch.distributions.kl_divergence(torch.distributions.Normal(mu, std), torch.distributions.Normal(0, 1)).sum(dim=1)
        loss_rec = ((x - x_hat) ** 2).sum(dim=1)

        return x_hat, loss_rec, loss_kl

    def latent_adata(self, adata):
        x = torch.Tensor(adata.to_df().values).to(self.device)
        latent_z = self.encode(x)[0].cpu().detach().numpy()
        latent = scanpy.AnnData(X=latent_z, obs=adata.obs.copy())
        return latent

    def train_SCRESID(self, train_adata, epochs=100, batch_size=128, lr=5e-5):
        anndataset = AnnDataSet(train_adata)
        train_loader = torch.utils.data.DataLoader(anndataset, batch_size=batch_size, shuffle=True, drop_last=False)
        SCRESID_loss, loss_rec, loss_kl = 0, 0, 0
        optim_SCRESID = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
        pbar = tqdm.tqdm(range(epochs))
        for epoch in pbar:
            pbar.set_description("Training Epoch {}".format(epoch))
            for _, x in enumerate(train_loader):
                x = x.to(self.device)
                _, loss_rec, loss_kl = self.forward(x)
                SCRESID_loss = (0.5 * loss_rec + 0.5 * (loss_kl * self.kl_weight)).mean()
                optim_SCRESID.zero_grad()
                SCRESID_loss.backward()
                torch.nn.utils.clip_grad_norm(self.parameters(), 10)
                optim_SCRESID.step()

    def predict(self, train_adata, cell_to_pred, key_dic, ratio=0.05):
        print(f"Get adata of {cell_to_pred} and control...")
        ctrl_to_pred = train_adata[((train_adata.obs[key_dic['cell_type_key']] == cell_to_pred) &
                                    (train_adata.obs[key_dic['condition_key']] == key_dic['ctrl_key']))]
        print(f"Get adata of without {cell_to_pred} and control...")
        ctrl_no_pred = train_adata[(train_adata.obs[key_dic['cell_type_key']] != cell_to_pred) &
                                 (train_adata.obs[key_dic['condition_key']] == key_dic['ctrl_key'])]
        print("Get stimulated adata of condition...")
        stim_adata = train_adata[(train_adata.obs[key_dic['condition_key']] == key_dic['stim_key'])]
        print("Get latent adata of control and stimulated...")
        control = self.latent_adata(ctrl_no_pred).to_df().values
        stimulated = self.latent_adata(stim_adata).to_df().values
        print("Compute distance of stimulated and control by POT...")
        M = ot.dist(stimulated, control, metric='euclidean')
        print("Compute G of stimulated and control by emd in POT...")
        G = ot.emd(torch.ones(stimulated.shape[0]) / stimulated.shape[0], torch.ones(control.shape[0]) / control.shape[0], torch.tensor(M), numItermax=100000)
        print("Match max idx from G...")
        match_idx = torch.max(G, 0)[1].numpy()
        stim_matched = stimulated[match_idx]
        print("Get delta list of matched stimulated and control...")
        deltas = stim_matched - control
        print(f"Get latent adata of {cell_to_pred}...")
        test_z = self.latent_adata(ctrl_to_pred).to_df().values
        print(f"Compute cosine similarity with latent adata of {cell_to_pred} and control...")
        cos_sim = sklearn.metrics.pairwise.cosine_similarity(numpy.array(test_z).reshape(-1, self.latent_dim), numpy.array(control).reshape(-1, self.latent_dim))
        print(f"Select top control with ratio {ratio}...")
        n_top = int(numpy.ceil(control.shape[0] * ratio))
        top_indices = numpy.argsort(cos_sim)[0][-n_top:]
        print(f"Normalize weights of {n_top} top cosine similarity...")
        normalized_weights = cos_sim[0][top_indices] / numpy.sum(cos_sim[0][top_indices])
        print("Compute delta of predict...")
        delta_pred = numpy.sum(normalized_weights[:, numpy.newaxis] * numpy.array(deltas).reshape(-1, self.latent_dim)[top_indices], axis=0)
        pred_z = test_z + delta_pred # attention mechanism
        print("Decode predict latent embedding...")
        pred_x = self.decoder(torch.Tensor(pred_z).to(self.device)).cpu().detach().numpy()
        print("Create predict adata and save key predict...")
        pred_adata = scanpy.AnnData(X=pred_x, obs=ctrl_to_pred.obs.copy(), var=ctrl_to_pred.var.copy())
        pred_adata.obs[key_dic['condition_key']] = key_dic['pred_key']
        print("Predict process completed.")
        return pred_adata