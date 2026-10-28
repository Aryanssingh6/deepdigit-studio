"""
DeepDigit Studio
=================
A Deep Learning and Probabilistic Modelling Project (dataset: MNIST).

Modules (mapped to your 8 practicals)
  Part 1  Probability analysis (joint / marginal / conditional)   -> Practical 1, 2, 3
  Part 2  Autoencoder                                              -> Practical 6
  Part 3  Markov chain on digit classes (latent space walk)        -> Practical 5
  Part 4  RBM with 64 / 128 / 256 hidden units                     -> Practical 4
  Part 5  GAN                                                      -> Practical 7
  Part 6  Latent mixing (style-mixing idea on MNIST)               -> Practical 8
  Part 7  Final comparison table + charts

Run:   python deepdigit_studio.py
Needs: pip install torch torchvision numpy pandas matplotlib
All results are saved in the "outputs" folder.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

# ----------------------------------------------------------------------------
# Settings (change here if you want faster / better results)
# ----------------------------------------------------------------------------
SEED = 42
OUT = "outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLF_EPOCHS = 3
AE_EPOCHS = 10
AE_LATENT = 16
RBM_EPOCHS = 5
HIDDEN_SIZES = [64, 128, 256]
GAN_EPOCHS = 30
Z_DIM = 100
N_EVAL = 2000          # number of test images used for evaluation

os.makedirs(OUT, exist_ok=True)
torch.manual_seed(SEED)
np.random.seed(SEED)
print("Using device:", DEVICE)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def load_mnist():
    """Return train/test images flattened to 784 in range [0, 1]."""
    tr = datasets.MNIST("data", train=True, download=True)
    te = datasets.MNIST("data", train=False, download=True)
    xtr = tr.data.float().div(255).view(-1, 784)
    xte = te.data.float().div(255).view(-1, 784)
    return xtr, tr.targets, xte, te.targets


def save_grid(imgs, path, ncols=10, title=""):
    """Save a grid of flattened 28x28 images."""
    imgs = imgs.detach().cpu().view(-1, 28, 28).numpy()
    rows = int(np.ceil(len(imgs) / ncols))
    canvas = np.zeros((rows * 28, ncols * 28))
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncols)
        canvas[r * 28:(r + 1) * 28, c * 28:(c + 1) * 28] = im
    plt.figure(figsize=(ncols * 0.9, rows * 0.9 + 0.6))
    plt.imshow(canvas, cmap="gray", vmin=0, vmax=1)
    plt.axis("off")
    plt.title(title)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_losses(curves, path, title, xlabel="Epoch"):
    plt.figure(figsize=(6, 4))
    for name, values in curves.items():
        plt.plot(values, label=name)
    plt.xlabel(xlabel)
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# Helper classifier: used as a "judge" for reconstructions and generated images
# ----------------------------------------------------------------------------
def train_classifier(xtr, ytr, xte, yte):
    model = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, 10)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=128, shuffle=True)
    for ep in range(CLF_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = F.cross_entropy(model(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        acc = (model(xte.to(DEVICE)).argmax(1).cpu() == yte).float().mean().item()
    print(f"[Classifier] test accuracy = {acc:.4f}")
    return model


@torch.no_grad()
def class_probs(clf, x):
    """Softmax probability over the 10 digits for each image."""
    return F.softmax(clf(x.to(DEVICE)), dim=1).cpu()


def evaluate_images(name, original, recon, labels, clf, true_dist):
    """Return metrics + average digit probability distribution of the images."""
    recon = recon.detach().cpu()
    probs = class_probs(clf, recon)
    pred = probs.argmax(1)
    dist = probs.mean(0).numpy()                      # P(digit) of the images
    tv = 0.5 * np.abs(dist - true_dist).sum()         # distance to real distribution
    result = {
        "Model": name,
        "MSE": ((recon - original) ** 2).mean().item() if original is not None else np.nan,
        "Digit accuracy": (pred == labels).float().mean().item() if labels is not None else np.nan,
        "Mean confidence": probs.max(1).values.mean().item(),
        "Distribution distance (TV)": tv,
    }
    return result, dist


# ----------------------------------------------------------------------------
# PART 1: Probability analysis (joint, marginal, conditional)
# ----------------------------------------------------------------------------
def probability_analysis(xtr, ytr):
    print("\n" + "=" * 60)
    print("PART 1: Joint, Marginal and Conditional Probability")
    print("=" * 60)

    ink = xtr.sum(1).numpy()                                  # total brightness
    ink_level = pd.qcut(ink, 3, labels=["Low", "Medium", "High"])
    parity = np.where(ytr.numpy() % 2 == 0, "Even", "Odd")
    cols_with_ink = (xtr.view(-1, 28, 28) > 0.3).any(dim=1)   # (N, 28) columns touched
    width = cols_with_ink.sum(1).numpy()
    width_level = np.where(width > np.median(width), "Wide", "Narrow")

    df = pd.DataFrame({"Digit": ytr.numpy(), "Parity": parity,
                       "Ink": ink_level, "Width": width_level})

    # ---- Two random variables: Parity and Ink ----
    joint2 = pd.crosstab(df["Parity"], df["Ink"], normalize=True)
    print("\nJoint P(Parity, Ink):\n", joint2.round(4))
    print("Marginal P(Parity):\n", joint2.sum(axis=1).round(4))
    print("Marginal P(Ink):\n", joint2.sum(axis=0).round(4))
    cond_ink_given_parity = joint2.div(joint2.sum(axis=1), axis=0)
    cond_parity_given_ink = joint2.div(joint2.sum(axis=0), axis=1)
    print("\nConditional P(Ink | Parity):\n", cond_ink_given_parity.round(4))
    print("Conditional P(Parity | Ink):\n", cond_parity_given_ink.round(4))
    assert np.isclose(joint2.values.sum(), 1.0), "2-variable joint must sum to 1"
    print("Check: total joint probability (2 variables) =", round(joint2.values.sum(), 6))

    # ---- Three random variables: Parity, Ink, Width ----
    joint3 = df.groupby(["Parity", "Ink", "Width"], observed=True).size() / len(df)
    print("\nJoint P(Parity, Ink, Width):\n", joint3.round(4).to_frame("P"))
    print("Marginal P(Width):\n", joint3.groupby("Width").sum().round(4))
    cond3 = joint3 / joint3.groupby(level=["Parity", "Ink"]).transform("sum")
    print("Conditional P(Width | Parity, Ink):\n", cond3.round(4).to_frame("P"))
    assert np.isclose(joint3.sum(), 1.0), "3-variable joint must sum to 1"
    print("Check: total joint probability (3 variables) =", round(joint3.sum(), 6))

    # ---- Useful extra: P(Digit | Ink) ----
    digit_given_ink = pd.crosstab(df["Digit"], df["Ink"], normalize="columns")
    print("\nConditional P(Digit | Ink):\n", digit_given_ink.round(4))

    joint2.to_csv(f"{OUT}/joint_parity_ink.csv")
    joint3.to_frame("P").to_csv(f"{OUT}/joint_parity_ink_width.csv")
    digit_given_ink.to_csv(f"{OUT}/digit_given_ink.csv")


# ----------------------------------------------------------------------------
# PART 2: Autoencoder
# ----------------------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, latent=AE_LATENT):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(784, 128), nn.ReLU(), nn.Linear(128, latent))
        self.dec = nn.Sequential(nn.Linear(latent, 128), nn.ReLU(), nn.Linear(128, 784), nn.Sigmoid())

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        return self.decode(self.encode(x))


def train_autoencoder(xtr):
    print("\n" + "=" * 60)
    print("PART 2: Autoencoder (784 -> 128 -> %d -> 128 -> 784)" % AE_LATENT)
    print("=" * 60)
    ae = AutoEncoder().to(DEVICE)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loader = DataLoader(TensorDataset(xtr), batch_size=128, shuffle=True)
    losses = []
    for ep in range(AE_EPOCHS):
        total = 0.0
        for (x,) in loader:
            x = x.to(DEVICE)
            loss = F.mse_loss(ae(x), x)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        losses.append(total / len(xtr))
        print(f"  epoch {ep + 1}/{AE_EPOCHS}  loss = {losses[-1]:.5f}")
    ae.eval()
    plot_losses({"Autoencoder MSE": losses}, f"{OUT}/ae_loss.png", "Autoencoder training loss")
    return ae


# ----------------------------------------------------------------------------
# PART 3: Markov chain on digit classes (random walk in latent space)
# ----------------------------------------------------------------------------
def markov_chain_demo(ae, xtr, ytr, steps=100000):
    print("\n" + "=" * 60)
    print("PART 3: Markov Chain over digit classes")
    print("=" * 60)
    with torch.no_grad():
        z = ae.encode(xtr.to(DEVICE)).cpu()
    centroids = torch.stack([z[ytr == d].mean(0) for d in range(10)])
    dist = torch.cdist(centroids, centroids)

    # Closer digits (in latent space) are more likely to follow each other.
    off_diag = dist[~torch.eye(10, dtype=torch.bool)]
    temperature = off_diag.mean().item() / 2
    logits = -dist / temperature
    logits.fill_diagonal_(float("-inf"))          # no self transition
    P = F.softmax(logits, dim=1).numpy()          # transition matrix (rows sum to 1)

    states = [str(d) for d in range(10)]
    P_df = pd.DataFrame(P, index=states, columns=states)
    print("Transition matrix P(next | current):\n", P_df.round(3))
    assert np.allclose(P.sum(1), 1.0)

    # Sampling: walk the chain and count visits
    rng = np.random.default_rng(SEED)
    state = 0
    counts = np.zeros(10)
    for _ in range(steps):
        state = rng.choice(10, p=P[state])
        counts[state] += 1
    simulated = counts / counts.sum()

    # Exact stationary distribution: eigenvector of P^T with eigenvalue 1
    vals, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    pi = pi / pi.sum()

    table = pd.DataFrame({"Simulated": simulated, "Exact (eigenvector)": pi}, index=states)
    print("\nStationary distribution:\n", table.round(4))
    print("Check: sum simulated =", round(simulated.sum(), 4), " sum exact =", round(pi.sum(), 4))

    x = np.arange(10)
    plt.figure(figsize=(7, 4))
    plt.bar(x - 0.2, simulated, 0.4, label="Simulated")
    plt.bar(x + 0.2, pi, 0.4, label="Exact")
    plt.xticks(x, states)
    plt.xlabel("Digit state")
    plt.ylabel("Probability")
    plt.title("Markov chain stationary distribution")
    plt.legend()
    plt.savefig(f"{OUT}/markov_stationary.png", dpi=150, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5, 4.5))
    plt.imshow(P, cmap="viridis")
    plt.colorbar(label="P(next | current)")
    plt.xticks(x, states)
    plt.yticks(x, states)
    plt.xlabel("Next digit")
    plt.ylabel("Current digit")
    plt.title("Transition matrix")
    plt.savefig(f"{OUT}/markov_transition.png", dpi=150, bbox_inches="tight")
    plt.close()
    P_df.to_csv(f"{OUT}/markov_transition_matrix.csv")


# ----------------------------------------------------------------------------
# PART 4: RBM
# ----------------------------------------------------------------------------
class RBM:
    """Binary RBM trained with Contrastive Divergence (CD-1)."""

    def __init__(self, n_visible=784, n_hidden=128):
        self.W = torch.randn(n_visible, n_hidden, device=DEVICE) * 0.01
        self.vb = torch.zeros(n_visible, device=DEVICE)
        self.hb = torch.zeros(n_hidden, device=DEVICE)

    def p_h_given_v(self, v):
        return torch.sigmoid(v @ self.W + self.hb)

    def p_v_given_h(self, h):
        return torch.sigmoid(h @ self.W.t() + self.vb)

    def train(self, loader, epochs=RBM_EPOCHS, lr=0.05):
        errors = []
        for ep in range(epochs):
            total = 0.0
            for (v0,) in loader:
                v0 = v0.to(DEVICE)
                ph0 = self.p_h_given_v(v0)
                h0 = torch.bernoulli(ph0)
                pv1 = self.p_v_given_h(h0)
                ph1 = self.p_h_given_v(pv1)
                b = v0.size(0)
                self.W += lr * (v0.t() @ ph0 - pv1.t() @ ph1) / b
                self.vb += lr * (v0 - pv1).mean(0)
                self.hb += lr * (ph0 - ph1).mean(0)
                total += ((v0 - pv1) ** 2).mean().item() * b
            errors.append(total / len(loader.dataset))
        return errors

    def reconstruct(self, v):
        """v -> h (sampled) -> P(v | h). The output is the conditional probability of each pixel."""
        v = v.to(DEVICE)
        h = torch.bernoulli(self.p_h_given_v(v))
        return self.p_v_given_h(h).cpu()


def rbm_experiments(xtr, xte_eval, yte_eval, clf, true_dist):
    print("\n" + "=" * 60)
    print("PART 4: RBM with different hidden sizes", HIDDEN_SIZES)
    print("=" * 60)
    loader = DataLoader(TensorDataset(xtr), batch_size=100, shuffle=True)
    results, dists, curves = [], {}, {}
    for h in HIDDEN_SIZES:
        torch.manual_seed(SEED)
        rbm = RBM(784, h)
        errs = rbm.train(loader)
        curves[f"{h} hidden"] = errs
        recon = rbm.reconstruct(xte_eval)
        name = f"RBM-{h}"
        res, dist = evaluate_images(name, xte_eval, recon, yte_eval, clf, true_dist)
        results.append(res)
        dists[name] = dist
        print(f"  {name}: MSE={res['MSE']:.4f}  accuracy={res['Digit accuracy']:.3f}")
        both = torch.cat([xte_eval[:10], recon[:10]])
        save_grid(both, f"{OUT}/rbm_{h}_reconstruction.png", 10,
                  f"RBM-{h}: top = original, bottom = reconstructed")
    plot_losses(curves, f"{OUT}/rbm_loss.png", "RBM reconstruction error per epoch")
    return results, dists


# ----------------------------------------------------------------------------
# PART 5: GAN
# ----------------------------------------------------------------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(Z_DIM, 256), nn.LeakyReLU(0.2),
            nn.Linear(256, 512), nn.LeakyReLU(0.2),
            nn.Linear(512, 784), nn.Tanh())

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(512, 256), nn.LeakyReLU(0.2), nn.Dropout(0.3),
            nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x)


def train_gan(xtr):
    print("\n" + "=" * 60)
    print("PART 5: GAN (%d epochs)" % GAN_EPOCHS)
    print("=" * 60)
    data = xtr * 2 - 1                                  # normalize to [-1, 1]
    loader = DataLoader(TensorDataset(data), batch_size=128, shuffle=True, drop_last=True)
    G, D = Generator().to(DEVICE), Discriminator().to(DEVICE)
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    fixed_z = torch.randn(40, Z_DIM, device=DEVICE)
    g_hist, d_hist = [], []

    for ep in range(1, GAN_EPOCHS + 1):
        g_tot = d_tot = 0.0
        for (real,) in loader:
            real = real.to(DEVICE)
            b = real.size(0)
            ones = torch.ones(b, 1, device=DEVICE)
            zeros = torch.zeros(b, 1, device=DEVICE)

            # --- train Discriminator ---
            fake = G(torch.randn(b, Z_DIM, device=DEVICE))
            d_loss = bce(D(real), ones) + bce(D(fake.detach()), zeros)
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # --- train Generator ---
            g_loss = bce(D(fake), ones)
            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            g_tot += g_loss.item()
            d_tot += d_loss.item()
        g_hist.append(g_tot / len(loader))
        d_hist.append(d_tot / len(loader))
        print(f"  epoch {ep}/{GAN_EPOCHS}  D loss={d_hist[-1]:.3f}  G loss={g_hist[-1]:.3f}")
        if ep == 1 or ep % 10 == 0:
            with torch.no_grad():
                imgs = (G(fixed_z) + 1) / 2
            save_grid(imgs, f"{OUT}/gan_epoch_{ep:02d}.png", 10, f"GAN samples - epoch {ep}")

    plot_losses({"Generator": g_hist, "Discriminator": d_hist}, f"{OUT}/gan_loss.png", "GAN losses")
    G.eval()
    return G


# ----------------------------------------------------------------------------
# PART 6: Latent mixing (style-mixing idea on MNIST)
# ----------------------------------------------------------------------------
@torch.no_grad()
def latent_mixing(G, ae, xte):
    print("\n" + "=" * 60)
    print("PART 6: Latent mixing and interpolation")
    print("=" * 60)
    # (a) GAN: first k latent values from source A, the rest from source B
    z_a = torch.randn(8, Z_DIM, device=DEVICE)
    z_b = torch.randn(8, Z_DIM, device=DEVICE)
    rows = [(G(z_a) + 1) / 2]
    for cut in [25, 50, 75]:
        z_mix = torch.cat([z_a[:, :cut], z_b[:, cut:]], dim=1)
        rows.append((G(z_mix) + 1) / 2)
    rows.append((G(z_b) + 1) / 2)
    save_grid(torch.cat(rows), f"{OUT}/gan_latent_mixing.png", 8,
              "Rows: source A | cutoff 25 | cutoff 50 | cutoff 75 | source B")

    # (b) Autoencoder: smooth interpolation between two real digits
    za = ae.encode(xte[:4].to(DEVICE))
    zb = ae.encode(xte[4:8].to(DEVICE))
    alphas = torch.linspace(0, 1, 10, device=DEVICE)
    imgs = []
    for i in range(4):
        for a in alphas:
            imgs.append(ae.decode(((1 - a) * za[i] + a * zb[i]).unsqueeze(0)))
    save_grid(torch.cat(imgs), f"{OUT}/ae_interpolation.png", 10,
              "Autoencoder interpolation: left = digit A, right = digit B")
    print("  Saved gan_latent_mixing.png and ae_interpolation.png")


# ----------------------------------------------------------------------------
# PART 7: Final comparison
# ----------------------------------------------------------------------------
def final_report(results, dists, true_dist):
    print("\n" + "=" * 60)
    print("PART 7: Final comparison")
    print("=" * 60)
    table = pd.DataFrame(results).set_index("Model").round(4)
    print(table)
    table.to_csv(f"{OUT}/final_comparison.csv")

    names = ["Real test data"] + list(dists.keys())
    all_dist = [true_dist] + [dists[n] for n in dists]
    width = 0.8 / len(names)
    x = np.arange(10)
    plt.figure(figsize=(10, 4.5))
    for i, (n, d) in enumerate(zip(names, all_dist)):
        plt.bar(x + i * width, d, width, label=n)
    plt.xticks(x + 0.4 - width / 2, [str(i) for i in range(10)])
    plt.xlabel("Digit")
    plt.ylabel("Average probability")
    plt.title("Digit probability distribution of reconstructed / generated images")
    plt.legend(fontsize=8)
    plt.savefig(f"{OUT}/digit_distribution_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------
def main():
    xtr, ytr, xte, yte = load_mnist()
    xe, ye = xte[:N_EVAL], yte[:N_EVAL]
    true_dist = np.bincount(ye.numpy(), minlength=10) / len(ye)

    clf = train_classifier(xtr, ytr, xte, yte)

    probability_analysis(xtr, ytr)                         # Part 1

    ae = train_autoencoder(xtr)                            # Part 2
    with torch.no_grad():
        ae_recon = ae(xe.to(DEVICE)).cpu()
    save_grid(torch.cat([xe[:10], ae_recon[:10]]), f"{OUT}/ae_reconstruction.png", 10,
              "Autoencoder: top = original, bottom = reconstructed")

    markov_chain_demo(ae, xtr, ytr)                        # Part 3

    results, dists = [], {}
    res, dist = evaluate_images(f"Autoencoder-{AE_LATENT}", xe, ae_recon, ye, clf, true_dist)
    results.append(res)
    dists[res["Model"]] = dist

    rbm_results, rbm_dists = rbm_experiments(xtr, xe, ye, clf, true_dist)   # Part 4
    results += rbm_results
    dists.update(rbm_dists)

    G = train_gan(xtr)                                     # Part 5
    with torch.no_grad():
        fake = (G(torch.randn(N_EVAL, Z_DIM, device=DEVICE)) + 1) / 2
    res, dist = evaluate_images("GAN (generated)", None, fake, None, clf, true_dist)
    results.append(res)
    dists[res["Model"]] = dist

    latent_mixing(G, ae, xte)                              # Part 6
    final_report(results, dists, true_dist)                # Part 7
    print("\nDone. Check the '%s' folder for all images and tables." % OUT)


if __name__ == "__main__":
    main()
