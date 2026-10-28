"""
Offline training script for the DeepDigit Studio web app (pure NumPy).

It re-implements the same models and hyper-parameters as deepdigit_studio.py
(classifier, autoencoder, RBM 64/128/256, GAN) without PyTorch, and saves
everything the Streamlit app needs into assets/.

Run once (takes a few minutes on CPU):
    python train_numpy.py --mnist_dir /path/with/idx_gz_files
The 4 MNIST files are: train-images-idx3-ubyte.gz, train-labels-idx1-ubyte.gz,
t10k-images-idx3-ubyte.gz, t10k-labels-idx1-ubyte.gz
"""
import argparse
import gzip
import os
import time

import numpy as np

SEED = 42
CLF_EPOCHS, AE_EPOCHS, AE_LATENT = 3, 10, 16
RBM_EPOCHS, HIDDEN_SIZES = 5, [64, 128, 256]
GAN_EPOCHS, Z_DIM, N_EVAL = 30, 100, 2000


# ----------------------------------------------------------------------------- data
def read_idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    if raw[2] == 8 and raw[3] == 3:  # images
        n, r, c = int.from_bytes(raw[4:8], "big"), int.from_bytes(raw[8:12], "big"), int.from_bytes(raw[12:16], "big")
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, r * c)
    return np.frombuffer(raw, np.uint8, offset=8)


# ----------------------------------------------------------------------------- tiny NN engine
class Dense:
    def __init__(self, i, o, rng):
        k = 1 / np.sqrt(i)
        self.W = rng.uniform(-k, k, (i, o)).astype(np.float32)
        self.b = rng.uniform(-k, k, o).astype(np.float32)
        self.m = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self.v = [np.zeros_like(self.W), np.zeros_like(self.b)]
        self.t = 0

    def forward(self, x, train=True):
        self.x = x
        return x @ self.W + self.b

    def backward(self, g):
        self.dW, self.db = self.x.T @ g, g.sum(0)
        return g @ self.W.T

    def step(self, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        for p, gr, m, v in ((self.W, self.dW, self.m[0], self.v[0]), (self.b, self.db, self.m[1], self.v[1])):
            m *= b1; m += (1 - b1) * gr
            v *= b2; v += (1 - b2) * gr * gr
            p -= lr * (m / (1 - b1 ** self.t)) / (np.sqrt(v / (1 - b2 ** self.t)) + eps)


class Act:
    def __init__(self, kind, a=0.2):
        self.kind, self.a = kind, a

    def forward(self, x, train=True):
        k = self.kind
        if k == "relu":
            self.mask = x > 0
            return x * self.mask
        if k == "leaky":
            self.mask = x > 0
            return np.where(self.mask, x, self.a * x)
        if k == "tanh":
            self.y = np.tanh(x)
            return self.y
        if k == "sigmoid":
            self.y = 1 / (1 + np.exp(-x))
            return self.y

    def backward(self, g):
        k = self.kind
        if k == "relu":
            return g * self.mask
        if k == "leaky":
            return np.where(self.mask, g, self.a * g)
        if k == "tanh":
            return g * (1 - self.y ** 2)
        if k == "sigmoid":
            return g * self.y * (1 - self.y)


class Dropout:
    def __init__(self, p, rng):
        self.p, self.rng = p, rng

    def forward(self, x, train=True):
        if not train:
            return x
        self.mask = (self.rng.random(x.shape) > self.p).astype(np.float32) / (1 - self.p)
        return x * self.mask

    def backward(self, g):
        return g * self.mask


class Net:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x, train=True):
        for l in self.layers:
            x = l.forward(x, train)
        return x

    def backward(self, g):
        for l in reversed(self.layers):
            g = l.backward(g)
        return g

    def step(self, lr, **kw):
        for l in self.layers:
            if isinstance(l, Dense):
                l.step(lr, **kw)

    def dense(self):
        return [l for l in self.layers if isinstance(l, Dense)]


def batches(n, bs, rng, drop_last=False):
    idx = rng.permutation(n)
    end = n - n % bs if drop_last else n
    for s in range(0, end, bs):
        yield idx[s:s + bs]


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def sigm(x):
    return 1 / (1 + np.exp(-x))


# ----------------------------------------------------------------------------- models
def train_classifier(xtr, ytr, xte, yte, rng):
    net = Net([Dense(784, 128, rng), Act("relu"), Dense(128, 10, rng)])
    for ep in range(CLF_EPOCHS):
        for ix in batches(len(xtr), 128, rng):
            p = softmax(net.forward(xtr[ix]))
            g = p.copy(); g[np.arange(len(ix)), ytr[ix]] -= 1
            net.backward(g / len(ix)); net.step(1e-3)
    acc = (net.forward(xte, False).argmax(1) == yte).mean()
    print(f"[Classifier] test accuracy = {acc:.4f}")
    return net, float(acc)


def train_ae(xtr, rng):
    enc = Net([Dense(784, 128, rng), Act("relu"), Dense(128, AE_LATENT, rng)])
    dec = Net([Dense(AE_LATENT, 128, rng), Act("relu"), Dense(128, 784, rng), Act("sigmoid")])
    losses = []
    for ep in range(AE_EPOCHS):
        tot = 0.0
        for ix in batches(len(xtr), 128, rng):
            x = xtr[ix]
            out = dec.forward(enc.forward(x))
            d = out - x
            tot += (d ** 2).mean() * len(ix)
            enc.backward(dec.backward(2 * d / d.size))
            enc.step(1e-3); dec.step(1e-3)
        losses.append(tot / len(xtr))
        print(f"  AE epoch {ep + 1}/{AE_EPOCHS} loss={losses[-1]:.5f}")
    return enc, dec, losses


class RBM:
    def __init__(self, nh, rng):
        self.rng = rng
        self.W = (rng.standard_normal((784, nh)) * 0.01).astype(np.float32)
        self.vb = np.zeros(784, np.float32)
        self.hb = np.zeros(nh, np.float32)

    def ph(self, v): return sigm(v @ self.W + self.hb)
    def pv(self, h): return sigm(h @ self.W.T + self.vb)

    def train(self, x, epochs=RBM_EPOCHS, lr=0.05, bs=100):
        errs = []
        for ep in range(epochs):
            tot = 0.0
            for ix in batches(len(x), bs, self.rng):
                v0 = x[ix]
                ph0 = self.ph(v0)
                h0 = (self.rng.random(ph0.shape) < ph0).astype(np.float32)
                pv1 = self.pv(h0)
                ph1 = self.ph(pv1)
                b = len(ix)
                self.W += lr * (v0.T @ ph0 - pv1.T @ ph1) / b
                self.vb += lr * (v0 - pv1).mean(0)
                self.hb += lr * (ph0 - ph1).mean(0)
                tot += ((v0 - pv1) ** 2).mean() * b
            errs.append(tot / len(x))
            print(f"    RBM-{self.W.shape[1]} epoch {ep + 1} err={errs[-1]:.5f}")
        return errs

    def reconstruct(self, v):
        ph = self.ph(v)
        h = (self.rng.random(ph.shape) < ph).astype(np.float32)
        return self.pv(h)


def train_gan(xtr, rng):
    G = Net([Dense(Z_DIM, 256, rng), Act("leaky"), Dense(256, 512, rng), Act("leaky"),
             Dense(512, 784, rng), Act("tanh")])
    D = Net([Dense(784, 512, rng), Act("leaky"), Dropout(0.3, rng), Dense(512, 256, rng), Act("leaky"),
             Dropout(0.3, rng), Dense(256, 1, rng)])
    data = (xtr * 2 - 1).astype(np.float32)
    fixed_z = rng.standard_normal((40, Z_DIM)).astype(np.float32)
    g_hist, d_hist, snaps = [], [], []
    adam = dict(b1=0.5, b2=0.999)
    for ep in range(1, GAN_EPOCHS + 1):
        t0 = time.time(); gt = dt = 0.0; nb = 0
        for ix in batches(len(data), 128, rng, drop_last=True):
            real = data[ix]; b = len(ix)
            z = rng.standard_normal((b, Z_DIM)).astype(np.float32)
            fake = G.forward(z)
            # --- discriminator: real -> 1, fake -> 0 (one pass over the concatenated batch)
            xs = np.concatenate([real, fake]); t = np.concatenate([np.ones(b), np.zeros(b)])[:, None].astype(np.float32)
            logit = D.forward(xs)
            d_loss = (np.logaddexp(0, logit) - t * logit).sum() / b
            D.backward((sigm(logit) - t) / b); D.step(2e-4, **adam)
            # --- generator: make D say "real"
            logit = D.forward(fake)
            g_loss = (np.logaddexp(0, logit) - logit).mean()
            gx = D.backward((sigm(logit) - 1) / b)
            G.backward(gx); G.step(2e-4, **adam)
            gt += g_loss; dt += d_loss; nb += 1
        g_hist.append(gt / nb); d_hist.append(dt / nb)
        snaps.append(((G.forward(fixed_z, False) + 1) / 2 * 255).clip(0, 255).astype(np.uint8))
        print(f"  GAN epoch {ep}/{GAN_EPOCHS} D={d_hist[-1]:.3f} G={g_hist[-1]:.3f}  ({time.time() - t0:.1f}s)")
    return G, g_hist, d_hist, np.stack(snaps), fixed_z


# ----------------------------------------------------------------------------- evaluation
def evaluate(name, original, recon, labels, clf, true_dist):
    probs = softmax(clf.forward(recon, False))
    dist = probs.mean(0)
    return {
        "Model": name,
        "MSE": float(((recon - original) ** 2).mean()) if original is not None else np.nan,
        "Digit accuracy": float((probs.argmax(1) == labels).mean()) if labels is not None else np.nan,
        "Mean confidence": float(probs.max(1).mean()),
        "Distribution distance (TV)": float(0.5 * np.abs(dist - true_dist).sum()),
    }, dist


def pack(prefix, net, out):
    for i, l in enumerate(net.dense()):
        out[f"{prefix}_W{i}"] = l.W
        out[f"{prefix}_b{i}"] = l.b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mnist_dir", required=True)
    ap.add_argument("--out", default="assets")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    d = a.mnist_dir
    xtr_u8 = read_idx(f"{d}/train-images-idx3-ubyte.gz"); ytr = read_idx(f"{d}/train-labels-idx1-ubyte.gz").astype(np.int64)
    xte_u8 = read_idx(f"{d}/t10k-images-idx3-ubyte.gz"); yte = read_idx(f"{d}/t10k-labels-idx1-ubyte.gz").astype(np.int64)
    xtr, xte = xtr_u8.astype(np.float32) / 255, xte_u8.astype(np.float32) / 255
    xe, ye = xte[:N_EVAL], yte[:N_EVAL]
    true_dist = np.bincount(ye, minlength=10) / len(ye)

    W = {}
    clf, clf_acc = train_classifier(xtr, ytr, xte, yte, rng); pack("clf", clf, W)

    # ---- Part 1 categories (same definitions as the report) ----
    ink = xtr.sum(1)
    q = np.quantile(ink, [1 / 3, 2 / 3])
    ink_level = np.digitize(ink, q, right=True)          # 0 Low, 1 Medium, 2 High
    width = (xtr.reshape(-1, 28, 28) > 0.3).any(1).sum(1)
    wide = (width > np.median(width)).astype(np.int8)    # 1 = Wide

    enc, dec, ae_loss = train_ae(xtr, rng); pack("enc", enc, W); pack("dec", dec, W)
    ae_recon = dec.forward(enc.forward(xe, False), False)
    results, dists = [], {}
    r, dist = evaluate(f"Autoencoder-{AE_LATENT}", xe, ae_recon, ye, clf, true_dist); results.append(r); dists[r["Model"]] = dist

    z_tr = enc.forward(xtr, False)
    centroids = np.stack([z_tr[ytr == k].mean(0) for k in range(10)])

    rbm_err = {}
    for h in HIDDEN_SIZES:
        rbm = RBM(h, np.random.default_rng(SEED))
        rbm_err[h] = np.array(rbm.train(xtr))
        W[f"rbm{h}_W"], W[f"rbm{h}_vb"], W[f"rbm{h}_hb"] = rbm.W, rbm.vb, rbm.hb
        r, dist = evaluate(f"RBM-{h}", xe, rbm.reconstruct(xe), ye, clf, true_dist); results.append(r); dists[r["Model"]] = dist
        print("   ", r)

    G, g_hist, d_hist, snaps, fixed_z = train_gan(xtr, rng); pack("gen", G, W)
    fake = (G.forward(rng.standard_normal((N_EVAL, Z_DIM)).astype(np.float32), False) + 1) / 2
    r, dist = evaluate("GAN (generated)", None, fake, None, clf, true_dist); results.append(r); dists["GAN (generated)"] = dist

    np.savez_compressed(
        f"{a.out}/model_weights.npz", **W, centroids=centroids, ae_loss=np.array(ae_loss),
        rbm64_err=rbm_err[64], rbm128_err=rbm_err[128], rbm256_err=rbm_err[256],
        gan_g=np.array(g_hist), gan_d=np.array(d_hist), gan_snaps=snaps, gan_fixed_z=fixed_z,
        clf_acc=clf_acc, true_dist=true_dist,
        metrics_names=np.array([r["Model"] for r in results]),
        metrics=np.array([[r["MSE"], r["Digit accuracy"], r["Mean confidence"], r["Distribution distance (TV)"]] for r in results]),
        dist_names=np.array(list(dists)), dist_vals=np.stack(list(dists.values())),
    )
    np.savez_compressed(f"{a.out}/data.npz", xte=xte_u8, yte=yte.astype(np.uint8), ytr=ytr.astype(np.uint8),
                        ink=ink_level.astype(np.uint8), wide=wide)
    print("Saved assets. Metrics:")
    for r in results:
        print(r)


if __name__ == "__main__":
    main()
