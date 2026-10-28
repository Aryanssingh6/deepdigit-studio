# DeepDigit Studio — interactive web app

Streamlit UI for the DeepDigit Studio project (MNIST · probability tables, autoencoder, Markov chain,
RBM 64/128/256, GAN, latent mixing, final comparison).

## Pages
| Page | What you can do |
|---|---|
| Overview | Pipeline of the whole project + headline numbers |
| Live Playground | Pick a test digit or **upload your own**; classifier, autoencoder and 3 RBMs react together |
| 1 · Probability Lab | Choose any two variables → joint, marginal, conditional tables (sum-to-1 checks); 3-variable P(Width ∣ Parity, Ink) |
| 2 · Autoencoder | See the 16-number latent code; **edit 16 sliders and the decoder redraws the digit** |
| 3 · Markov Chain | Step through the chain, transition matrix, simulated vs exact stationary distribution, convergence curve |
| 4 · RBM | Compare 64/128/256 hidden units, hidden-layer state, learned filters, training error |
| 5 · GAN | Generate digits from a seed, watch training epoch by epoch, loss curves |
| 6 · Latent Mixing | Splice two GAN noise vectors; interpolate between two real digits with the autoencoder |
| 7 · Final Comparison | Scoreboard table, per-metric charts, digit-distribution chart |

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a new **GitHub repository** and push this whole folder (keep the `assets/` folder — it holds the trained weights, ~7 MB).
   ```bash
   git init && git add . && git commit -m "DeepDigit Studio app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/deepdigit-studio.git
   git push -u origin main
   ```
2. Go to <https://share.streamlit.io>, sign in with GitHub → **Create app** → *Deploy a public app from GitHub*.
3. Repository: `<your-username>/deepdigit-studio` · Branch: `main` · Main file path: `app.py` → **Deploy**.
4. After ~1–2 minutes you get a public `https://<name>.streamlit.app` link.

## How it works
`app.py` loads small NumPy weight files from `assets/` and runs all models (classifier, autoencoder, RBMs, GAN
generator) with plain NumPy, so the deployed app needs no PyTorch / GPU and starts in seconds.

`train_numpy.py` is the offline trainer that produced `assets/`. It re-implements the project's PyTorch script with the
same architectures and hyper-parameters (report §5). To retrain:
```bash
python train_numpy.py --mnist_dir <folder with the 4 MNIST .gz files>
```
The original project files are kept untouched in `original/`.

> Numbers shown in the app come from this NumPy run; they are very close to, but not digit-for-digit identical with,
> the PyTorch figures in the report (different random number generator).
