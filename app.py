"""
DeepDigit Studio - interactive web app
======================================
A Streamlit front-end that shows how every part of the DeepDigit Studio project works:
probability tables, autoencoder, Markov chain, RBM, GAN, latent mixing and the final
comparison. Inference runs in pure NumPy on weights trained by train_numpy.py, so the
app starts instantly and needs no GPU / PyTorch.
"""
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

ASSETS = Path(__file__).parent / "assets"
ACCENT, TEAL, VIOLET, ROSE, LIME, MUTED = "#ffb454", "#5ad1c5", "#8b8cf7", "#f0657a", "#9bd36a", "#5b6577"
SERIES = [ACCENT, TEAL, VIOLET, ROSE, LIME, "#c9a0ff"]
PAGES = ["Overview", "Live Playground", "1 · Probability Lab", "2 · Autoencoder", "3 · Markov Chain",
         "4 · RBM", "5 · GAN", "6 · Latent Mixing", "7 · Final Comparison"]

st.set_page_config(page_title="DeepDigit Studio", page_icon="🔢", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=JetBrains+Mono:wght@400;600&display=swap');
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-weight: 600; }
[data-testid="stMetric"] { background: #171a21; border: 1px solid #262b36; border-radius: 12px; padding: 14px 16px; }
.block-container { padding-top: 2.2rem; max-width: 1250px; }
.hero { padding: 6px 0 18px 0; }
.hero h1 { font-size: 3.1rem; margin: 0; line-height: 1.05; color: #e8e6e1; }
.hero h1 span { color: #ffb454; }
.hero p { color: #a8aab3; font-size: 1.12rem; max-width: 720px; margin: 10px 0 0 0; }
.badge { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:.74rem; padding:3px 10px;
         border:1px solid #3a4152; border-radius:999px; color:#ffb454; margin-right:6px; }
.flow { display:grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 8px 0 4px 0; }
.card { background:#171a21; border:1px solid #262b36; border-radius:12px; padding:14px 16px; }
.card b { color:#ffb454; font-family:'JetBrains Mono',monospace; font-size:.8rem; }
.card h4 { margin:4px 0 4px 0; font-family:'Space Grotesk',sans-serif; font-size:1.05rem; }
.card p { margin:0; color:#a8aab3; font-size:.88rem; line-height:1.4; }
@media (max-width: 900px) { .flow { grid-template-columns: repeat(2, 1fr); } .hero h1 { font-size: 2.3rem; } }
.mono { font-family:'JetBrains Mono',monospace; }
.walk { font-family:'JetBrains Mono',monospace; font-size:1.05rem; color:#ffb454; word-break:break-word; }
</style>
""",
    unsafe_allow_html=True,
)


# ----------------------------------------------------------------------------- loading + numpy inference
@st.cache_resource(show_spinner="Loading models…")
def load():
    w = dict(np.load(ASSETS / "model_weights.npz", allow_pickle=False))
    d = dict(np.load(ASSETS / "data.npz", allow_pickle=False))
    xte = d["xte"].astype(np.float32) / 255
    return w, d, xte


W, DATA, XTE = load()
YTE = DATA["yte"].astype(int)
N_EVAL = 2000


def relu(x): return np.maximum(x, 0)
def lrelu(x): return np.where(x > 0, x, 0.2 * x)
def sig(x): return 1 / (1 + np.exp(-x))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def judge(x):
    """Helper classifier (784-128-10) -> softmax probabilities."""
    x = np.atleast_2d(x)
    return softmax(relu(x @ W["clf_W0"] + W["clf_b0"]) @ W["clf_W1"] + W["clf_b1"])


def encode(x):
    return relu(np.atleast_2d(x) @ W["enc_W0"] + W["enc_b0"]) @ W["enc_W1"] + W["enc_b1"]


def decode(z):
    return sig(relu(np.atleast_2d(z) @ W["dec_W0"] + W["dec_b0"]) @ W["dec_W1"] + W["dec_b1"])


def generate(z):
    h = lrelu(np.atleast_2d(z) @ W["gen_W0"] + W["gen_b0"])
    h = lrelu(h @ W["gen_W1"] + W["gen_b1"])
    return (np.tanh(h @ W["gen_W2"] + W["gen_b2"]) + 1) / 2


def rbm_hidden(x, hsize, seed):
    rng = np.random.default_rng(seed)
    ph = sig(np.atleast_2d(x) @ W[f"rbm{hsize}_W"] + W[f"rbm{hsize}_hb"])
    return (rng.random(ph.shape) < ph).astype(np.float32), ph


def rbm_recon(x, hsize, seed=0):
    h, _ = rbm_hidden(x, hsize, seed)
    return sig(h @ W[f"rbm{hsize}_W"].T + W[f"rbm{hsize}_vb"])


@st.cache_data
def test_latents():
    return encode(XTE)


# ----------------------------------------------------------------------------- UI helpers
def grid(imgs, ncols, scale=4, pad=2, bg=22):
    """Tile flattened 28x28 images (values 0..1) into one upscaled picture."""
    imgs = np.asarray(imgs, dtype=np.float32).reshape(-1, 28, 28)
    nrows = int(np.ceil(len(imgs) / ncols))
    H, Wd = nrows * (28 + pad) + pad, ncols * (28 + pad) + pad
    canvas = np.full((H, Wd), bg / 255, np.float32)
    for i, im in enumerate(imgs):
        r, c = divmod(i, ncols)
        canvas[pad + r * (28 + pad): pad + r * (28 + pad) + 28, pad + c * (28 + pad): pad + c * (28 + pad) + 28] = im
    return Image.fromarray((canvas.clip(0, 1) * 255).astype(np.uint8)).resize((Wd * scale, H * scale), Image.NEAREST)


def show_digit(x, width=150, caption=None):
    st.image(grid(x, 1, scale=6, pad=0, bg=0), width=width, caption=caption)


def bar_probs(probs, highlight=None, height=150):
    df = pd.DataFrame({"digit": list(range(10)), "p": np.asarray(probs, dtype=float)})
    hl = -1 if highlight is None else int(highlight)
    return (
        alt.Chart(df).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("digit:O", title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("p:Q", title=None, scale=alt.Scale(domain=[0, 1])),
            color=alt.condition(alt.datum.digit == hl, alt.value(ACCENT), alt.value(MUTED)),
            tooltip=["digit:O", alt.Tooltip("p:Q", format=".4f")],
        ).properties(height=height)
    )


def line_chart(df, x, y, color, title=None, height=260, logx=False):
    return (
        alt.Chart(df).mark_line(point=alt.OverlayMarkDef(size=28), strokeWidth=2.2)
        .encode(
            x=alt.X(x, scale=alt.Scale(type="log") if logx else alt.Undefined),
            y=alt.Y(y, scale=alt.Scale(zero=False)),
            color=alt.Color(color, scale=alt.Scale(range=SERIES), legend=alt.Legend(orient="top", title=None)),
            tooltip=list(df.columns),
        ).properties(height=height, title=title or "")
    )


def heatmap(df, xtitle, ytitle, height=None, fmt=".3f"):
    t = df.copy()
    t.index = [str(i) for i in t.index]
    t.columns = [str(c) for c in t.columns]
    long = t.stack().rename("p").reset_index()
    long.columns = ["r", "c", "p"]
    vmax = max(float(long["p"].max()), 1e-9)
    base = alt.Chart(long).encode(
        x=alt.X("c:O", sort=list(t.columns), title=xtitle, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("r:O", sort=list(t.index), title=ytitle),
    )
    rect = base.mark_rect(cornerRadius=3, stroke="#0f1115", strokeWidth=2).encode(
        color=alt.Color("p:Q", scale=alt.Scale(domain=[0, vmax], range=["#1b2130", ACCENT]), legend=None),
        tooltip=["r", "c", alt.Tooltip("p:Q", format=".4f")],
    )
    text = base.mark_text(fontSize=12, font="JetBrains Mono").encode(
        text=alt.Text("p:Q", format=fmt),
        color=alt.condition(alt.datum.p > 0.55 * vmax, alt.value("#14161c"), alt.value("#e8e6e1")),
    )
    return (rect + text).properties(height=height or max(160, 46 * len(t)))


def page_header(badge, title, blurb):
    st.markdown(f'<span class="badge">{badge}</span>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f"<p style='color:#a8aab3;max-width:820px;font-size:1.05rem'>{blurb}</p>", unsafe_allow_html=True)


def digit_picker(prefix, default=0):
    """Pick a MNIST test image: by class + random button, or by index. Returns the index."""
    st.session_state.setdefault(f"{prefix}_idx", default)
    st.session_state.setdefault(f"{prefix}_cls", "Any")

    def _rand():
        c = st.session_state[f"{prefix}_cls"]
        pool = np.arange(len(YTE)) if c == "Any" else np.where(YTE == int(c))[0]
        st.session_state[f"{prefix}_idx"] = int(np.random.choice(pool))

    c1, c2, c3 = st.columns([1, 1, 1.2])
    c1.selectbox("Digit class", ["Any"] + [str(i) for i in range(10)], key=f"{prefix}_cls")
    c2.number_input("Test image #", 0, len(YTE) - 1, key=f"{prefix}_idx")
    c3.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
    c3.button("🎲 Random image", key=f"{prefix}_btn", on_click=_rand, width="stretch")
    return int(st.session_state[f"{prefix}_idx"])


def preprocess_upload(file):
    """Turn an uploaded photo/drawing into an MNIST-style 28x28 image (white digit on black)."""
    im = Image.open(file).convert("L")
    a = np.asarray(im, dtype=np.float32)
    if a.mean() > 127:  # dark ink on white paper -> invert
        a = 255 - a
    ys, xs = np.where(a > 60)
    if len(ys) == 0:
        return None
    a = a[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    h, w = a.shape
    s = 20 / max(h, w)
    small = Image.fromarray(a.astype(np.uint8)).resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(small, ((28 - small.width) // 2, (28 - small.height) // 2))
    out = np.asarray(canvas, dtype=np.float32) / 255
    out = out / max(out.max(), 1e-6)
    # MNIST convention: centre the digit by its centre of mass, not by its bounding box
    yy, xx = np.mgrid[0:28, 0:28]
    m = out.sum()
    cy, cx = (yy * out).sum() / m, (xx * out).sum() / m
    out = np.roll(out, (int(round(14 - cy)), int(round(14 - cx))), axis=(0, 1))
    return out.reshape(784)


# ----------------------------------------------------------------------------- Part 1 data
ORDER = {"Parity": ["Even", "Odd"], "Ink": ["Low", "Medium", "High"], "Width": ["Narrow", "Wide"], "Digit": list(range(10))}


@st.cache_data
def prob_frame():
    y = DATA["ytr"].astype(int)
    return pd.DataFrame({
        "Digit": y,
        "Parity": np.where(y % 2 == 0, "Even", "Odd"),
        "Ink": np.array(ORDER["Ink"])[DATA["ink"]],
        "Width": np.where(DATA["wide"] == 1, "Wide", "Narrow"),
    })


# ----------------------------------------------------------------------------- Part 3 helpers
@st.cache_data
def markov_matrix():
    c = W["centroids"].astype(np.float64)
    dist = np.linalg.norm(c[:, None] - c[None], axis=2)
    off = dist[~np.eye(10, dtype=bool)]
    temp = off.mean() / 2
    logits = -dist / temp
    np.fill_diagonal(logits, -np.inf)
    P = softmax(logits)
    vals, vecs = np.linalg.eig(P.T)
    pi = np.real(vecs[:, np.argmin(np.abs(vals - 1))])
    return P, pi / pi.sum(), temp


@st.cache_data
def simulate(steps, seed):
    P, pi, _ = markov_matrix()
    cdf = P.cumsum(1)
    u = np.random.default_rng(seed).random(steps)
    states = np.empty(steps, np.int8)
    s = 0
    for i in range(steps):
        s = min(int(np.searchsorted(cdf[s], u[i])), 9)
        states[i] = s
    return states


# ============================================================================= PAGES
def page_overview():
    st.markdown(
        """<div class="hero"><span class="badge">DEEP LEARNING II · MNIST</span>
        <h1>DeepDigit <span>Studio</span></h1>
        <p>How can a computer learn what data looks like? Seven ideas — probability tables, autoencoders,
        Markov chains, RBMs, GANs and latent mixing — all run on the same handwritten digits and scored on one scoreboard.</p></div>""",
        unsafe_allow_html=True,
    )
    first =[int(np.where(YTE == d)[0][0]) for d in range(10)]
    st.image(grid(XTE[first], 10, scale=6), width="stretch")

    m = W["metrics"]
    names = list(W["metrics_names"])
    ae, r256, gan = names.index("Autoencoder-16"), names.index("RBM-256"), names.index("GAN (generated)")
    c = st.columns(4)
    c[0].metric("Helper classifier accuracy", f"{float(W['clf_acc']) * 100:.2f}%", help="The 'judge' used to score every other model")
    c[1].metric("Autoencoder MSE (16-d latent)", f"{m[ae, 0]:.4f}")
    c[2].metric("RBM-256 digit accuracy", f"{m[r256, 1] * 100:.1f}%")
    c[3].metric("GAN mean confidence", f"{m[gan, 2]:.3f}")

    st.subheader("The pipeline")
    st.markdown(
        """
<div class="flow">
<div class="card"><b>DATA</b><h4>MNIST</h4><p>70,000 digits, 28×28 px → 784 numbers scaled to 0–1. 60k train / 10k test.</p></div>
<div class="card"><b>JUDGE</b><h4>Helper classifier</h4><p>784-128-10 network. Scores every reconstruction and generated image.</p></div>
<div class="card"><b>PART 1</b><h4>Probability Lab</h4><p>Joint, marginal &amp; conditional tables for Parity, Ink, Width.</p></div>
<div class="card"><b>PART 2</b><h4>Autoencoder</h4><p>Squeezes a digit into 16 numbers and rebuilds it.</p></div>
<div class="card"><b>PART 3</b><h4>Markov chain</h4><p>10 digit-states; transitions from latent-space distances. Sampled vs exact stationary distribution.</p></div>
<div class="card"><b>PART 4</b><h4>RBM ×3</h4><p>64 / 128 / 256 hidden units trained with contrastive divergence.</p></div>
<div class="card"><b>PART 5</b><h4>GAN</h4><p>Generator vs discriminator — creates brand-new digits from noise.</p></div>
<div class="card"><b>PART 6–7</b><h4>Mixing &amp; scoreboard</h4><p>Latent mixing, interpolation and a final same-measure comparison.</p></div>
</div>""",
        unsafe_allow_html=True,
    )
    with st.expander("About these numbers"):
        st.write(
            "Models in this app were trained with `train_numpy.py`, a NumPy re-implementation of the project's PyTorch script "
            "with the same architectures and hyper-parameters (see report §5). Random seeds differ from the PyTorch run, so results "
            "are very close to — but not digit-for-digit identical with — the figures printed in the report."
        )


def page_playground():
    page_header("LIVE", "Live Playground", "Pick any MNIST test digit — or upload your own — and watch the judge, the autoencoder and the three RBMs process it at the same time.")
    src = st.radio("Input", ["MNIST test image", "Upload my own digit"], horizontal=True, label_visibility="collapsed")
    x, truth = None, None
    if src == "MNIST test image":
        idx = digit_picker("pg", default=7)
        x, truth = XTE[idx], YTE[idx]
    else:
        f = st.file_uploader("Upload a picture of a single handwritten digit (PNG / JPG)", type=["png", "jpg", "jpeg"])
        st.caption("Tip: a dark pen on white paper works best. The image is cropped, resized to 20×20 and centred in 28×28, exactly like MNIST.")
        if f is not None:
            x = preprocess_upload(f)
            if x is None:
                st.warning("Couldn't find any ink in that image.")
        if x is None:
            return
    seed = st.slider("RBM sampling seed (RBMs sample their hidden units, so every seed gives a slightly different rebuild)", 0, 50, 0)

    p = judge(x)[0]
    ae = decode(encode(x))[0]
    rb = {h: rbm_recon(x, h, seed)[0] for h in (64, 128, 256)}

    cols = st.columns([1, 1, 1, 1, 1])
    items = [("Original", x, None), ("Autoencoder (16-d)", ae, None)] + [(f"RBM-{h}", rb[h], None) for h in rb]
    for col, (name, img, _) in zip(cols, items):
        with col:
            st.markdown(f"**{name}**")
            show_digit(img, width=140)
            pr = judge(img)[0]
            mse = "" if name == "Original" else f"MSE `{np.mean((img - x) ** 2):.4f}` · "
            st.markdown(f"{mse}judge: **{pr.argmax()}** ({pr.max() * 100:.0f}%)")

    st.subheader("What the judge thinks of the original")
    left, right = st.columns([2, 1])
    left.altair_chart(bar_probs(p, p.argmax()), width="stretch")
    with right:
        st.metric("Predicted digit", int(p.argmax()), delta=None if truth is None else ("✓ correct" if truth == p.argmax() else f"✗ true label {truth}"),
                  delta_color="off")
        st.metric("Confidence", f"{p.max() * 100:.1f}%")
    st.caption("The autoencoder squeezes the digit into just 16 numbers; each RBM samples a binary hidden code (64/128/256 units) and rebuilds pixel probabilities. A better model keeps the judge's answer unchanged while MSE stays low.")


def page_probability():
    page_header("PART 1 · PRACTICALS 1-3", "Probability Lab", "Every image is turned into simple categories — Parity (from the label), Ink (total brightness in 3 equal groups) and Width (columns containing ink vs. the median). Count co-occurrences, divide by 60,000 and you have a joint distribution.")
    df = prob_frame()
    tab2, tab3, tab4 = st.tabs(["Two variables", "Three variables", "P(Digit | Ink)"])

    with tab2:
        c = st.columns(3)
        a = c[0].selectbox("Rows (X)", ["Parity", "Width", "Ink", "Digit"], index=0)
        b = c[1].selectbox("Columns (Y)", [v for v in ["Ink", "Width", "Parity", "Digit"] if v != a], index=0)
        cond = c[2].radio("Conditional view", [f"P({b} | {a})", f"P({a} | {b})"], horizontal=False)
        joint = pd.crosstab(df[a], df[b], normalize=True).reindex(index=ORDER[a], columns=ORDER[b]).fillna(0)
        n1, n2 = st.columns(2)
        with n1:
            st.markdown(f"##### Joint  P({a}, {b})")
            st.altair_chart(heatmap(joint, b, a), width="stretch")
            st.success(f"Σ P(x, y) = {joint.values.sum():.6f}  ✓ sums to 1")
        with n2:
            st.markdown(f"##### Conditional  {cond}")
            if cond.startswith(f"P({b} |"):
                c_df = joint.div(joint.sum(axis=1), axis=0)
                st.altair_chart(heatmap(c_df, b, a), width="stretch")
                st.caption(f"Each **row** sums to 1:  P({b}|{a}) = P({a},{b}) / P({a})")
            else:
                c_df = joint.div(joint.sum(axis=0), axis=1)
                st.altair_chart(heatmap(c_df, b, a), width="stretch")
                st.caption(f"Each **column** sums to 1:  P({a}|{b}) = P({a},{b}) / P({b})")
        st.markdown("##### Marginals")
        m1, m2 = st.columns(2)
        for col, name, vals in ((m1, a, joint.sum(axis=1)), (m2, b, joint.sum(axis=0))):
            d = pd.DataFrame({name: [str(i) for i in vals.index], "P": vals.values})
            col.altair_chart(
                alt.Chart(d).mark_bar(color=TEAL, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
                .encode(x=alt.X(f"{name}:O", sort=list(d[name]), axis=alt.Axis(labelAngle=0)), y=alt.Y("P:Q", scale=alt.Scale(domain=[0, max(d.P) * 1.2])),
                        tooltip=[name, alt.Tooltip("P:Q", format=".4f")]).properties(height=170, title=f"P({name})"),
                width="stretch")

    with tab3:
        st.markdown("Fix **Parity** and **Ink** and see how the distribution of **Width** changes — the same idea as factorising a joint distribution in a Bayesian / Markov network:  `P(Width | Parity, Ink) = P(Parity, Ink, Width) / P(Parity, Ink)`")
        c = st.columns(2)
        par = c[0].radio("Parity =", ORDER["Parity"], horizontal=True)
        ink = c[1].radio("Ink =", ORDER["Ink"], horizontal=True, index=2)
        j3 = df.groupby(["Parity", "Ink", "Width"]).size() / len(df)
        sub = j3.loc[(par, ink)].reindex(ORDER["Width"]).fillna(0)
        den = sub.sum()
        t = pd.DataFrame({"Width": ORDER["Width"], "P(Parity, Ink, Width)": sub.values, "P(Parity, Ink)": den, "P(Width | Parity, Ink)": (sub / den).values})
        l, r = st.columns([1.2, 1])
        l.dataframe(t.style.format({k: "{:.4f}" for k in t.columns[1:]}), hide_index=True, width="stretch")
        l.success(f"Σ P(x, y, z) over all 12 cells = {j3.sum():.6f}  ✓")
        r.altair_chart(
            alt.Chart(t).mark_bar(color=ACCENT, cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(x=alt.X("Width:O", axis=alt.Axis(labelAngle=0)), y=alt.Y("P(Width | Parity, Ink):Q", scale=alt.Scale(domain=[0, 1]), title="P(Width | Parity, Ink)"),
                    tooltip=["Width", alt.Tooltip("P(Width | Parity, Ink):Q", format=".4f")]).properties(height=220),
            width="stretch")
        full = j3.rename("P").reset_index()
        with st.expander("Full 3-variable joint table (12 cells)"):
            st.dataframe(full.style.format({"P": "{:.4f}"}), hide_index=True, width="stretch")

    with tab4:
        dgi = pd.crosstab(df["Digit"], df["Ink"], normalize="columns").reindex(columns=ORDER["Ink"])
        st.markdown("Which digits use a lot of ink? Each **column** sums to 1 — given the ink level, how likely is each digit?")
        st.altair_chart(heatmap(dgi, "Ink level", "Digit", height=420), width="stretch")


def page_autoencoder():
    page_header("PART 2 · PRACTICAL 6", "Autoencoder", "Encoder: 784 → 128 → 16.  Decoder: 16 → 128 → 784. Forced to rebuild a digit from only 16 numbers (the latent code), the network keeps what matters and drops the rest.")
    idx = digit_picker("ae", default=0)
    x = XTE[idx]
    z = encode(x)[0]
    rec = decode(z)[0]
    pr = judge(rec)[0]
    c = st.columns([1, 1, 2.4])
    with c[0]:
        st.markdown("**Original**"); show_digit(x, 170)
    with c[1]:
        st.markdown("**Reconstruction**"); show_digit(rec, 170)
    with c[2]:
        st.markdown("**The 16-number latent code**")
        zdf = pd.DataFrame({"unit": [f"z{i + 1}" for i in range(16)], "value": z})
        st.altair_chart(
            alt.Chart(zdf).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
            .encode(x=alt.X("unit:O", sort=list(zdf.unit), title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y("value:Q", title=None),
                    color=alt.condition(alt.datum.value > 0, alt.value(ACCENT), alt.value(TEAL)), tooltip=["unit", alt.Tooltip("value:Q", format=".3f")]).properties(height=170),
            width="stretch")
        st.markdown(f"MSE `{np.mean((rec - x) ** 2):.4f}` · judge reads the rebuild as **{pr.argmax()}** ({pr.max() * 100:.0f}%) — true label **{YTE[idx]}**")

    st.subheader("Be the decoder")
    st.caption("These sliders start at the encoder's 16 numbers for the chosen digit. Move them and the decoder draws a new image instantly — each number controls some feature of the stroke.")
    Z = test_latents()
    lo, hi = np.percentile(Z, 0.5, axis=0), np.percentile(Z, 99.5, axis=0)
    zs = np.zeros(16, dtype=np.float32)
    cols = st.columns(4)
    for j in range(16):
        lo_j, hi_j = float(min(lo[j], z[j])), float(max(hi[j], z[j]))
        zs[j] = cols[j % 4].slider(f"z{j + 1}", lo_j, hi_j, float(z[j]), step=0.01, key=f"aez_{idx}_{j}")
    edit = decode(zs)[0]
    pe = judge(edit)[0]
    a, b = st.columns([1, 3])
    with a:
        show_digit(edit, 170, "Decoded from your sliders")
    with b:
        st.markdown(f"Judge reads this as **{pe.argmax()}** with {pe.max() * 100:.0f}% confidence")
        st.altair_chart(bar_probs(pe, pe.argmax(), height=140), width="stretch")

    st.subheader("Training")
    loss = pd.DataFrame({"epoch": np.arange(1, len(W["ae_loss"]) + 1), "MSE loss": W["ae_loss"], "series": "Autoencoder"})
    st.altair_chart(line_chart(loss, "epoch:Q", "MSE loss:Q", "series:N", "Training loss per epoch (10 epochs, Adam lr 1e-3)"), width="stretch")


def page_markov():
    P, pi, temp = markov_matrix()
    page_header("PART 3 · PRACTICAL 5", "Markov Chain over digits", "The ten digits are the states. Digits whose average latent codes (centroids) are close get a higher chance of following each other: softmax of negative distance, with no self-transitions. Walk the chain long enough and visit frequencies settle to a fixed distribution.")
    protos = decode(W["centroids"])
    st.markdown("##### The 10 states — each digit's centroid, decoded back into an image")
    st.image(grid(protos, 10, scale=6), width="stretch")

    st.subheader("Take a walk, one step at a time")
    st.session_state.setdefault("mc_hist", [0])
    st.session_state.setdefault("mc_rng", np.random.default_rng(0))
    b1, b2, b3 = st.columns([1, 1, 4])
    if b1.button("▶ Step", width="stretch"):
        cur = st.session_state.mc_hist[-1]
        st.session_state.mc_hist.append(int(st.session_state.mc_rng.choice(10, p=P[cur])))
    if b2.button("⏩ +20", width="stretch"):
        for _ in range(20):
            cur = st.session_state.mc_hist[-1]
            st.session_state.mc_hist.append(int(st.session_state.mc_rng.choice(10, p=P[cur])))
    if b3.button("↺ Reset"):
        st.session_state.mc_hist = [0]
    hist = st.session_state.mc_hist
    cur = hist[-1]
    l, r = st.columns([1.3, 1])
    with l:
        tail = hist[-24:]
        st.markdown(f"<div class='walk'>{' → '.join(map(str, tail))}</div>", unsafe_allow_html=True)
        ncol = min(12, len(tail))
        st.image(grid(protos[tail], ncol, scale=4), width=ncol * 62 + 8)
        st.caption(f"{len(hist) - 1} steps taken. Currently in state **{cur}**.")
    with r:
        st.markdown(f"**P(next | current = {cur})** — the row of the transition matrix")
        st.altair_chart(bar_probs(P[cur], None, height=170), width="stretch")

    st.subheader("Transition matrix & stationary distribution")
    c = st.columns([1, 1.2])
    steps = c[0].select_slider("Walk length", [1_000, 5_000, 10_000, 50_000, 100_000, 200_000], value=100_000, format_func=lambda v: f"{v:,}")
    seed = c[1].number_input("Random seed", 0, 9999, 42)
    sim_states = simulate(int(steps), int(seed))
    sim = np.bincount(sim_states, minlength=10) / steps
    left, right = st.columns(2)
    with left:
        st.markdown("**Transition matrix** P(next | current)")
        st.altair_chart(heatmap(pd.DataFrame(P, index=range(10), columns=range(10)), "Next digit", "Current digit", height=380, fmt=".2f"), width="stretch")
    with right:
        st.markdown("**Stationary distribution** — simulated vs exact (eigenvector of Pᵀ, eigenvalue 1)")
        d = pd.DataFrame({"digit": list(range(10)) * 2, "P": np.concatenate([sim, pi]), "kind": ["Simulated"] * 10 + ["Exact"] * 10})
        st.altair_chart(
            alt.Chart(d).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
                x=alt.X("digit:O", axis=alt.Axis(labelAngle=0), title="Digit state"), xOffset="kind:N", y=alt.Y("P:Q", title=None),
                color=alt.Color("kind:N", scale=alt.Scale(domain=["Simulated", "Exact"], range=[ACCENT, TEAL]), legend=alt.Legend(orient="top", title=None)),
                tooltip=["digit:O", "kind:N", alt.Tooltip("P:Q", format=".4f")]).properties(height=330),
            width="stretch")
    m = st.columns(4)
    m[0].metric("Max |simulated − exact|", f"{np.abs(sim - pi).max():.4f}")
    m[1].metric("Most visited (exact)", f"digit {pi.argmax()}  ({pi.max():.4f})")
    m[2].metric("Least visited (exact)", f"digit {pi.argmin()}  ({pi.min():.4f})")
    m[3].metric("Row sums of P", f"{P.sum(1).min():.3f} – {P.sum(1).max():.3f}")

    oh = np.zeros((len(sim_states), 10), np.float32)
    oh[np.arange(len(sim_states)), sim_states] = 1
    cum = oh.cumsum(0)
    pts = np.unique(np.geomspace(10, len(sim_states), 40).astype(int))
    tv = [0.5 * np.abs(cum[n - 1] / n - pi).sum() for n in pts]
    st.altair_chart(line_chart(pd.DataFrame({"steps": pts, "distance to exact (TV)": tv, "series": "TV distance"}), "steps:Q", "distance to exact (TV):Q", "series:N",
                               "Convergence: the longer the walk, the closer to the exact distribution", height=240, logx=True), width="stretch")
    st.caption(f"Temperature used for the softmax = mean off-diagonal centroid distance / 2 = {temp:.3f}. The matrix is a modelling choice, not observed digit sequences.")


def page_rbm():
    page_header("PART 4 · PRACTICAL 4", "Restricted Boltzmann Machine", "Visible layer = 784 pixels, hidden layer = 64 / 128 / 256 units, no connections inside a layer. Trained with 1-step contrastive divergence. To rebuild an image: pixels → P(h|v) → sample h → P(v|h).")
    idx = digit_picker("rbm", default=0)
    seed = st.slider("Sampling seed", 0, 50, 0, key="rbm_seed")
    n = st.slider("How many consecutive test digits to show", 5, 16, 10)
    idxs = np.arange(idx, idx + n) % len(XTE)
    rows = [XTE[idxs]] + [rbm_recon(XTE[idxs], h, seed) for h in (64, 128, 256)]
    st.image(grid(np.concatenate(rows), n, scale=5), width="stretch")
    st.caption("Rows from top: original · RBM-64 · RBM-128 · RBM-256. More hidden units → sharper rebuild.")
    mse = [float(np.mean((rows[i + 1] - rows[0]) ** 2)) for i in range(3)]
    c = st.columns(3)
    for col, h, m_ in zip(c, (64, 128, 256), mse):
        col.metric(f"RBM-{h} MSE on these {n} digits", f"{m_:.4f}")

    st.subheader("Inside the hidden layer")
    hs = st.radio("Hidden size", [64, 128, 256], horizontal=True, index=1)
    h, ph = rbm_hidden(XTE[idx], hs, seed)
    a, b = st.columns([1, 1.4])
    with a:
        st.markdown(f"**Sampled hidden code h** for test image #{idx} (white = unit on)")
        side = 8 if hs == 64 else 16
        hv = h[0].reshape(-1, side)
        img = Image.fromarray((hv * 255).astype(np.uint8)).resize((hv.shape[1] * 14, hv.shape[0] * 14), Image.NEAREST)
        st.image(img, width=hv.shape[1] * 14)
        st.caption(f"{int(h.sum())} of {hs} units switched on.")
    with b:
        st.markdown("**What the hidden units look for** — weights of the first 64 units, drawn as 28×28 images")
        Wm = W[f"rbm{hs}_W"][:, :64].T.copy()
        Wm = (Wm - Wm.min(1, keepdims=True)) / (np.ptp(Wm, axis=1, keepdims=True) + 1e-9)
        st.image(grid(Wm, 16, scale=2, pad=1), width="stretch")

    st.subheader("Training error & scoreboard")
    err = pd.concat([pd.DataFrame({"epoch": np.arange(1, 6), "reconstruction error": W[f"rbm{h}_err"], "model": f"{h} hidden"}) for h in (64, 128, 256)])
    l, r = st.columns([1.3, 1])
    l.altair_chart(line_chart(err, "epoch:Q", "reconstruction error:Q", "model:N", "Reconstruction error per epoch (CD-1, lr 0.05, 5 epochs)"), width="stretch")
    names = list(W["metrics_names"])
    tab = pd.DataFrame(W["metrics"][[names.index(f"RBM-{h}") for h in (64, 128, 256)]], index=[f"RBM-{h}" for h in (64, 128, 256)],
                       columns=["MSE", "Digit accuracy", "Mean confidence", "TV distance"])
    r.markdown("**Evaluation on 2,000 test images**")
    r.dataframe(tab.style.format("{:.4f}").highlight_min(subset=["MSE", "TV distance"], color="#3a5a2a").highlight_max(subset=["Digit accuracy", "Mean confidence"], color="#3a5a2a"), width="stretch")


def page_gan():
    page_header("PART 5 · PRACTICAL 7", "Generative Adversarial Network", "A generator turns 100 random numbers into a 784-pixel image; a discriminator tries to tell real from fake. Each is trained to beat the other — and the generator slowly learns to draw digits.")
    t1, t2, t3 = st.tabs(["Generate", "Watch it learn", "Losses"])
    with t1:
        c = st.columns(3)
        seed = c[0].number_input("Noise seed", 0, 99999, 7)
        n = c[1].select_slider("How many digits", [8, 16, 24, 40], value=24)
        scale = c[2].slider("Noise scale σ", 0.2, 1.5, 1.0, 0.05, help="Training used σ = 1. Smaller σ → safer, more average digits. Larger σ → wilder.")
        z = (np.random.default_rng(int(seed)).standard_normal((n, 100)) * scale).astype(np.float32)
        imgs = generate(z)
        pr = judge(imgs)
        st.image(grid(imgs, 8, scale=5), width="stretch")
        ncols = 8
        pred = pd.DataFrame(np.array([f"{p.argmax()} ({p.max() * 100:.0f}%)" for p in pr]).reshape(-1, ncols))
        st.markdown("**Judge's reading of each image** (same grid order, predicted digit and confidence)")
        st.dataframe(pred, hide_index=True, width="stretch")
        c = st.columns(2)
        c[0].metric("Mean judge confidence", f"{pr.max(1).mean():.3f}")
        c[1].metric("Most generated digit", int(np.bincount(pr.argmax(1), minlength=10).argmax()))
    with t2:
        snaps = W["gan_snaps"]
        ep = st.select_slider("Epoch", options=list(range(1, snaps.shape[0] + 1)), value=snaps.shape[0])
        st.image(grid(snaps[ep - 1].astype(np.float32) / 255, 10, scale=5), width="stretch")
        st.caption(f"The same 40 noise vectors after epoch {ep}. Epoch 1 is mostly blobs; by epoch 30 strokes and shapes appear.")
        st.markdown("**Progress montage** — first 10 samples at epochs 1, 5, 10, 20, 30")
        eps = [1, 5, 10, 20, 30]
        st.image(grid(np.concatenate([snaps[e - 1][:10].astype(np.float32) / 255 for e in eps]), 10, scale=5), width="stretch")
    with t3:
        d = pd.DataFrame({"epoch": np.tile(np.arange(1, len(W["gan_g"]) + 1), 2), "loss": np.concatenate([W["gan_g"], W["gan_d"]]),
                          "network": ["Generator"] * len(W["gan_g"]) + ["Discriminator"] * len(W["gan_d"])})
        st.altair_chart(line_chart(d, "epoch:Q", "loss:Q", "network:N", "Generator vs discriminator loss (BCE, Adam lr 2e-4, β1 0.5)", height=320), width="stretch")
        st.caption("Neither loss goes to zero — that's healthy. If the discriminator won outright, the generator would stop learning.")


def page_mixing():
    page_header("PART 6 · PRACTICAL 8", "Latent Mixing", "Practical 8 used StyleGAN2 style-mixing on faces. A plain fully-connected GAN has no separate style layers, so here the idea is shown on a smaller scale: splice two latent codes, or slide smoothly between two real digits.")
    t1, t2 = st.tabs(["GAN: splice two noise vectors", "Autoencoder: interpolate two digits"])
    with t1:
        c = st.columns(3)
        sa = c[0].number_input("Seed for source A", 0, 99999, 1)
        sb = c[1].number_input("Seed for source B", 0, 99999, 2)
        k = c[2].slider("Custom cutoff k", 0, 100, 40, help="First k noise values from A, the remaining 100−k from B")
        za = np.random.default_rng(int(sa)).standard_normal((8, 100)).astype(np.float32)
        zb = np.random.default_rng(int(sb)).standard_normal((8, 100)).astype(np.float32)
        mix = lambda cut: np.concatenate([za[:, :cut], zb[:, cut:]], 1)
        labels = ["Source A", "k = 25", "k = 50", "k = 75", "Source B", f"k = {k} (yours)"]
        rows = [generate(za), generate(mix(25)), generate(mix(50)), generate(mix(75)), generate(zb), generate(mix(k))]
        st.image(grid(np.concatenate(rows), 8, scale=6), width="stretch")
        st.caption("Rows: " + " · ".join(labels))
        pt = pd.DataFrame([[f"{judge(r_)[j].argmax()}" for j in range(8)] for r_ in rows], index=labels, columns=[f"#{i + 1}" for i in range(8)])
        st.markdown("**What the judge reads in each row**")
        st.dataframe(pt, width="stretch")
        st.caption("Larger k → more like A, smaller k → more like B, but the change isn't clean: some columns hold the shape of one source, others turn into a different digit. That's expected without style layers.")
    with t2:
        c = st.columns(2)
        with c[0]:
            st.markdown("##### Digit A")
            ia = digit_picker("mixA", default=3)
        with c[1]:
            st.markdown("##### Digit B")
            ib = digit_picker("mixB", default=8)
        alpha = st.slider("Blend  α  (0 = A, 1 = B)", 0.0, 1.0, 0.5, 0.01)
        za_, zb_ = encode(XTE[ia])[0], encode(XTE[ib])[0]
        cur = decode((1 - alpha) * za_ + alpha * zb_)[0]
        pc = judge(cur)[0]
        a, b, c2, d_ = st.columns([1, 1, 1, 2])
        with a: show_digit(XTE[ia], 130, f"A (label {YTE[ia]})")
        with b: show_digit(cur, 130, f"α = {alpha:.2f}")
        with c2: show_digit(XTE[ib], 130, f"B (label {YTE[ib]})")
        with d_:
            st.markdown("**Judge's probabilities for the blend**")
            st.altair_chart(bar_probs(pc, pc.argmax(), height=150), width="stretch")
        al = np.linspace(0, 1, 10)
        strip = np.concatenate([decode((1 - t) * za_ + t * zb_) for t in al])
        st.image(grid(strip, 10, scale=6), width="stretch")
        st.caption("Ten evenly spaced points on the straight line between the two latent codes, decoded one by one.")


def page_comparison():
    page_header("PART 7", "Final Comparison", "Every model is scored on the same 2,000 test images with the same four measures. The GAN has no original image to compare with, so only confidence and distribution distance apply to it.")
    names = list(W["metrics_names"])
    tab = pd.DataFrame(W["metrics"], index=names, columns=["MSE", "Digit accuracy", "Mean confidence", "Distribution distance (TV)"])
    ae_like = tab.drop(index="GAN (generated)")
    c = st.columns(3)
    c[0].metric("Lowest MSE", f"{ae_like['MSE'].min():.4f}", ae_like["MSE"].idxmin(), delta_color="off", delta_arrow="off")
    c[1].metric("Highest digit accuracy", f"{ae_like['Digit accuracy'].max() * 100:.1f}%", ae_like["Digit accuracy"].idxmax(), delta_color="off", delta_arrow="off")
    c[2].metric("Closest to real digit distribution (TV)", f"{tab['Distribution distance (TV)'].min():.4f}", tab["Distribution distance (TV)"].idxmin(), delta_color="off", delta_arrow="off")
    best = {"MSE": ae_like["MSE"].min(), "Distribution distance (TV)": tab["Distribution distance (TV)"].min(),
            "Digit accuracy": ae_like["Digit accuracy"].max(), "Mean confidence": tab["Mean confidence"].max()}
    disp = tab.apply(lambda col: col.map(lambda v: "n/a" if pd.isna(v) else f"{v:.4f}"))   # GAN has no MSE / accuracy
    green = pd.DataFrame("", index=tab.index, columns=tab.columns)
    for col, b in best.items():
        green.loc[tab[col] == b, col] = "background-color:#3a5a2a"
    st.dataframe(disp.style.apply(lambda _: green, axis=None), width="stretch")
    st.caption(f"Green = best in column. The helper classifier itself scores {float(W['clf_acc']) * 100:.2f}% on the 10,000 test images, so no model can be expected to beat that on accuracy.")

    st.subheader("Metric by metric")
    long = tab.reset_index(names="Model").melt("Model", var_name="Measure", value_name="Value").dropna()
    order = list(tab.index)
    cols = st.columns(2)
    for i, (measure, note) in enumerate((("MSE", "lower is better"), ("Digit accuracy", "higher is better"), ("Mean confidence", "higher is better"), ("Distribution distance (TV)", "lower is better"))):
        d = long[long.Measure == measure]
        cols[i % 2].altair_chart(
            alt.Chart(d).mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3, color=ACCENT)
            .encode(y=alt.Y("Model:N", sort=order, title=None), x=alt.X("Value:Q", title=None), tooltip=["Model", alt.Tooltip("Value:Q", format=".4f")])
            .properties(height=200, title=f"{measure} ({note})"),
            width="stretch")

    st.subheader("Digit distribution — who draws which digits?")
    dn = list(W["dist_names"])
    dv = W["dist_vals"]
    rows = [{"digit": k, "P": float(W["true_dist"][k]), "source": "Real test data"} for k in range(10)]
    rows += [{"digit": k, "P": float(dv[i][k]), "source": dn[i]} for i in range(len(dn)) for k in range(10)]
    dd = pd.DataFrame(rows)
    sel = st.multiselect("Models to show", dn, default=dn)
    dd = dd[dd.source.isin(["Real test data"] + sel)]
    st.altair_chart(
        alt.Chart(dd).mark_bar().encode(
            x=alt.X("digit:O", axis=alt.Axis(labelAngle=0)), xOffset="source:N", y=alt.Y("P:Q", title="Average probability"),
            color=alt.Color("source:N", scale=alt.Scale(range=["#8a8f9c"] + SERIES), legend=alt.Legend(orient="top", title=None)),
            tooltip=["digit:O", "source", alt.Tooltip("P:Q", format=".4f")]).properties(height=340),
        width="stretch")
    st.caption("Mean judge probability of each digit over the 2,000 images produced by each model. The closer a model's bars are to the real data, the more faithfully it keeps the digit mix.")


# ============================================================================= router
with st.sidebar:
    st.markdown("### 🔢 DeepDigit Studio")
    st.caption("Aryan Singh · Roll 69 · Batch C4\nDeep Learning II · 2026–27")
    st.caption("NumPy inference · weights from `train_numpy.py`")

# Single page: all sections stacked one after another
for _i, _section in enumerate((
    page_overview, page_playground, page_probability, page_autoencoder, page_markov,
    page_rbm, page_gan, page_mixing, page_comparison,
)):
    if _i:
        st.divider()
    _section()
