# FNO-CTCM – Fourier Neural Operator Continuous Time Consistency Model
*A one or two step generative model for CIFAR-10*

---

### Motivation
Continuous time consistency models (CTCMs) learn a single neural network $f_{\theta}(x_t, t)$ that is **consistent across noise scales**:

$$
\mathbb{E}_{x_0,\epsilon,t,\Delta t}
\left\|
f_{\theta}\bigl(x_0+\sigma(t)\,\epsilon, t\bigr)
- f_{\theta}\bigl(x_0+\sigma(t-\Delta t)\,\epsilon, t-\Delta t\bigr)
  \right\|_2^2 \longrightarrow 0
  $$

Sampling becomes **one step**: draw pure Gaussian noise $x_T$ and evaluate $x_0=f_{\theta}(x_T,T)$.

Here we **replace the usual U-Net** with a **Fourier Neural Operator (FNO)** from the
[`neuraloperator`](https://github.com/neuraloperator/neuraloperator) library, giving a
global receptive field in a few layers.

---

### Key features
| Component               | Technique                                                      |
|-------------------------|----------------------------------------------------------------|
| Backbone                | 6-layer **Weighted FNO-2D** with 2× spatial skip-convs         |
| Time conditioning       | **Low-freq sinusoidal** + learned MLP → FiLM modulation        |
| Training loss           | Consistency loss **+** JVP tangent loss (0.1 × weight)         |
| Regularisation          | Weight-decay, spectral weight-drop, Exponential-MA (EMA)       |
| Optimiser / schedule    | AdamW (lr 3e-4) + cosine decay                                 |
| Sampling modes          | `one_step` (FID target < 2.3) or `two_step` (FID ≈ 1.9)        |

---

### File tree
```
fno_ctcm/
├── configs/
│   └── default_cifar.yaml
├── data/
│   └── dataset.py
├── inference/
│   └── sampler.py
├── models/
│   ├── fno_generator.py
│   ├── discriminator.py
│   └── time_embedding.py
├── training/
│   └── train.py
├── utils/
│   ├── ema.py
│   ├── losses.py
│   ├── fid_score.py
│   ├── checkpoint.py
│   ├── visualize.py
│   └── config.py
├── requirements.txt
└── README.md
```

---

### Maths summary
*Noise schedule* (TrigFlow, Lu & Song 2024):

$$
\sigma(t)=\sigma_{\mathrm{min}}
\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^{t},
\quad t\in[0,1]
$$

*Consistency objective* (sample $\Delta t\sim\mathrm{U}[0,\delta_{\max}]$):

$$
\mathcal{L}_{\text{cons}}(\theta)=
\mathbb{E}\left[
\left\|
f_{\theta}(x_t,t)-f_{\theta}(x_{t'},t')
\right\|_2^2
\right]
$$

*Tangent (JVP) regulariser*:

$$
\mathcal{L}_{\text{tan}}(\theta)=
\lambda_{\text{tan}}
\mathbb{E}\left[
\left\|\partial_t f_{\theta}(x_t,t)-\frac{x_{t'}-x_t}{t'-t}\right\|_2^2
\right]
$$

Total loss: $\mathcal{L}=\mathcal{L}_{\text{cons}}+\mathcal{L}_{\text{tan}}$

---

### Quick start
```bash
conda create -n fno_ctcm python=3.11
pip install -r requirements.txt

# Train
python -m training.train --config configs/default_cifar.yaml

# Sample 50,000 images & compute FID
python -m inference.sampler --checkpoint ckpt/best_ema.pt --num 50000
python -m utils.fid_score --gen_dir samples --real_dir cifar_stats
```