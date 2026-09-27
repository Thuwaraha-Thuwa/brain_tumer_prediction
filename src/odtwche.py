"""ODTWCHE contrast enhancement (paper Sec. 3.1).

Otsu Double Threshold Weighted Constrained Histogram Equalization, 6 steps:
  1. Otsu double threshold   -> split the histogram into 3 parts
  2. weighted constraint     -> flatten very frequent grey levels (avoid over-enhancement)
  3. PSO                     -> find the best weights (goal: maximum entropy = most detail)
  4. histogram equalisation  -> of each part inside its own grey range
  5. adaptive gamma          -> improve global contrast
  6. Wiener filter           -> remove noise
The authors did not publish code; this is a re-implementation from the paper text.
"""
import cv2
import numpy as np
from scipy.signal import wiener
from tqdm.auto import tqdm

from . import config as C


def histogram(img):
    return np.bincount(img.ravel(), minlength=256).astype(float)


def entropy(hist):
    p = hist[hist > 0] / hist.sum()
    return float(-(p * np.log2(p)).sum())


# ---------------------------------------------------------------- step 1
def otsu_double_threshold(hist):
    """Try every pair t1 < t2 and keep the one with the largest between-class variance."""
    p = hist / hist.sum()
    P = np.cumsum(p)                       # probability of levels 0..k
    M = np.cumsum(p * np.arange(256))      # mean * probability of levels 0..k
    t1, t2 = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    w = [P[t1], P[t2] - P[t1], 1 - P[t2]]
    m = [M[t1], M[t2] - M[t1], M[-1] - M[t2]]
    with np.errstate(divide="ignore", invalid="ignore"):
        score = sum(mi ** 2 / wi for mi, wi in zip(m, w))
    score[(t1 >= t2) | (w[0] <= 0) | (w[1] <= 0) | (w[2] <= 0)] = -np.inf
    i, j = np.unravel_index(np.argmax(score), score.shape)
    return int(i), int(j)


# ---------------------------------------------------------------- step 2
def weighted_constraint(p, alpha):
    """p_w = p_max * ((p - p_min) / (p_max - p_min)) ** alpha   (alpha in 0..1)"""
    nz = p > 0
    if nz.sum() < 2 or p[nz].max() == p[nz].min():
        return p.copy()
    pmax, pmin = p[nz].max(), p[nz].min()
    pw = np.zeros_like(p)
    pw[nz] = np.maximum(pmax * ((p[nz] - pmin) / (pmax - pmin)) ** alpha, pmin)
    return pw


# ---------------------------------------------------------------- step 4
def equalize_parts(hist, t1, t2, alphas):
    """Lookup table: equalise each of the 3 parts inside its own range."""
    lut = np.arange(256, dtype=float)
    for (lo, hi), a in zip([(0, t1), (t1 + 1, t2), (t2 + 1, 255)], alphas):
        h = hist[lo:hi + 1]
        if hi > lo and h.sum() > 0:
            pw = weighted_constraint(h / h.sum(), a)
            lut[lo:hi + 1] = lo + (hi - lo) * np.cumsum(pw) / pw.sum()
    return np.clip(np.round(lut), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------- step 3
def pso(fitness, dim=3, particles=12, iterations=15, seed=0):
    """Particle Swarm Optimisation: particles move towards the best solutions found."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.1, 1.0, (particles, dim))        # positions = candidate alphas
    v = np.zeros_like(x)                                # velocities
    best_x, best_f = x.copy(), np.array([fitness(xi) for xi in x])
    g = best_x[best_f.argmax()]
    for _ in range(iterations):
        r1, r2 = rng.random((2, particles, dim))
        v = 0.7 * v + 1.5 * r1 * (best_x - x) + 1.5 * r2 * (g - x)
        x = np.clip(x + v, 0.1, 1.0)
        f = np.array([fitness(xi) for xi in x])
        better = f > best_f
        best_x[better], best_f[better] = x[better], f[better]
        g = best_x[best_f.argmax()]
    return g


# ---------------------------------------------------------------- step 5
def adaptive_gamma(img, alpha=0.5):
    """AGCWD: dark levels get a small gamma (brightened), bright levels stay."""
    pw = weighted_constraint(histogram(img) / img.size, alpha)
    gamma = 1 - np.cumsum(pw) / pw.sum()
    lut = np.round(255 * (np.arange(256) / 255) ** gamma).astype(np.uint8)
    return lut[img]


# ---------------------------------------------------------------- step 6
def wiener_filter(img, window=3):
    with np.errstate(divide="ignore", invalid="ignore"):   # flat black background
        out = wiener(img.astype(float), (window, window))
    return np.clip(np.nan_to_num(out), 0, 255).round().astype(np.uint8)


# ---------------------------------------------------------------- full method
def odtwche(img, return_steps=False):
    """img: 2-D uint8 grayscale MRI slice."""
    hist = histogram(img)
    t1, t2 = otsu_double_threshold(hist)                                  # 1

    def fitness(alphas):                                                  # entropy of result
        lut = equalize_parts(hist, t1, t2, alphas)
        return entropy(np.bincount(lut, weights=hist, minlength=256))

    alphas = pso(fitness)                                                 # 2 + 3
    equalized = equalize_parts(hist, t1, t2, alphas)[img]                 # 4
    gamma = adaptive_gamma(equalized)                                     # 5
    result = wiener_filter(gamma)                                         # 6
    if return_steps:
        return result, {"thresholds": (t1, t2), "alphas": alphas,
                        "equalized": equalized, "gamma": gamma}
    return result


def psnr(original, enhanced):
    mse = np.mean((original.astype(float) - enhanced.astype(float)) ** 2)
    return 10 * np.log10(255 ** 2 / mse) if mse > 0 else float("inf")


def enhance_dataset(files):
    """Enhance the given PNG files: data/images -> data/enhanced."""
    for f in tqdm(files):
        img = cv2.imread(str(C.IMAGES_DIR / f), cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(str(C.ENHANCED_DIR / f), odtwche(img))
