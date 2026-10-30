"""Helpers for calibrating simple detectors on the Sen1Floods11 training events
(CAL) only: stratified pixel sampling, classifier fitting, and F1-optimal
score thresholds. Test chips are never touched here."""
import numpy as np


def sample_pixels(gt, ok, n_per_class=1_000_000, seed=0):
    """Indices (flat) of up to n_per_class water and land pixels among ok."""
    rng = np.random.default_rng(seed)
    g = gt.ravel() & ok.ravel()
    l = ~gt.ravel() & ok.ravel()
    iw, il = np.flatnonzero(g), np.flatnonzero(l)
    iw = rng.choice(iw, min(n_per_class, iw.size), replace=False)
    il = rng.choice(il, min(n_per_class, il.size), replace=False)
    idx = np.concatenate([iw, il])
    rng.shuffle(idx)
    return idx


def best_threshold(score, gt, ok, higher_is_water=True):
    """F1-optimal cut on score over all ok pixels (exact, via sorting)."""
    s = score.ravel()[ok.ravel()].astype(np.float32)
    y = gt.ravel()[ok.ravel()]
    if not higher_is_water:
        s = -s
    order = np.argsort(-s, kind="stable")
    s, y = s[order], y[order]
    tp = np.cumsum(y, dtype=np.int64)
    k = np.arange(1, s.size + 1, dtype=np.int64)
    P = int(y.sum())
    f1 = 2 * tp / (k + P)
    # only evaluate cuts between distinct score values
    edge = np.flatnonzero(np.diff(s) != 0)
    if edge.size == 0:
        return float(s[0]), float(f1[-1])
    j = edge[np.argmax(f1[edge])]
    thr = 0.5 * (s[j] + s[j + 1])
    return (float(thr) if higher_is_water else float(-thr)), float(f1[j])


def fit_logreg(F, y, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
    m.fit(F, y)
    return m


def fit_gb(F, y, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
                                       early_stopping=True, validation_fraction=0.1,
                                       random_state=seed)
    m.fit(F, y)
    return m


def predict_proba(model, F, chunk=5_000_000):
    out = np.empty(F.shape[0], np.float32)
    for s in range(0, F.shape[0], chunk):
        out[s:s + chunk] = model.predict_proba(F[s:s + chunk])[:, 1]
    return out


def stack_features(cols, shape):
    """cols: list of (N,H,W) float arrays -> (N*H*W, k) float32 and finite mask."""
    F = np.stack([c.reshape(-1) for c in cols], axis=1).astype(np.float32)
    ok = np.isfinite(F).all(axis=1)
    return F, ok
