#!/usr/bin/env python3
"""
Film Dust Blower - detection and repair engine for the Lightroom Classic plugin.

Finds dust specks and hairs on film scans and repairs them by copying nearby
image texture (grain included) with a smooth tone correction, the way a
healing brush does.

Usage:
    python3 dust_engine.py --check
    python3 dust_engine.py --job JOB.txt --out RESULT.txt [options]

Job file, one line per image, tab separated:
    <id> <TAB> <input image> <TAB> <output TIFF or -> <TAB> <preview JPEG or ->

Result file, one line per image, tab separated:
    <id> <TAB> ok <TAB> <specks> <TAB> <hairs> <TAB> <output or ->
    <id> <TAB> fail <TAB> <reason>

Requires: numpy, opencv-python, tifffile
    python3 -m pip install numpy opencv-python tifffile
"""

import argparse
import math
import os
import sys
import traceback

_MISSING = []   # reported by --check
try:
    import numpy as np
except ImportError:
    np = None
    _MISSING.append("numpy")
try:
    import cv2
except ImportError:
    cv2 = None
    _MISSING.append("opencv-python")
try:
    import tifffile
except ImportError:
    tifffile = None
    _MISSING.append("tifffile")

ICC_TAG = 34675


# --------------------------------------------------------------------------
# Input / output
# --------------------------------------------------------------------------

def load_image(path):
    """Return (float32 RGB image in 0..1, original dtype, ICC profile bytes or None, alpha or None)."""
    ext = os.path.splitext(path)[1].lower()
    icc = None
    if ext in (".tif", ".tiff"):
        if tifffile is None:
            raise RuntimeError("tifffile is not installed")
        with tifffile.TiffFile(path) as tf:
            page = tf.pages[0]
            arr = page.asarray()
            if ICC_TAG in page.tags:
                icc = bytes(page.tags[ICC_TAG].value)
        if arr.ndim == 3 and arr.shape[0] in (3, 4) and arr.shape[2] not in (3, 4):
            arr = np.moveaxis(arr, 0, -1)
    else:
        data = np.fromfile(path, dtype=np.uint8)
        arr = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
        if arr is None:
            raise ValueError("could not read image")
        if arr.ndim == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA if arr.shape[2] == 4 else cv2.COLOR_BGR2RGB)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=2)
    alpha = None
    if arr.shape[2] == 4:
        alpha = arr[..., 3].copy()
        arr = arr[..., :3]
    dtype = arr.dtype
    scale = float(np.iinfo(dtype).max) if np.issubdtype(dtype, np.integer) else 1.0
    img = arr.astype(np.float32) / scale
    return img, dtype, icc, alpha


def save_tiff(path, img, dtype, icc, alpha):
    if np.issubdtype(dtype, np.integer):
        mx = np.iinfo(dtype).max
        out = np.clip(np.rint(img * mx), 0, mx).astype(dtype)
    else:
        out = img.astype(dtype)
    if alpha is not None:
        out = np.dstack([out, alpha.astype(out.dtype)])
    extratags = [(ICC_TAG, 7, len(icc), icc, True)] if icc else []
    tmp = path + ".part"
    tifffile.imwrite(tmp, out, photometric="rgb", compression="zlib", extratags=extratags,
                     extrasamples=[2] if alpha is not None else None)
    os.replace(tmp, path)


def luminance(img):
    return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(np.float32)


# --------------------------------------------------------------------------
# Detection
#
# Dust blocks light, so on a scan of a negative (or a slide) it is a small,
# sharp, dark mark. After a negative is converted to a positive it becomes a
# bright mark; that case is handled by inverting the lightness first.
#
# For each of several scales:
#   black-hat = morphological closing - image   (lights up small dark marks)
#   z         = black-hat measured against its local typical level
# Marks are kept when they stand out strongly (z), are visible (absolute
# contrast), are isolated from similar marks around them (so foliage, text and
# texture are left alone), and are the right size and shape.
# --------------------------------------------------------------------------

SE = 11                    # closing window at each scale (pixels at that scale)


class Spot:
    __slots__ = ("x0", "y0", "x1", "y1", "mask", "kind", "cx", "cy", "size", "thick", "score", "feats")

    def __init__(self, x0, y0, mask, kind, score, thick):
        self.x0, self.y0 = int(x0), int(y0)
        self.y1, self.x1 = self.y0 + mask.shape[0], self.x0 + mask.shape[1]
        self.mask = mask
        self.kind = kind
        ys, xs = np.nonzero(mask)
        self.cx = float(xs.mean()) + self.x0 if len(xs) else self.x0
        self.cy = float(ys.mean()) + self.y0 if len(ys) else self.y0
        self.size = max(mask.shape)
        self.thick = thick
        self.score = score
        self.feats = None


def _grain_noise(Ys, block=32):
    """
    Local grain level: mean absolute high-pass in blocks, median-filtered so
    that dust and sharp picture edges do not inflate it.
    """
    h, w = Ys.shape
    hp = np.abs(Ys - cv2.GaussianBlur(Ys, (0, 0), 2.0))
    sh, sw = max(1, h // block), max(1, w // block)
    sm = cv2.resize(hp, (sw, sh), interpolation=cv2.INTER_AREA).astype(np.float32)
    if min(sm.shape) >= 3:
        sm = cv2.medianBlur(sm, 3)
    noise = cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR) * 1.25
    return np.maximum(noise, 0.002)


def _refine_full_res(Y, x0, y0, m, d_full):
    """
    Redraw a speck found at a coarse scale on the full-resolution image: pixels
    clearly darker than the surrounding background, near the coarse outline.
    Returns (mask in the same frame as m, equivalent diameter in full-res px).
    """
    H, W = Y.shape
    pad = int(max(3, 0.5 * d_full))
    X0, Y0 = max(0, x0 - pad), max(0, y0 - pad)
    X1, Y1 = min(W, x0 + m.shape[1] + pad), min(H, y0 + m.shape[0] + pad)
    big = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    big[y0 - Y0:y0 - Y0 + m.shape[0], x0 - X0:x0 - X0 + m.shape[1]] = m
    near = cv2.dilate(big, _disk(max(2, pad // 2))) > 0
    ring = (cv2.dilate(big, _disk(pad)) > 0) & ~near
    roi = cv2.GaussianBlur(Y[Y0:Y1, X0:X1], (0, 0), 0.8)
    if ring.sum() < 10 or (big > 0).sum() < 4:
        return m, d_full
    bg = float(np.median(roi[ring]))
    mn = float(np.percentile(roi[big > 0], 2))
    if bg - mn <= 1e-4:
        return m, d_full
    core = ((roi < bg - 0.3 * (bg - mn)) & near).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n <= 1:
        return m, d_full
    # keep the parts that overlap the coarse outline
    ids = np.unique(lab[(big > 0) & (core > 0)])
    ids = ids[ids > 0]
    if len(ids) == 0:
        return m, d_full
    fine = np.isin(lab, ids).astype(np.uint8)
    out = fine[y0 - Y0:y0 - Y0 + m.shape[0], x0 - X0:x0 - X0 + m.shape[1]]
    if out.sum() < 3:
        return m, d_full
    return out, 2.0 * math.sqrt(out.sum() / math.pi)


def _disk(r):
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def detect(Y, sensitivity=1.0, max_size_frac=0.015, hairs=True, min_contrast=0.02, debug=None):
    """
    Y: float32 lightness, dust = dark. Returns a list of Spot (full-resolution masks).
    max_size_frac: largest speck diameter as a fraction of the short side.
    debug: optional list; rejected candidates are appended as (scale, x, y, reason).
    """
    H, W = Y.shape
    short = min(H, W)
    max_d = max(6.0, max_size_frac * short)
    sens = max(0.2, float(sensitivity))
    z_seed = 14.0 / math.sqrt(sens)  # contrast against grain needed to start a mark
    z_grow = 4.0                  # contrast needed to belong to it
    iso_ratio = 2.2 / math.sqrt(sens)
    score_min = 4.0 - 1.5 * math.log(sens)   # sensitivity 2 -> 2.96, 0.5 -> 5.04

    def reject(s, x, y, why):
        if debug is not None:
            debug.append((s, x * s, y * s, why))

    scales = [1]
    while (SE - 2) * scales[-1] < max_d and scales[-1] < 32:
        scales.append(scales[-1] * 2)

    found = []
    for s in scales:
        Ys = Y if s == 1 else cv2.resize(Y, (max(1, W // s), max(1, H // s)), interpolation=cv2.INTER_AREA)
        Ys = cv2.GaussianBlur(Ys, (0, 0), 0.7)
        closed = cv2.morphologyEx(Ys, cv2.MORPH_CLOSE, np.ones((SE, SE), np.uint8))
        bh = closed - Ys
        z = bh / _grain_noise(Ys)
        contrast = bh / np.maximum(closed, 0.02)       # fraction of light blocked
        seed = (z > z_seed) & (bh > min_contrast * np.maximum(closed, 0.05))
        if not seed.any():
            continue
        grow = (z > z_grow).astype(np.uint8)
        n, labels, stats, cents = cv2.connectedComponentsWithStats(grow, connectivity=8)
        ids = np.unique(labels[seed])
        ids = ids[ids > 0]
        hs, ws = Ys.shape
        for i in ids:
            x, y, bw, bhh, area = stats[i]
            if area < 2:
                continue
            # touching the image edge: probably the film edge or holder, skip
            if x <= 1 or y <= 1 or x + bw >= ws - 1 or y + bhh >= hs - 1:
                reject(s, x, y, "edge")
                continue
            comp = labels[y:y + bhh, x:x + bw] == i
            # trim the region to pixels at least 35 % as strong as its peak, so a
            # strong speck does not swell into the grain around it
            sub = bh[y:y + bhh, x:x + bw]
            pk_val = float(sub[comp].max())
            core_m = (comp & (sub >= 0.35 * pk_val)).astype(np.uint8)
            n2, lab2 = cv2.connectedComponents(core_m, connectivity=8)
            py, px = np.unravel_index(int(np.argmax(np.where(comp, sub, -1.0))), comp.shape)
            comp = lab2 == lab2[py, px]
            ys_, xs_ = np.nonzero(comp)
            oy, ox = int(ys_.min()), int(xs_.min())
            comp = comp[oy:int(ys_.max()) + 1, ox:int(xs_.max()) + 1]
            x, y = x + ox, y + oy
            bhh, bw = comp.shape
            area = int(comp.sum())
            d_eq = 2.0 * math.sqrt(area / math.pi)
            length = float(max(bw, bhh, 1))
            # shape: elongation from the minimum-area rectangle
            cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if not cnts:
                continue
            (_, _), (rw, rh), _ = cv2.minAreaRect(max(cnts, key=len))
            rl, rs = max(rw, rh, 1.0), max(min(rw, rh), 1.0)
            thick = max(1.0, area / rl)
            elong = rl / thick
            is_hair = elong > 4.0 and rl * s > 3 * SE
            if is_hair and s > 2:
                reject(s, x, y, "hair-coarse-scale")
                continue
            if is_hair:
                if not hairs or thick > SE - 2:
                    reject(s, x, y, "hair-off-or-thick")
                    continue
                if rl * s > 0.3 * short:
                    reject(s, x, y, "hair-too-long")
                    continue
            else:
                # each scale handles its own size band; larger marks belong to coarser scales
                # bands overlap a little; duplicates are merged afterwards
                if d_eq > SE - 2 and s != scales[-1]:
                    reject(s, x, y, "big-for-scale %.1f" % d_eq)
                    continue
                if s > 1 and d_eq < (SE - 2) / 2.0 - 1.5:
                    reject(s, x, y, "small-for-scale %.1f" % d_eq)
                    continue
                if d_eq * s > max_d:
                    reject(s, x, y, "too-big")
                    continue
            peak = float(bh[y:y + bhh, x:x + bw][comp].max())
            zpk = float(z[y:y + bhh, x:x + bw][comp].max())
            cpk = float(contrast[y:y + bhh, x:x + bw][comp].max())
            if is_hair and zpk < 1.3 * z_seed:
                reject(s, x, y, "hair-weak %.1f" % zpk)
                continue
            # isolation: compare with the strongest similar marks around it
            r_out = int(max(4, 2.5 * (thick if is_hair else d_eq)))
            pad = r_out + 1
            X0, Y0 = max(0, x - pad), max(0, y - pad)
            X1, Y1 = min(ws, x + bw + pad), min(hs, y + bhh + pad)
            big = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
            big[y - Y0:y - Y0 + bhh, x - X0:x - X0 + bw] = comp
            outer = cv2.dilate(big, _disk(r_out))
            inner = cv2.dilate(big, _disk(2))
            ring = (outer > 0) & (inner == 0)
            if ring.sum() < 8:
                reject(s, x, y, "no-ring")
                continue
            ring_level = float(np.percentile(bh[Y0:Y1, X0:X1][ring], 95))
            if peak < iso_ratio * ring_level:
                reject(s, x, y, "not-isolated %.2f" % (peak / max(ring_level, 1e-6)))
                continue
            # all-round test: dust is surrounded by lighter picture on every side,
            # whereas the inside corner of a dark shape or a gap between leaves is
            # open to darkness on one side
            Yroi = Ys[Y0:Y1, X0:X1]
            core = float(np.percentile(Yroi[big > 0], 10))
            ry, rx = np.nonzero(ring)
            ccy, ccx = (y - Y0) + bhh / 2.0, (x - X0) + bw / 2.0
            sector = ((np.arctan2(ry - ccy, rx - ccx) + math.pi) / (2 * math.pi) * 8).astype(int) % 8
            vals = Yroi[ry, rx]
            lighter = 0
            used = 0
            for k in range(8):
                v = vals[sector == k]
                if len(v) < 2:
                    continue
                used += 1
                if np.median(v) - core >= 0.45 * peak:
                    lighter += 1
            need = used - (2 if is_hair else 1)       # a hair leaves through two sectors
            if used < 6 or lighter < need:
                reject(s, x, y, "open-side %d/%d" % (lighter, used))
                continue
            # combined evidence: contrast against grain, isolation from similar
            # marks nearby, and the share of light blocked
            iso = peak / max(ring_level, 1e-6)
            score = math.log(max(zpk, 1e-3)) + 1.2 * math.log(max(iso, 1e-3)) + 0.8 * math.log(max(cpk, 1e-3))
            if score < score_min:
                reject(s, x, y, "weak-score %.2f" % score)
                continue
            feats = dict(z=zpk, c=cpk, iso=peak / max(ring_level, 1e-6), light=lighter, used=used,
                         d=(thick if is_hair else d_eq) * s, hair=is_hair, s=s)
            # full-resolution mask with a halo for the soft edge of the dust
            m = comp.astype(np.uint8)
            if s > 1:
                m = cv2.resize(m, (bw * s, bhh * s), interpolation=cv2.INTER_NEAREST)
                if is_hair:
                    m, _ = _refine_full_res(Y, x * s, y * s, m, thick * s)
                    if m.any():
                        dtm = cv2.distanceTransform(cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0),
                                                    cv2.DIST_L2, 3)
                        thick = max(1.0, 2.0 * float(dtm.max()) / s)
                else:
                    m, d_eq = _refine_full_res(Y, x * s, y * s, m, d_eq * s)
                    d_eq = d_eq / s
            halo = int(round(1.5 + 0.15 * (thick if is_hair else d_eq) * s))
            m = cv2.copyMakeBorder(m, halo, halo, halo, halo, cv2.BORDER_CONSTANT, value=0)
            m = cv2.dilate(m, _disk(halo)) > 0
            spot = Spot(x * s - halo, y * s - halo, m, "hair" if is_hair else "speck", score,
                        (thick if is_hair else d_eq) * s)
            spot.feats = feats
            found.append(spot)

    # the same speck can be found at two scales: keep the larger detection. A much
    # smaller mark inside a large soft one (a crisp speck on a pale patch) is kept
    # as well, so that it is judged on its own
    found.sort(key=lambda sp: -sp.mask.sum())
    kept = []
    for sp in found:
        dup = False
        for k in kept:
            if (abs(sp.cx - k.cx) < max(k.size, sp.size) / 2.0 and abs(sp.cy - k.cy) < max(k.size, sp.size) / 2.0):
                if sp.size < 0.35 * k.size:
                    continue
                dup = True
                break
        if not dup:
            kept.append(sp)
    # clip masks to the image
    out = []
    for sp in kept:
        x0, y0 = max(0, sp.x0), max(0, sp.y0)
        x1, y1 = min(W, sp.x1), min(H, sp.y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        m = sp.mask[y0 - sp.y0:y1 - sp.y0, x0 - sp.x0:x1 - sp.x0]
        if m.any():
            ns_ = Spot(x0, y0, m, sp.kind, sp.score, sp.thick)
            ns_.feats = sp.feats
            out.append(ns_)
    return out


# --------------------------------------------------------------------------
# Learned dust classifier
#
# The detector above proposes candidates generously (both light and dark
# marks). A small gradient-boosted tree model, trained on hand-cleaned film
# scans, then decides which candidates are dust. Its strongest cues are the
# structure around a mark (edges, branches, silhouettes mean picture), its
# size, contrast against grain and isolation from similar marks.
# The model lives in dust_model.json next to this file.
# --------------------------------------------------------------------------

_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        import json
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dust_model.json")
        with open(path, "r", encoding="utf-8") as f:
            m = json.load(f)
        trees = []
        for t in m["trees"]:
            trees.append((np.array(t["f"], int), np.array(t["t"], float), np.array(t["l"], int),
                          np.array(t["r"], int), np.array(t["leaf"], bool), np.array(t["v"], float),
                          np.array(t["m"], bool)))
        _MODEL = (float(m["base"]), trees, float(m["threshold"]),
                  float(m.get("band_threshold", m["threshold"])), len(m.get("features", [])) or 20,
                  float(m.get("prefilter_z", 9.0)), float(m.get("prefilter_iso", 2.0)))
        RULES.update({k: float(m[k]) for k in RULES if k in m})
    return _MODEL


# rules applied after the model (their values come with the model file)
RULES = {"vis_min": 0.0, "busy_struct": 99.0, "busy_margin": 0.0,
         "sure_stand": 99.0, "sure_con": 99.0, "sure_struct": 0.0, "sure_size": 0.0}


def model_score(X):
    base, trees = _load_model()[:2]
    out = np.full(len(X), base)
    for f, t, l, r, leaf, v, m in trees:
        node = np.zeros(len(X), int)
        for _ in range(64):
            act = ~leaf[node]
            if not act.any():
                break
            idx = np.where(act)[0]
            nn = node[idx]
            xv = X[idx, f[nn]]
            go_left = np.where(np.isnan(xv), m[nn], xv <= t[nn])
            node[idx] = np.where(go_left, l[nn], r[nn])
        out += v[node]
    return out


def _gray601(img):
    return (0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]).astype(np.float32)


def _block_percentile(a, pct, q, bs=64):
    h, w = a.shape
    if h < bs or w < bs:
        return float(np.percentile(np.percentile(a, pct), q))
    blk = a[:h // bs * bs, :w // bs * bs].reshape(h // bs, bs, w // bs, bs).transpose(0, 2, 1, 3).reshape(-1, bs * bs)
    return float(np.percentile(np.percentile(blk, pct, axis=1), q))


def _frame_stats(img, Y):
    g = _gray601(img)
    hp = np.abs(g - cv2.GaussianBlur(g, (0, 0), 3))
    noise = max(_block_percentile(hp, 90, 10), 1e-4)            # grain level in the flattest areas
    lp = cv2.GaussianBlur(g, (0, 0), 2.5)
    gm = np.hypot(cv2.Sobel(lp, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(lp, cv2.CV_32F, 0, 1, ksize=3)) / 8
    gnoise = max(_block_percentile(gm, 99, 10), 1e-5)            # edge level in the flattest areas
    mn = img.min(axis=2)
    # finest scale: dust lies on the film and is as sharp as the grain, usually
    # sharper than anything the lens drew
    hp1 = Y - cv2.GaussianBlur(Y, (0, 0), 1.0)
    hp3 = Y - cv2.GaussianBlur(Y, (0, 0), 3.0)
    fnoise = max(_block_percentile(np.abs(hp1), 50, 10) * 1.4826, 1e-5)
    return dict(noise=noise, gm=gm, gnoise=gnoise, Y=Y, hp1=hp1, hp3=hp3, fnoise=fnoise,
                lo=float(np.percentile(Y, 0.5)), hi=float(np.percentile(Y, 99.5)),
                hpY=Y - cv2.GaussianBlur(Y, (0, 0), 3),
                mn=mn, lum=img.mean(axis=2),
                lo_mn=float(np.percentile(mn, 1)), hi_mn=float(np.percentile(mn, 99.7)),
                colour=1.0 if float(np.abs(img[::7, ::7, 0] - img[::7, ::7, 1]).mean()) > 0.002 else 0.0)


def _cand_features(img, work, sp, sign, fr):
    """Properties of one candidate (identical to those the model was trained on)."""
    H, W = work.shape
    m = sp.mask
    y0, y1, x0, x1 = sp.y0, sp.y1, sp.x0, sp.x1
    size = max(m.shape)
    pad = int(max(12, 2.5 * size))
    Y0, Y1, X0, X1 = max(0, y0 - pad), min(H, y1 + pad), max(0, x0 - pad), min(W, x1 + pad)
    full = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    full[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = m
    inner = cv2.dilate(full, np.ones((5, 5), np.uint8)) > 0
    ring = (cv2.dilate(full, np.ones((2 * pad // 2 + 1,) * 2, np.uint8)) > 0) & ~inner
    if ring.sum() < 10:
        return None
    roi = img[Y0:Y1, X0:X1]
    core = full > 0
    bg = np.median(roi[ring], axis=0)
    wv = work[Y0:Y1, X0:X1]
    sel = core & (wv <= np.percentile(wv[core], 30))
    if sel.sum() == 0:
        sel = core
    pk = roi[sel].mean(axis=0)
    diff = (pk - bg) * sign
    g = _gray601(roi)
    hp = g - cv2.GaussianBlur(g, (0, 0), 3)
    busy = float(np.percentile(np.abs(hp[ring]), 90))
    ringstd = float(np.std(g[ring]))
    sat_pk = float((np.max(pk) - np.min(pk)) / (np.max(pk) + 1e-4))
    lum_pk = float(pk.mean())
    lo, hi = fr["lo"], fr["hi"]
    rel = (lum_pk - lo) / max(hi - lo, 1e-4) if sign > 0 else (hi - lum_pk) / max(hi - lo, 1e-4)
    cy, cx = int(sp.cy), int(sp.cx)
    R = 48
    a0, a1, b0, b1 = max(0, cy - R), min(H, cy + R), max(0, cx - R), min(W, cx + R)
    wide = fr["hpY"][a0:a1, b0:b1]
    excl = np.ones(wide.shape, bool)
    excl[max(0, y0 - a0 - 4):max(0, y1 - a0 + 4), max(0, x0 - b0 - 4):max(0, x1 - b0 + 4)] = False
    wide_busy = float(np.percentile(np.abs(wide[excl]), 90)) if excl.sum() > 20 else busy
    # structure nearby
    Rs = int(max(40, 4 * size))
    a0, a1, b0, b1 = max(0, cy - Rs), min(H, cy + Rs), max(0, cx - Rs), min(W, cx + Rs)
    win = fr["gm"][a0:a1, b0:b1].copy()
    e = int(size / 2 + 8)
    win[max(0, cy - e - a0):max(0, cy + e - a0), max(0, cx - e - b0):max(0, cx + e - b0)] = 0
    struct = float(np.percentile(win, 99)) / fr["gnoise"]
    struct95 = float(np.percentile(win, 95)) / fr["gnoise"]
    # colour at the very peak
    r2 = int(max(2, size / 3))
    w2 = (slice(max(0, cy - r2), cy + r2 + 1), slice(max(0, cx - r2), cx + r2 + 1))
    R2 = int(max(10, size * 1.5))
    rg = (slice(max(0, cy - R2), cy + R2 + 1), slice(max(0, cx - R2), cx + R2 + 1))
    lw = fr["lum"][w2]
    j = np.unravel_index(np.argmin(lw) if sign < 0 else np.argmax(lw), lw.shape)
    pkc = img[w2][j]
    bgm = float(np.median(fr["mn"][rg]))
    minch_rel = (float(pkc.min()) - fr["lo_mn"]) / max(fr["hi_mn"] - fr["lo_mn"], 1e-4)
    sat_peak = float((pkc.max() - pkc.min()) / (pkc.max() + 1e-4))
    minch_rise = float(pkc.min()) - bgm
    return dict(z=sp.feats["z"], iso=sp.feats["iso"], hair=float(sp.feats["hair"]), size=float(size),
                wide_busy=wide_busy, busy=busy, ringstd=ringstd, rise=float(diff.mean()), sat_pk=sat_pk,
                rel_white=float(rel), struct=struct, struct95=struct95, minch_rel=minch_rel,
                sat_peak=sat_peak, minch_rise=minch_rise, dark=1.0 if sign < 0 else 0.0)


def _crisp_features(fr, sp, sign):
    """
    How sharp a mark is, and whether it is sharper than the picture around it.
    Dust lies on the film and is as sharp as the grain; picture detail is
    softened by the lens. The crispness measure follows film dust (Magnus
    Holmes, MIT licence): the share of a mark's contrast that survives a fine
    high-pass, also against a perfectly sharp mark of the same shape. The
    comparison with the surroundings is our own: the mark's finest-scale peak
    against the finest-scale detail around it.
    """
    Y = fr["Y"]
    H, W = Y.shape
    m = sp.mask
    size = max(m.shape)
    R = int(max(40, 4 * size))
    cx, cy = int(sp.cx), int(sp.cy)
    X0, X1 = max(0, min(sp.x0, cx - R)), min(W, max(sp.x1, cx + R))
    Y0, Y1 = max(0, min(sp.y0, cy - R)), min(H, max(sp.y1, cy + R))
    full = np.zeros((Y1 - Y0, X1 - X0), np.uint8)
    full[sp.y0 - Y0:sp.y1 - Y0, sp.x0 - X0:sp.x1 - X0] = m
    obj = full > 0
    if obj.sum() < 2:
        return None
    yy = sign * Y[Y0:Y1, X0:X1]
    near = cv2.dilate(full, np.ones((7, 7), np.uint8)) > 0
    ring = (cv2.dilate(full, np.ones((21, 21), np.uint8)) > 0) & ~near
    if ring.sum() < 10:
        return None
    con = max(float(np.percentile(yy[obj], 85)) - float(np.median(yy[ring])), 1e-4)
    crisp = float(np.percentile(np.abs(fr["hp3"][Y0:Y1, X0:X1][obj]), 95)) / con
    of = obj.astype(np.float32)
    ideal = float(np.percentile(np.abs(of - cv2.GaussianBlur(of, (0, 0), 3.0))[obj], 95))
    h1 = fr["hp1"][Y0:Y1, X0:X1]
    fine_pk = max(float((sign * h1)[obj].max()), 1e-6)
    env = ~(cv2.dilate(full, np.ones((9, 9), np.uint8)) > 0)
    ev = np.abs(h1[env])
    env99 = float(np.percentile(ev, 99.5)) if ev.size > 50 else fine_pk
    gv = fr["gm"][Y0:Y1, X0:X1][env]
    g99 = float(np.percentile(gv, 99)) if gv.size > 50 else 1e-5
    # the same comparison at the mark's own size (a band-pass matched to its
    # diameter), so that heavy grain or other dust nearby doesn't hide it
    sb = min(STAND_SIGMAS, key=lambda s: abs(s - max(0.8, size / 5.0)))
    pad = int(6 * sb) + 2
    P0, P1, Q0, Q1 = max(0, Y0 - pad), min(H, Y1 + pad), max(0, X0 - pad), min(W, X1 + pad)
    Yp = Y[P0:P1, Q0:Q1]
    band = (cv2.GaussianBlur(Yp, (0, 0), sb) - cv2.GaussianBlur(Yp, (0, 0), 2 * sb))[Y0 - P0:Y1 - P0, X0 - Q0:X1 - Q0]
    band_pk = max(float((sign * band)[obj].max()), 1e-6)
    occ = fr.get("occ")
    free = env if occ is None else (env & (occ[Y0:Y1, X0:X1] == 0))
    bv = np.abs(band[free])
    band99 = float(np.percentile(bv, 99.5)) if bv.size > 50 else band_pk
    return dict(crisp=crisp, crisp_rel=crisp / max(ideal, 1e-3), fine_z=fine_pk / fr["fnoise"],
                stand=fine_pk / max(env99, 1e-6), env_fine=env99 / fr["fnoise"],
                env_soft=env99 / max(g99, 1e-6), con_z=con / fr["noise"],
                stand_m=band_pk / max(band99, 1e-6))


STAND_SIGMAS = (0.8, 1.2, 1.8, 2.7, 4.0, 6.0)
CRISP_KEYS = ("crisp", "crisp_rel", "fine_z", "stand", "env_fine", "env_soft", "con_z", "stand_m")


def _occupancy(shape, spots, grow=3):
    """Where candidate marks are (grown a little), to leave them out of each other's surroundings."""
    occ = np.zeros(shape, np.uint8)
    k = np.ones((2 * grow + 1, 2 * grow + 1), np.uint8)
    H, W = shape
    for sp in spots:
        m = cv2.dilate(cv2.copyMakeBorder(sp.mask.astype(np.uint8), grow, grow, grow, grow,
                                          cv2.BORDER_CONSTANT, value=0), k)
        y0, x0 = sp.y0 - grow, sp.x0 - grow
        a0, b0 = max(0, y0), max(0, x0)
        a1, b1 = min(H, y0 + m.shape[0]), min(W, x0 + m.shape[1])
        if a1 > a0 and b1 > b0:
            occ[a0:a1, b0:b1] |= m[a0 - y0:a1 - y0, b0 - x0:b1 - x0]
    return occ


def _design_row(f, fr, crowd):
    eps = 1e-4
    col = fr["colour"]
    lz = math.log(f["z"] + eps)
    st = math.log(f["struct"] + eps)
    flat = math.log(f["wide_busy"] / fr["noise"] + eps)
    return [lz, math.log(f["iso"] + eps), math.log(f["size"] + 1), flat, st, math.log(f["struct95"] + eps),
            math.log(f["busy"] / fr["noise"] + eps), math.log1p(crowd),
            math.log(abs(f["rise"]) + eps) - math.log(f["ringstd"] + eps),
            f["dark"], f["hair"], f["sat_pk"] * col, col * st, col * flat, f["dark"] * lz,
            f["minch_rel"], f["sat_peak"], f["minch_rise"], f["rel_white"], col]


def find_dust_learned(img, sensitivity=1.0, max_size_frac=0.015, hairs=True, max_spots=400, fr=None, scored=None):
    """
    img: float RGB in 0..1, dust expected to show light (a positive made from a
    negative). Returns accepted spots, strongest first. If scored is a list, the
    centre and score of every candidate measured are appended to it.
    """
    Y = luminance(img)
    if fr is None:
        fr = _frame_stats(img, Y)
    cands = []
    for sign in (1.0, -1.0):
        work = (1.0 - Y) if sign > 0 else Y
        for sp in detect(work, sensitivity=4.0, max_size_frac=max_size_frac, hairs=hairs):
            cands.append((sign, work, sp))
    if not cands:
        return []
    centres = np.array([[sp.cx, sp.cy] for _, _, sp in cands])
    model = _load_model()
    n_feat, zmin, imin = model[4], model[5], model[6]
    if n_feat > 20:
        fr["occ"] = _occupancy(Y.shape, [sp for _, _, sp in cands])
    rows, kept, vis, stc, stm, sgn = [], [], [], [], [], []
    for sign, work, sp in cands:
        # the model never accepts marks this weak (checked on the training frames),
        # so skip the costly measurements for them
        if sp.feats["z"] < zmin or sp.feats["iso"] < imin:
            continue
        f = _cand_features(img, work, sp, sign, fr)
        if f is None:
            continue
        d2 = ((centres - [sp.cx, sp.cy]) ** 2).sum(1)
        crowd = float(((d2 < 80 ** 2) & (d2 > 1)).sum())
        row = _design_row(f, fr, crowd)
        con_z, stand_m = float("inf"), 0.0
        if n_feat > len(row):
            c = _crisp_features(fr, sp, sign)
            if c is None:
                continue
            row += [math.log(max(c[k], 1e-4)) for k in CRISP_KEYS]
            con_z, stand_m = c["con_z"], c["stand_m"]
        rows.append(row)
        kept.append(sp)
        vis.append(con_z)
        stc.append(f["struct"])
        stm.append(stand_m if f["size"] <= RULES["sure_size"] else 0.0)
        sgn.append(sign)
    if not rows:
        return []
    X = np.nan_to_num(np.array(rows, float), nan=0.0, posinf=50.0, neginf=-50.0)
    s = model_score(X)
    if scored is not None:
        scored.extend((sp.cx, sp.cy, float(v)) for sp, v in zip(kept, s))
    gain = math.log2(max(0.25, float(sensitivity)))
    thr = model[2] - 0.5 * gain
    # clearly visible only (the specks a person heals), and in structured
    # surroundings (branches, windows, foliage) only marks the model is sure of
    vis_min = RULES["vis_min"] / math.sqrt(max(0.25, float(sensitivity)))
    ok = [s[i] > thr and vis[i] >= vis_min and (stc[i] <= RULES["busy_struct"] or s[i] > thr + RULES["busy_margin"])
          for i in range(len(s))]
    # a small, clearly visible light mark that is much sharper than everything
    # around it at its own size is dust, unless its surroundings are structured
    for i in range(len(s)):
        if (not ok[i] and sgn[i] > 0 and stm[i] >= RULES["sure_stand"] and vis[i] >= RULES["sure_con"]
                and stc[i] <= RULES["sure_struct"]):
            ok[i] = True
    order = [i for i in np.argsort(-s) if ok[i]]
    accepted = []
    for i in order:
        sp = kept[i]
        # a light mark and a dark halo can both be found at one place: keep the stronger
        if any(abs(sp.cx - o.cx) < max(sp.size, o.size) / 2 and abs(sp.cy - o.cy) < max(sp.size, o.size) / 2 for o in accepted):
            continue
        sp.score = float(s[i])
        accepted.append(sp)
        if len(accepted) >= max_spots:
            break
    return accepted


# --------------------------------------------------------------------------
# Context and merging
# --------------------------------------------------------------------------

DISC_Q = 4                                          # disc search runs at 1/4 size
DISC_SIGMAS = (1.2, 1.9, 3.0, 4.8, 7.7, 12.3)       # centre-surround scales at 1/4 size
DISC_Z = 7.0                                        # response against the noise at that scale
DISC_SMOOTH = 1.6                                   # finer detail around the disc, against the flattest areas
DISC_STRUCT = 2.2                                   # edges around the disc (classifier's structure measure)
DISC_LIFT = (0.004, 0.08)                           # how much the disc lifts its surroundings
DISC_ELONG = 1.9


def _block_noise(D, b):
    """Robust noise of a response map in blocks of b px, each block then the median of its 3x3 neighbours."""
    h, w = D.shape
    bh, bw = max(1, h // b), max(1, w // b)
    blk = np.abs(D[:bh * b, :bw * b]).reshape(bh, b, bw, b).transpose(0, 2, 1, 3).reshape(bh, bw, -1)
    raw = np.median(blk[..., ::2], axis=2).astype(np.float32) * 1.4826
    med = cv2.medianBlur(raw, 3) if min(bh, bw) >= 3 else raw
    return cv2.resize(med, (w, h), interpolation=cv2.INTER_NEAREST) + 1e-6


def find_discs(view, fr, sensitivity=1.0):
    """
    Faint, soft, round lifts on smooth ground: out-of-focus dust and the soft
    patches hard water leaves, too faint for the speck detector at full size.
    The idea of searching at reduced size, where the grain averages out, with a
    difference of Gaussians per scale measured against that scale's own noise,
    comes from film dust (Magnus Holmes, MIT licence); this is our own version.
    Returns Spot objects (circular masks, full resolution).
    """
    Y = fr["Y"]
    H, W = Y.shape
    q = DISC_Q
    w, h = W // q, H // q
    if w < 64 or h < 64:
        return []
    y4 = cv2.resize(Y[:h * q, :w * q], (w, h), interpolation=cv2.INTER_AREA)
    # detail finer than any disc, for the smooth-ground test
    fine = np.abs(y4 - cv2.GaussianBlur(y4, (0, 0), 1.0))
    fine_lvl = cv2.GaussianBlur(fine, (0, 0), 6.0)
    flat_ref = max(float(np.percentile(fine_lvl, 10)), 1e-6)
    zmin = DISC_Z / math.sqrt(max(0.25, float(sensitivity)))
    blurs = [cv2.GaussianBlur(y4, (0, 0), s) for s in DISC_SIGMAS]
    blurs.append(cv2.GaussianBlur(y4, (0, 0), DISC_SIGMAS[-1] * 1.6))
    zs, resp = [], []
    for k, s in enumerate(DISC_SIGMAS):
        dog = blurs[k] - cv2.GaussianBlur(y4, (0, 0), s * 1.6)
        noise = _block_noise(dog, int(max(16, 6 * s)))
        zs.append(dog / noise)
        resp.append(dog)
    Z = np.stack(zs)
    best = Z.max(axis=0)
    scale = Z.argmax(axis=0)
    out = []
    for k, s in enumerate(DISC_SIGMAS):
        win = int(2 * round(1.5 * s) + 1)
        peak = (Z[k] >= cv2.dilate(Z[k], np.ones((win, win), np.uint8))) & (Z[k] > zmin) & (scale == k)
        for yq, xq in zip(*np.nonzero(peak)):
            r4 = 1.6 * s                                   # disc radius at 1/4 size
            if xq < 2 * r4 or yq < 2 * r4 or xq > w - 2 * r4 or yq > h - 2 * r4:
                continue
            # shape: the region above half the peak response, around the peak
            R = int(3 * r4) + 2
            a0, a1, b0, b1 = max(0, yq - R), min(h, yq + R + 1), max(0, xq - R), min(w, xq + R + 1)
            sub = resp[k][a0:a1, b0:b1]
            reg = (sub >= 0.5 * resp[k][yq, xq]).astype(np.uint8)
            n_, lab = cv2.connectedComponents(reg, connectivity=8)
            reg = lab == lab[yq - a0, xq - b0]
            ys_, xs_ = np.nonzero(reg)
            if len(ys_) < 5:
                continue
            cov = np.cov(np.vstack([xs_, ys_]).astype(np.float64))
            ev = np.linalg.eigvalsh(cov)
            elong = math.sqrt(max(ev[1], 1e-9) / max(ev[0], 1e-9))
            if elong > DISC_ELONG:
                continue
            # lift against the ring around it, relative to its brightness
            yy, xx = np.ogrid[a0 - yq:a1 - yq, b0 - xq:b1 - xq]
            dd = np.sqrt(yy * yy + xx * xx)
            ring = (dd > 1.6 * r4) & (dd < 2.8 * r4)
            inner = dd < 0.6 * r4
            if ring.sum() < 8 or inner.sum() < 1:
                continue
            bgv = float(np.median(y4[a0:a1, b0:b1][ring]))
            lift = (float(np.mean(blurs[k][a0:a1, b0:b1][inner])) - bgv) / max(bgv, 0.02)
            if not (DISC_LIFT[0] <= lift <= DISC_LIFT[1]):
                continue
            # smooth ground: no finer detail around it than in the frame's flattest areas
            smooth = float(np.percentile(fine_lvl[a0:a1, b0:b1][ring], 90)) / flat_ref
            if smooth > DISC_SMOOTH:
                continue
            cx, cy = (xq + 0.5) * q, (yq + 0.5) * q
            rf = r4 * q
            struct, flat = _region_context(fr, int(cx - rf), int(cy - rf), int(2 * rf), int(2 * rf))
            if struct > DISC_STRUCT:
                continue
            rr = int(math.ceil(1.15 * rf))
            x0, y0 = int(cx) - rr, int(cy) - rr
            yy2, xx2 = np.ogrid[-rr:rr + 1, -rr:rr + 1]
            m = (yy2 * yy2 + xx2 * xx2) <= rr * rr
            # clip to the image
            cx0, cy0 = max(0, x0), max(0, y0)
            cx1, cy1 = min(W, x0 + 2 * rr + 1), min(H, y0 + 2 * rr + 1)
            m = m[cy0 - y0:cy1 - y0, cx0 - x0:cx1 - x0]
            if not m.any():
                continue
            sp = Spot(cx0, cy0, m, "speck", float(Z[k][yq, xq]), 2.0 * rf)
            sp.feats = dict(source="disc", z=float(Z[k][yq, xq]), lift=lift, smooth=smooth, struct=struct,
                            elong=elong, r=rf)
            out.append(sp)
    # one disc per place: the strongest
    out.sort(key=lambda s: -s.score)
    kept = []
    for sp in out:
        if any(math.hypot(sp.cx - o.cx, sp.cy - o.cy) < 0.8 * max(sp.size, o.size) / 2 for o in kept):
            continue
        kept.append(sp)
    return kept


def find_dust(view, sensitivity=1.0, max_size_frac=0.015, hairs=True, max_spots=400):
    """
    Both detectors on one image (dust showing light): specks and hairs judged by
    the learned classifier, and faint soft discs on smooth ground. Returns
    (spots, note).
    """
    Y = luminance(view)
    fr = _frame_stats(view, Y)
    specks = find_dust_learned(view, sensitivity=sensitivity, max_size_frac=max_size_frac, hairs=hairs,
                               max_spots=max_spots, fr=fr)
    discs = find_discs(view, fr, sensitivity=sensitivity)
    spots = merge_spots(specks, discs)
    n_disc = len(spots) - len(specks)
    return spots[:max_spots], "%d specks and hairs, %d soft discs" % (len(specks), n_disc)


def _region_context(fr, x, y, w, h):
    """Structure and fine texture around a region, as the classifier measures them."""
    H, W = fr["gm"].shape
    size = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    Rs = int(max(40, 4 * size))
    e = int(size / 2 + 8)
    a0, a1, b0, b1 = max(0, cy - Rs), min(H, cy + Rs), max(0, cx - Rs), min(W, cx + Rs)
    win = fr["gm"][a0:a1, b0:b1].copy()
    win[max(0, cy - e - a0):max(0, cy + e - a0), max(0, cx - e - b0):max(0, cx + e - b0)] = 0
    struct = float(np.percentile(win, 99)) / fr["gnoise"]
    R = int(max(48, 1.5 * size))
    a0, a1, b0, b1 = max(0, cy - R), min(H, cy + R), max(0, cx - R), min(W, cx + R)
    wide = fr["hpY"][a0:a1, b0:b1]
    excl = np.ones(wide.shape, bool)
    excl[max(0, y - a0 - 4):max(0, y + h - a0 + 4), max(0, x - b0 - 4):max(0, x + w - b0 + 4)] = False
    flat = float(np.percentile(np.abs(wide[excl]), 90)) / fr["noise"] if excl.sum() > 20 else 99.0
    return struct, flat


def merge_spots(primary, extra):
    """All of primary, plus the spots of extra that don't overlap one of them."""
    out = list(primary)
    for sp in extra:
        if any(sp.x0 < o.x1 + 2 and o.x0 < sp.x1 + 2 and sp.y0 < o.y1 + 2 and o.y0 < sp.y1 + 2 for o in primary):
            continue
        out.append(sp)
    return out


# --------------------------------------------------------------------------
# Repair
# --------------------------------------------------------------------------

def dust_map(shape, spots):
    dust = np.zeros(shape, bool)
    for sp in spots:
        dust[sp.y0:sp.y1, sp.x0:sp.x1] |= sp.mask
    return dust


def heal(img, Y, spots, allowed=None):
    """Repair each spot in place by copying a well-matching nearby patch with tone correction."""
    dust = dust_map(Y.shape, spots)
    repaired = 0
    for sp in spots:
        geo = _spot_geometry(Y.shape, sp)
        if geo is None:
            continue
        src = find_source(Y, dust, sp, geo, allowed)
        if src is not None:
            _apply_heal(img, Y, sp, geo, src)
            repaired += 1
    return repaired


def _spot_geometry(shape, sp):
    """Working window around a spot: (X0, Y0, X1, Y1, M, ring, t, pad)."""
    H, W = shape
    t = max(2.0, sp.thick)
    pad = int(round(0.6 * t)) + 3
    X0, Y0 = max(0, sp.x0 - pad), max(0, sp.y0 - pad)
    X1, Y1 = min(W, sp.x1 + pad), min(H, sp.y1 + pad)
    h, w = Y1 - Y0, X1 - X0
    M = np.zeros((h, w), np.uint8)
    M[sp.y0 - Y0:sp.y1 - Y0, sp.x0 - X0:sp.x1 - X0] = sp.mask
    ring = (cv2.dilate(M, _disk(max(2, pad - 1))) > 0) & (M == 0)
    if ring.sum() < 6:
        return None
    return X0, Y0, X1, Y1, M, ring, t, pad


def find_source(Y, dust, sp, geo, allowed=None):
    """
    Search around the spot for the patch whose surroundings match best.
    Returns the offset (dx, dy) in pixels, or None.
    allowed: optional boolean map; the source must lie entirely inside it.
    """
    H, W = Y.shape
    X0, Y0, X1, Y1, M, ring, t, pad = geo
    h, w = Y1 - Y0, X1 - X0
    T = Y[Y0:Y1, X0:X1]
    Mb = M > 0
    cover = Mb | ring
    R = 0.5 * math.hypot(h, w)
    dists = sorted(set([int(round(v)) for v in (2.5 * t + pad, 4 * t + pad, 6 * t + pad, 1.2 * R, 1.8 * R, 2.6 * R)]))
    best, best_cost = None, None
    for dd in dists:
        for k in range(16):
            a = 2 * math.pi * k / 16
            dx, dy = int(round(dd * math.cos(a))), int(round(dd * math.sin(a)))
            sx0, sy0 = X0 + dx, Y0 + dy
            if sx0 < 0 or sy0 < 0 or sx0 + w > W or sy0 + h > H:
                continue
            # the source must not be dust, must not overlap the spot itself,
            # and must lie inside the picture
            if (dust[sy0:sy0 + h, sx0:sx0 + w] & cover).any():
                continue
            if allowed is not None and not allowed[sy0:sy0 + h, sx0:sx0 + w][cover].all():
                continue
            S = Y[sy0:sy0 + h, sx0:sx0 + w]
            diff = (T - S)[ring]
            cost = float(diff.var()) * (1.0 + 0.04 * dd / max(t, 1.0))
            if best_cost is None or cost < best_cost:
                best, best_cost = (dx, dy), cost
    return best


def _apply_heal(img, Y, sp, geo, src):
    X0, Y0, X1, Y1, M, ring, t, pad = geo
    dx, dy = src
    h, w = Y1 - Y0, X1 - X0
    sx0, sy0 = X0 + dx, Y0 + dy
    Mb = M > 0
    # tone correction: interpolate the target-source difference from the ring into the spot
    sigma = max(1.5, 0.5 * t)
    wring = ring.astype(np.float32)
    wb = cv2.GaussianBlur(wring, (0, 0), sigma) + 1e-6
    alpha = cv2.GaussianBlur(M.astype(np.float32), (0, 0), 0.8)
    alpha = np.clip(alpha * 1.6, 0, 1)
    alpha[Mb] = np.maximum(alpha[Mb], 0.999)
    region = img[Y0:Y1, X0:X1]
    srcimg = img[sy0:sy0 + h, sx0:sx0 + w]
    for c in range(img.shape[2]):
        Tc, Sc = region[..., c], srcimg[..., c]
        corr = cv2.GaussianBlur((Tc - Sc) * wring, (0, 0), sigma) / wb
        filled = Sc + corr
        region[..., c] = alpha * filled + (1 - alpha) * Tc
    img[Y0:Y1, X0:X1] = region
    Y[Y0:Y1, X0:X1] = luminance(region)


# --------------------------------------------------------------------------
# Lightroom heal spots
#
# Lightroom stores spot removal as "RetouchAreas": each area has a brush mask
# made of dabs ("d x y", normalised to the stored, un-oriented image), a brush
# radius (a fraction of the stored image width) and a source position. For
# these spots the plugin renders the photo with crop, lens corrections and
# transforms switched off, so a pixel in the render is the stored image under
# its orientation only.
# --------------------------------------------------------------------------

_POS = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (1.0, 1.0), "D": (0.0, 1.0)}
_NEIGHBOURS = {"A": "BD", "B": "AC", "C": "BD", "D": "CA"}


def orientation_affine(code):
    """A, B, C, D are displayed corners; the code says where stored TL and TR end up."""
    code = (code or "AB").strip().upper()
    if len(code) != 2 or code[0] not in _POS or code[1] not in _NEIGHBOURS[code[0]]:
        code = "AB"
    p0 = np.array(_POS[code[0]])
    p1 = np.array(_POS[code[1]])
    other = [c for c in _NEIGHBOURS[code[0]] if c != code[1]][0]
    p2 = np.array(_POS[other])
    return np.column_stack([p1 - p0, p2 - p0]), p0, code


def stored_to_displayed(pt, code):
    A, p0, _ = orientation_affine(code)
    return p0 + A @ np.asarray(pt, dtype=float)


def displayed_to_stored(pt, code):
    A, p0, _ = orientation_affine(code)
    return np.linalg.solve(A, np.asarray(pt, dtype=float) - p0)


def stored_width_px(W, H, code):
    """Width of the stored image in render pixels (the axis Lightroom's brush radius is measured on)."""
    _, _, c = orientation_affine(code)
    return W if c in ("AB", "BA", "CD", "DC") else H


def crop_polygon(crop, code, W, H):
    """
    Corners (render pixels) of the Lightroom crop drawn on the uncropped, oriented
    render. crop = (left, top, right, bottom, angle): two opposite corners in
    stored normalised coordinates plus the clockwise tilt in the displayed frame.
    """
    left, top, right, bottom, angle = crop
    p1 = stored_to_displayed((left, top), code) * [W, H]
    p2 = stored_to_displayed((right, bottom), code) * [W, H]
    t = math.radians(angle)
    u = np.array([math.cos(t), math.sin(t)])
    v = np.array([-math.sin(t), math.cos(t)])
    d = p2 - p1
    w, h = d @ u, d @ v
    return np.array([p1, p1 + w * u, p2, p1 + h * v])


def _order_path(points, max_gap=6.0):
    """Greedy nearest-neighbour ordering, starting from the point farthest from the centre."""
    pts = [tuple(p) for p in points]
    if len(pts) <= 2:
        return [pts]
    c = np.mean(pts, axis=0)
    cur = max(pts, key=lambda p: (p[0] - c[0]) ** 2 + (p[1] - c[1]) ** 2)
    left = set(pts)
    left.remove(cur)
    paths, path = [], [cur]
    while left:
        nxt = min(left, key=lambda p: (p[0] - cur[0]) ** 2 + (p[1] - cur[1]) ** 2)
        path_gap = math.hypot(nxt[0] - cur[0], nxt[1] - cur[1])
        left.remove(nxt)
        if path_gap > max_gap:      # a jump: start a new stroke (branching fibre)
            paths.append(path)
            path = []
        path.append(nxt)
        cur = nxt
    paths.append(path)
    return paths


def _resample(path, step):
    """Insert points so consecutive dabs are at most `step` apart."""
    out = [path[0]]
    for p in path[1:]:
        q = out[-1]
        dist = math.hypot(p[0] - q[0], p[1] - q[1])
        n = int(math.ceil(dist / step)) if step > 0 else 1
        for k in range(1, n + 1):
            out.append((q[0] + (p[0] - q[0]) * k / n, q[1] + (p[1] - q[1]) * k / n))
    return out


def spot_strokes(sp):
    """
    Brush strokes covering a spot, in render pixels: list of (radius, [(x, y), ...]).
    A speck becomes a single dab; a hair becomes one or more strokes along its length.
    """
    m = sp.mask.astype(np.uint8)
    if sp.kind != "hair":
        ys, xs = np.nonzero(m)
        cx, cy = xs.mean(), ys.mean()
        rmax = float(np.sqrt(((xs - cx) ** 2 + (ys - cy) ** 2).max())) + 1.0
        r = max(3.0, rmax / 0.85)          # Lightroom feathers the inner 15 %
        return [(r, [(sp.x0 + cx, sp.y0 + cy)])]
    dt = cv2.distanceTransform(cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0), cv2.DIST_L2, 3)[1:-1, 1:-1]
    r = max(2.5, (float(dt.max()) + 1.0) / 0.85)
    ridge = (dt > 0) & (dt >= cv2.dilate(dt, np.ones((3, 3), np.uint8)) - 1e-3)
    ys, xs = np.nonzero(ridge)
    # thin the ridge points to roughly one per pixel step
    keep = {}
    for x, y in zip(xs, ys):
        keep[(int(x) // 2, int(y) // 2)] = (float(x), float(y))
    paths = _order_path(list(keep.values()), max_gap=max(6.0, 2.5 * r))
    strokes = []
    for path in paths:
        if not path:
            continue
        dabs = _resample(path, 0.25 * r)
        dabs = [(sp.x0 + x, sp.y0 + y) for x, y in dabs]
        # Lightroom 14+ stores long strokes in a sealed format; keep each well under 250 dabs
        for k in range(0, len(dabs), 180):
            chunk = dabs[k:k + 181]
            strokes.append((r, chunk))
    return strokes


def lightroom_spots(spots, sources, W, H, code, origin=(0, 0)):
    """
    Text lines describing heal spots for the Lua side:
        spot <TAB> kind <TAB> radius <TAB> sourceX <TAB> sourceY <TAB> x y;x y;...
    All coordinates normalised to the stored image; origin offsets the render coordinates.
    """
    ox, oy = origin
    ws = float(stored_width_px(W, H, code))
    lines = []

    def norm(x, y):
        p = displayed_to_stored(((x + ox) / W, (y + oy) / H), code)
        return float(p[0]), float(p[1])

    for sp, src in zip(spots, sources):
        if src is None:
            continue
        dx, dy = src
        for r, dabs in spot_strokes(sp):
            nd = [norm(x, y) for x, y in dabs]
            sx, sy = norm(dabs[0][0] + dx, dabs[0][1] + dy)
            if not all(0 <= c <= 1 for p in nd for c in p) or not (0 <= sx <= 1 and 0 <= sy <= 1):
                continue
            lines.append("\t".join([
                "spot", sp.kind, "%.6f" % (r / ws), "%.6f" % sx, "%.6f" % sy,
                ";".join("%.6f %.6f" % p for p in nd)]))
    return lines


# --------------------------------------------------------------------------
# Preview
# --------------------------------------------------------------------------

def save_preview(path, img, spots, long_edge=2400):
    H, W = img.shape[:2]
    f = min(1.0, long_edge / float(max(H, W)))
    small = cv2.resize(img, (int(W * f), int(H * f)), interpolation=cv2.INTER_AREA)
    lo, hi = np.percentile(small, 0.5), np.percentile(small, 99.5)
    small = np.clip((small - lo) / max(hi - lo, 1e-3), 0, 1)
    bgr = cv2.cvtColor((small * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    for sp in spots:
        c = (int(sp.cx * f), int(sp.cy * f))
        r = int(max(6, 0.5 * sp.size * f + 4))
        col = (255, 200, 0) if sp.kind == "hair" else (0, 0, 255)
        if sp.kind == "hair":
            cv2.rectangle(bgr, (int(sp.x0 * f) - 3, int(sp.y0 * f) - 3), (int(sp.x1 * f) + 3, int(sp.y1 * f) + 3), col, 2)
        else:
            cv2.circle(bgr, c, r, col, 2, cv2.LINE_AA)
    n_s = sum(1 for s in spots if s.kind == "speck")
    n_h = len(spots) - n_s
    label = "%d specks, %d hairs" % (n_s, n_h)
    cv2.putText(bgr, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(bgr, label, (12, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if ok:
        buf.tofile(path)


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------

def _find(img, args):
    """
    Find dust with the learned classifier. The model was trained on positives
    made from negatives, where dust usually shows light; for raw negatives and
    slides (dust shows dark) the image is inverted first so it looks the same.
    Returns (luminance of the original image, spots).
    """
    Y = luminance(img)
    view = img if args.dust == "bright" else (1.0 - img)
    spots, note = find_dust(view, sensitivity=args.sensitivity, max_size_frac=args.max_size / 100.0,
                            hairs=not args.no_hairs, max_spots=args.max_spots)
    return Y, spots, note


def process(inp, outp, prev, args):
    """Cleaned-TIFF mode: the render already has the photo's crop applied."""
    img, dtype, icc, alpha = load_image(inp)
    Y, spots, note = _find(img, args)
    if prev:
        save_preview(prev, img, spots)
    if outp:
        heal(img, Y, spots)
        save_tiff(outp, img, dtype, icc, alpha)
    n_h = sum(1 for s in spots if s.kind == "hair")
    return len(spots) - n_h, n_h, note


def process_spots(inp, spots_out, prev, crop, orientation, args):
    """
    Heal-spot mode: the render is uncropped with lens corrections and transforms
    off. Dust is looked for inside the photo's crop only, and written out as
    Lightroom heal spots.
    """
    img, _, _, _ = load_image(inp)
    H, W = img.shape[:2]
    poly = crop_polygon(crop, orientation, W, H)
    inside = np.zeros((H, W), np.uint8)
    cv2.fillPoly(inside, [np.round(poly).astype(np.int32)], 1)
    # work on the bounding box of the crop only
    bx0, by0 = int(max(0, np.floor(poly[:, 0].min()))), int(max(0, np.floor(poly[:, 1].min())))
    bx1, by1 = int(min(W, np.ceil(poly[:, 0].max()))), int(min(H, np.ceil(poly[:, 1].max())))
    if bx1 - bx0 < 32 or by1 - by0 < 32:
        raise RuntimeError("crop area is too small")
    sub = np.ascontiguousarray(img[by0:by1, bx0:bx1])
    del img
    allowed = inside[by0:by1, bx0:bx1]
    edge = max(3, int(round(0.003 * min(sub.shape[:2]))))
    allowed = cv2.erode(allowed, np.ones((2 * edge + 1, 2 * edge + 1), np.uint8)) > 0
    Y, spots, note = _find(sub, args)
    spots = [sp for sp in spots
             if allowed[min(int(sp.cy), allowed.shape[0] - 1), min(int(sp.cx), allowed.shape[1] - 1)]]
    dust = dust_map(Y.shape, spots)
    sources = []
    for sp in spots:
        geo = _spot_geometry(Y.shape, sp)
        sources.append(find_source(Y, dust, sp, geo, allowed) if geo is not None else None)
    lines = lightroom_spots(spots, sources, W, H, orientation, origin=(bx0, by0))
    with open(spots_out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    if prev:
        save_preview(prev, sub, [sp for sp, s in zip(spots, sources) if s is not None])
    kept = [sp for sp, s in zip(spots, sources) if s is not None]
    n_h = sum(1 for s in kept if s.kind == "hair")
    return len(kept) - n_h, n_h, len(lines), note


def run_job(args):
    results = []
    with open(args.job, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\r\n") for ln in f if ln.strip()]
    for ln in lines:
        parts = ln.split("\t")
        pid = parts[0]
        try:
            inp = parts[1]
            outp = parts[2] if len(parts) > 2 and parts[2] not in ("", "-") else None
            prev = parts[3] if len(parts) > 3 and parts[3] not in ("", "-") else None
            spots_out = parts[4] if len(parts) > 4 and parts[4] not in ("", "-") else None
            if spots_out:
                crop = tuple(float(v) for v in parts[5].split(","))
                orientation = parts[6] if len(parts) > 6 else "AB"
                n_s, n_h, n_lines, note = process_spots(inp, spots_out, prev, crop, orientation, args)
                results.append("\t".join([pid, "ok", str(n_s), str(n_h), spots_out, str(n_lines), note]))
            else:
                n_s, n_h, note = process(inp, outp, prev, args)
                results.append("\t".join([pid, "ok", str(n_s), str(n_h), outp or "-", "-", note]))
        except Exception as e:
            msg = (str(e) or e.__class__.__name__).replace("\t", " ").replace("\n", " ")
            results.append("\t".join([pid, "fail", msg]))
            if args.verbose:
                traceback.print_exc()
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(results) + "\n")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true")
    p.add_argument("--job")
    p.add_argument("--out")
    p.add_argument("--dust", default="dark", choices=["dark", "bright"],
                   help="dark: raw negatives and slides; bright: converted negatives")
    p.add_argument("--sensitivity", type=float, default=1.0)
    p.add_argument("--max-size", type=float, default=1.5, help="largest speck, percent of the short side")
    p.add_argument("--no-hairs", action="store_true")
    p.add_argument("--max-spots", type=int, default=3000)
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args(argv)
    if args.check:
        if _MISSING:
            print("missing %s (Python %s at %s)" % (", ".join(_MISSING), sys.version.split()[0], sys.executable))
            return 1
        print("ok python %s numpy %s opencv %s tifffile %s" % (
            sys.version.split()[0], np.__version__, cv2.__version__, tifffile.__version__))
        return 0
    if not args.job or not args.out:
        p.error("--job and --out are required")
    if _MISSING:
        print("missing Python packages: " + ", ".join(_MISSING))
        return 1
    run_job(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
