"""Pure-torch core for optional film emulation, the last stage of the pipeline.

Film emulation makes a finished render look shot on film. Its stage today is
film grain (``add_grain``), described below.

No ComfyUI / Forge imports, so the same file drives every host and the offline
tests. The three image helpers are copied from vae_enhance_core.py on purpose:
each host loads the cores by file path under a private module name, so a bare
import between them would fail in Forge.

What it does
------------
Adds a seeded, spatially correlated, luminance-weighted noise field to the
colour channels of a finished image. Two controls, deliberately orthogonal:

* ``iso`` picks a preset that sets the grain's *character*: blob radius,
  tail-heaviness (fine film sums to Gaussian, fast film clumps) and how much
  colour grain a colour image gets. It never changes the amplitude.
* ``strength`` 0..100 sets the *amplitude* alone, in /255 units of midtone
  luma: 0.3 at 1 (under the midtone just-noticeable difference of ~3/255),
  3.3 at 50 (at threshold: reads as texture, not noise), 8 at 100 (grainy).
* ``match_texture`` (default on) keeps the *perceived* amount predictable.
  Texture an image already carries masks added grain (contrast masking: the
  visibility threshold rises with the masker's contrast, slope ~0.6 above
  threshold, Legge & Foley 1980). A clean render shows every bit of the
  grain, so it gets exactly the strength's amount; a textured or already
  grainy render gets more, (floor / 1.1) ** 0.6, at most 1.4x and at most
  12/255, so the grain stays as visible as it is on a clean image. It never
  reduces the amount: an earlier version scaled grain *down* on clean images
  and made strength 50 invisible on exactly the renders that need grain.

Why it looks like film and not like sensor noise:

* **Correlated.** White noise has zero neighbour correlation; grain clumps
  span 1-3 px. A Gaussian-blurred field at the preset sigma, renormalised,
  gives a lag-one autocorrelation of ~0.3-0.6 and reads as grain. The sigma
  grows with the *square root* of the image's long side (reference 1536 px).
  Fully frame-anchored scaling made the same preset 1.5x coarser at 1536 than
  at 1024 when viewed at 1:1, which is how grain is judged; fully
  pixel-anchored scaling would make a large image's grain vanish in a
  fit-to-screen view. The square root splits the difference.
* **Grain size follows exposure.** On film only the largest, most sensitive
  crystals develop where little light arrived, so shadows and underexposed
  areas show sparse, coarse, clumpy grain; where much light arrived crystals of
  every size develop and overlap into fine grain. The field is blurred twice
  from the same noise, coarse (preset sigma x 1.35) and fine (/ 1.35), and the
  two are blended per pixel by luminance, renormalised so only the size and
  clumpiness change, never the amount.
* **Amount follows tone, by film type.** ``film`` = "print" (default: the
  model renders prints and reversal slides) puts the most grain in the darks
  (~1.6x the midtone amount), less in the mids, little in the brights and
  none on pure black or pure white. "negative" (a negative scan) puts more in
  the highlights, where a scan of the dense negative is noisiest. The amount
  is never more than about half the pixel's distance to black or white, so
  grain never clips into raised blacks.
* **No grain on paper.** A uniform band along at least three image edges (a
  Polaroid or print border, a film-scan rebate) is detected and left
  bit-exact; inside the picture, flat near-white areas (paper white, a blown
  flat highlight) get none either. Grain lives in the emulsion, not on paper.
* **Luma grain is exactly neutral.** The same value is added to R, G and B, so
  a grey pixel stays grey to the last bit.
* **Colour grain never invents a hue.** It is a *multiplicative* fluctuation of
  each pixel's existing chroma vector, so its size is proportional to the
  pixel's own saturation: a neutral pixel gets exactly none, a faint tint gets
  a faint fluctuation of that tint, a saturated red gets visible colour grain,
  and the hue never moves. A frame whose mean chroma is under 8/255 (a "B&W"
  decode with a slight cast) is detected as monochrome and gets no colour
  grain at all. ``chroma`` = "auto" | "on" | "off".

Judge the result at 1:1. A downscaled preview hides grain, 2:1 exaggerates it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

ISO_NAMES = ("ISO 100", "ISO 200", "ISO 400", "ISO 800", "ISO 1600", "ISO 3200")
ISO_DEFAULT = "ISO 200"  # the model's own prompted grain measures a lag-one autocorrelation of 0.34 after the 9 px high-pass; ISO 200 measures 0.35
STRENGTH_DEFAULT = 50.0
CHROMA_MODES = ("auto", "on", "off")

# sigma: blur of the noise field in px at a 1536 px long side (blob radius);
# tail: cubic term that makes the amplitude distribution clumpy (kurtosis > 3);
# chroma: fraction of the amplitude that goes into colour grain on saturated pixels.
# Calibrated 2026-10-03 (knowledge_degrid.md §8.12): sigma against the model's own prompted
# grain (lag-one autocorrelation 0.34 after a 9 px high-pass; ISO 200 measures 0.35 and is the
# default, confirmed by eye on a smooth-skin portrait at strength 50); tail against kurtosis 3.0
# at ISO 100 rising to ~4.2 at ISO 3200 (x + t·x³ on a Gaussian field: t 0.01 -> 3.26,
# 0.02 -> 3.54, 0.03 -> 3.85, 0.04 -> ~4.2). At equal amplitude the presets differ in kind, not
# in visibility (CSF-weighted factors 1.00-1.16): ISO 100 blends into skin as fine texture,
# ISO 400 and up read as clumps sitting on it.
ISO_PRESETS: dict[str, dict[str, float]] = {
    "ISO 100": {"sigma": 0.40, "tail": 0.000, "chroma": 0.05},
    "ISO 200": {"sigma": 0.50, "tail": 0.005, "chroma": 0.08},
    "ISO 400": {"sigma": 0.60, "tail": 0.010, "chroma": 0.12},
    "ISO 800": {"sigma": 0.75, "tail": 0.020, "chroma": 0.18},
    "ISO 1600": {"sigma": 0.95, "tail": 0.030, "chroma": 0.25},
    "ISO 3200": {"sigma": 1.20, "tail": 0.040, "chroma": 0.35},
}
SCALE_REF = 1536  # px long side at which the preset sigmas apply
SIGMA_SCALE_EXP = 0.5  # sigma ~ (long side / SCALE_REF) ** this; 1 = frame-anchored, 0 = pixel-anchored
# match_texture: the texture floor is the 10th percentile, over 64 px blocks whose mean
# luminance weight is > 0.5 (blocks where grain would actually show), of the std of the
# 9 px box high-pass of luma (knowledge_degrid.md §8.15). At or below FLOOR_REF_255 the image
# counts as clean and gets the strength's amount; above it the amount is raised to make up for
# masking, never lowered.
FLOOR_REF_255 = 1.1  # floors of the renders where strength-50 grain was judged by eye: 0.45-1.23
MASK_EXP = 0.6  # contrast-masking slope above threshold (Legge & Foley 1980: ~0.6-0.7)
MASK_MAX_FACTOR = 1.4  # 2.0 overshot once combined with the dark weighting (night street, 2026-10-04)
AMP_CAP_255 = 12.0  # no boost takes the midtone amplitude above this
FLOOR_BLOCK = 64
FLOOR_PERCENTILE = 10.0
AMP_MIN_255, AMP_MAX_255, AMP_GAMMA = 0.3, 8.0, 1.35
CHROMA_REF_255 = 40.0  # saturation at which the colour grain's std equals chroma * luma amplitude
MONO_THRESHOLD_255 = 8.0  # frame mean chroma magnitude below this = monochrome (a Krea 2 "B&W" decode carries ~5.6 of tint; colour photos 20+)

# Per-ISO loudness: coarse, clumpy grain reads louder than its amplitude, so every preset gives
# the same perceived amount at one strength. Multiplies the amplitude. ISO 100, 800 and 3200
# were matched by eye against ISO 200 on a B&W portrait (strength 50 and 100); they follow
# about (0.5 / sigma) ** 0.5. ISO 400 and 1600 are interpolated in log sigma.
ISO_LOUDNESS: dict[str, float] = {
    "ISO 100": 1.15, "ISO 200": 1.0, "ISO 400": 0.9, "ISO 800": 0.8, "ISO 1600": 0.72, "ISO 3200": 0.65,
}

FILM_TYPES = ("print", "negative")
FILM_DEFAULT = "print"
# Amount per tone: (luma, weight) knots, linearly interpolated, weight 1 at the 0.45 midtone where
# strength is defined.
AMOUNT_CURVES: dict[str, tuple[tuple[float, float], ...]] = {
    "print": ((0.0, 0.0), (0.03, 1.55), (0.10, 1.75), (0.25, 1.55), (0.45, 1.0), (0.65, 0.6), (0.80, 0.33), (0.90, 0.12), (0.95, 0.0), (1.0, 0.0)),
    "negative": ((0.0, 0.0), (0.03, 0.45), (0.12, 0.6), (0.45, 1.0), (0.70, 1.25), (0.85, 1.3), (0.92, 0.8), (0.97, 0.0), (1.0, 0.0)),
}
MIDTONE_REF = 0.45
HEADROOM = 0.5  # local grain std is at most this x the distance to black / white (+ 1/255)
# Grain size by exposure: coarse below EXPO_LO luma, fine above EXPO_HI, smoothstep between.
SIZE_SPREAD = 1.35  # coarse sigma = preset x this, fine sigma = preset / this (1.8:1 shadows to highlights)
EXPO_LO, EXPO_HI = 0.15, 0.75
TAIL_SHADOW_ADD = 0.015  # sparse shadow grain clumps more
TAIL_HIGHLIGHT_MUL = 0.5
# Paper white inside the picture: 16 px blocks brighter than PAPER_L, flatter than PAPER_HP_255
# (std of the 9 px high-pass) and greyer than PAPER_CHROMA_255 get no grain.
PAPER_BLOCK = 16
PAPER_L = 0.85
PAPER_HP_255 = 0.8
PAPER_CHROMA_255 = 6.0
# Edge frames: per side, a run of rows (columns) whose 5-95 % luma range is under FRAME_RANGE_255,
# starting within the outer FRAME_SEARCH of the dimension (card edges and backgrounds are messy),
# at least FRAME_MIN thick, followed by picture rows (at least 6 of the next 12 over FRAME_PICTURE_255). At least three
# sides, paper colours within FRAME_PAPER_SPREAD_255. Measured on generated Polaroids: paper
# rows vary 5-13/255, picture rows 170-200, card edges up to 90.
FRAME_RANGE_255 = 25.0
FRAME_PICTURE_255 = 50.0
FRAME_SEARCH = 0.03
FRAME_MIN = 0.012
FRAME_PAPER_SPREAD_255 = 18.0
FRAME_FEATHER = 3  # px of soft edge inside the picture rectangle
_LUMA = (0.299, 0.587, 0.114)
_BELL_A, _BELL_B = 0.5, 0.8  # luminance weight = L^a (1-L)^b, normalised to 1 at its peak


# -- helpers (copied from vae_enhance_core) ---------------------------------------


def luma(x_bchw: torch.Tensor) -> torch.Tensor:
    w = torch.tensor(_LUMA, dtype=x_bchw.dtype, device=x_bchw.device).view(1, 3, 1, 1)
    return (x_bchw[:, :3] * w).sum(1, keepdim=True)


def box(x_bchw: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x_bchw
    p = k // 2
    mode = "reflect" if min(x_bchw.shape[-2:]) > p else "replicate"
    return F.avg_pool2d(F.pad(x_bchw, (p, p, p, p), mode=mode), k, stride=1)


def gaussian_blur(x_bchw: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 0:
        return x_bchw
    radius = max(1, int(3 * sigma + 0.5))
    t = torch.arange(-radius, radius + 1, dtype=x_bchw.dtype, device=x_bchw.device)
    k = torch.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    c = x_bchw.shape[1]
    mode = "reflect" if min(x_bchw.shape[-2:]) > radius else "replicate"
    y = F.pad(x_bchw, (radius, radius, radius, radius), mode=mode)
    y = F.conv2d(y, k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    y = F.conv2d(y, k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)
    return y


# -- the pieces ---------------------------------------------------------------------


def strength_to_amp(strength: float) -> float:
    """Midtone luma amplitude in /255 units. 1 -> 0.3, 50 -> 3.3, 100 -> 8.0; 0 -> 0."""
    s = max(0.0, min(100.0, float(strength)))
    if s <= 0.0:
        return 0.0
    return AMP_MIN_255 + (AMP_MAX_255 - AMP_MIN_255) * (s / 100.0) ** AMP_GAMMA


_bell_peak = (_BELL_A / (_BELL_A + _BELL_B)) ** _BELL_A * (_BELL_B / (_BELL_A + _BELL_B)) ** _BELL_B


def luminance_weight(L: torch.Tensor) -> torch.Tensor:
    """Midtone bell in luma 0..1, equal to 1 at its peak (L ~ 0.38), 0 at black and white.
    Selects the midtone blocks the texture floor is measured on; the grain amount uses amount_weight."""
    L = L.clamp(0.0, 1.0)
    return (L ** _BELL_A) * ((1.0 - L) ** _BELL_B) / _bell_peak


def amount_weight(L: torch.Tensor, film: str = FILM_DEFAULT) -> torch.Tensor:
    """Grain amount per tone for a film type (AMOUNT_CURVES), 1 at the midtone reference."""
    knots = AMOUNT_CURVES[film]
    xs = torch.tensor([k[0] for k in knots], dtype=L.dtype, device=L.device)
    ys = torch.tensor([k[1] for k in knots], dtype=L.dtype, device=L.device)
    Lc = L.clamp(0.0, 1.0)
    idx = torch.searchsorted(xs, Lc.contiguous(), right=True).clamp(1, len(knots) - 1)
    x0, x1, y0, y1 = xs[idx - 1], xs[idx], ys[idx - 1], ys[idx]
    return y0 + (y1 - y0) * (Lc - x0) / (x1 - x0)


def exposure_mix(L: torch.Tensor) -> torch.Tensor:
    """0 = fine grain (well exposed), 1 = coarse grain (underexposed), smoothstep in luma."""
    t = ((EXPO_HI - L.clamp(0.0, 1.0)) / (EXPO_HI - EXPO_LO)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def blob_sigma(iso: str, height: int, width: int, scale_ref: int = SCALE_REF) -> float:
    return ISO_PRESETS[iso]["sigma"] * (max(height, width) / float(scale_ref)) ** SIGMA_SCALE_EXP


def texture_floor_255(x_bchw: torch.Tensor) -> float:
    """The fine texture an image already carries in its flat midtones, /255 (see FLOOR_* above)."""
    L = luma(x_bchw[:1])
    hp = (L - box(L, 9)) * 255.0
    w = luminance_weight(L)
    h, wd = hp.shape[-2:]
    block = FLOOR_BLOCK if min(h, wd) >= 2 * FLOOR_BLOCK else max(8, min(h, wd) // 4)
    hb, wb = h // block, wd // block
    if hb == 0 or wb == 0:
        return float(hp.std())
    def tiles(t):
        return t[:, :, : hb * block, : wb * block].reshape(hb, block, wb, block).permute(0, 2, 1, 3).reshape(hb * wb, -1)
    std = tiles(hp).std(dim=1)
    keep = tiles(w).mean(dim=1) > 0.5
    v = std[keep] if int(keep.sum()) >= 8 else std
    return float(torch.quantile(v.float(), FLOOR_PERCENTILE / 100.0))


def texture_factor(floor_255: float) -> float:
    """Amplitude multiplier for match_texture: 1 on a clean image (floor <= FLOOR_REF_255),
    (floor / FLOOR_REF_255) ** MASK_EXP above it, at most MASK_MAX_FACTOR. Never below 1."""
    return float(min(MASK_MAX_FACTOR, max(1.0, (max(floor_255, 0.0) / FLOOR_REF_255) ** MASK_EXP)))


def _unit(n: torch.Tensor) -> torch.Tensor:
    return (n - n.mean()) / (n.std() + 1e-12)


def grain_field(height: int, width: int, sigma: float, tail: float, seed: int, device, channels: int = 1) -> torch.Tensor:
    """Seeded noise field(s), blurred to the blob size, clumped by ``tail``, unit std each. [1, channels, H, W]

    Drawn on the CPU generator so a seed gives the same grain on any device.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    n = torch.randn(1, channels, height, width, generator=g).to(device)
    out = []
    for c in range(channels):
        f = n[:, c : c + 1]
        if sigma > 0.3:
            f = gaussian_blur(f, sigma)
        f = _unit(f)
        if tail > 0:
            f = _unit(f + float(tail) * f ** 3)
        out.append(f)
    return torch.cat(out, dim=1)


def grain_field_exposure(L: torch.Tensor, sigma: float, tail: float, seed: int, channels: int = 1) -> torch.Tensor:
    """Unit-variance grain whose blob size and clumpiness follow exposure. [1, channels, H, W]

    Each channel's white noise is blurred coarse and fine; the two are blended per pixel by
    exposure_mix(L) and renormalised with their measured correlation, so the local std is 1
    everywhere and only the size changes. The clumping term a·x³ is applied per pixel and divided
    by sqrt(1 + 6a + 15a²), the exact std of x + a·x³ for a unit Gaussian.
    """
    h, w = L.shape[-2:]
    dev = L.device
    g = torch.Generator(device="cpu").manual_seed(int(seed) & 0x7FFFFFFF)
    n = torch.randn(1, channels, h, w, generator=g).to(dev)
    t = exposure_mix(L)
    s_coarse, s_fine = sigma * SIZE_SPREAD, sigma / SIZE_SPREAD
    a = float(tail) * TAIL_HIGHLIGHT_MUL + (float(tail) + TAIL_SHADOW_ADD - float(tail) * TAIL_HIGHLIGHT_MUL) * t
    out = []
    for c in range(channels):
        nc = n[:, c : c + 1]
        fc = _unit(gaussian_blur(nc, s_coarse) if s_coarse > 0.3 else nc)
        ff = _unit(gaussian_blur(nc, s_fine) if s_fine > 0.3 else nc)
        rho = float((fc * ff).mean())
        f = ((1.0 - t) * ff + t * fc) / torch.sqrt((1.0 - t) ** 2 + t ** 2 + 2.0 * t * (1.0 - t) * rho)
        f = (f + a * f ** 3) / torch.sqrt(1.0 + 6.0 * a + 15.0 * a ** 2)
        out.append(f)
    return torch.cat(out, dim=1)


def detect_frame(x_bchw: torch.Tensor) -> dict[str, Any] | None:
    """A uniform border band along at least three edges (print / Polaroid border, film rebate).
    Returns {"top", "bottom", "left", "right"} band thicknesses in px and "paper_255", or None."""
    L = luma(x_bchw[:1])[0, 0] * 255.0
    H, W = L.shape

    def side(lines: torch.Tensor):
        n = lines.shape[0]
        q = torch.quantile(lines.float(), torch.tensor([0.05, 0.95], device=lines.device), dim=1)
        rng = (q[1] - q[0]).cpu()
        mean = lines.float().mean(dim=1).cpu()
        like = rng < FRAME_RANGE_255
        search = max(8, int(FRAME_SEARCH * n * 3))  # lines holds a third of the dimension
        best = None
        k = 0
        while k < min(search, n):
            if like[k]:
                e = k
                while e < n and like[e]:
                    e += 1
                if best is None or (e - k) > (best[1] - best[0]):
                    best = (k, e)
                k = e
            else:
                k += 1
        if best is None:
            return None
        start, end = best
        if end - start < max(8, int(FRAME_MIN * n * 3)) or end + 12 > n:
            return None
        # the picture's edge is a transition (a shadow line, then content): at least half of the
        # next 12 lines must look like picture
        if int((rng[end : end + 12] > FRAME_PICTURE_255).sum()) < 6:
            return None
        return end, float(mean[start:end].mean())

    third_h, third_w = H // 3, W // 3
    found = {
        "top": side(L[:third_h, :]),
        "bottom": side(torch.flip(L, [0])[:third_h, :]),
        "left": side(L[:, :third_w].T),
        "right": side(torch.flip(L, [1])[:, :third_w].T),
    }
    hits = {k: v for k, v in found.items() if v is not None}
    if len(hits) < 3:
        return None
    papers = [v[1] for v in hits.values()]
    if max(papers) - min(papers) > FRAME_PAPER_SPREAD_255:
        return None
    out = {k: (v[0] if v is not None else 0) for k, v in found.items()}
    if out["top"] + out["bottom"] >= H - 16 or out["left"] + out["right"] >= W - 16:
        return None
    out["paper_255"] = sum(papers) / len(papers)
    return out


def grain_mask(x_bchw: torch.Tensor, frame: dict[str, Any] | None) -> torch.Tensor:
    """1 where grain may go, 0 on a detected frame and on flat paper white. [1, 1, H, W]"""
    h, w = x_bchw.shape[-2:]
    dev = x_bchw.device
    m = torch.ones(1, 1, h, w, device=dev)
    if frame is not None:
        f = FRAME_FEATHER
        ys = torch.arange(h, device=dev, dtype=torch.float32)
        xs = torch.arange(w, device=dev, dtype=torch.float32)
        top, bot = float(frame["top"]), float(h - frame["bottom"])
        lef, rig = float(frame["left"]), float(w - frame["right"])
        ry = ((ys - top + 0.5) / f).clamp(0, 1) * ((bot - ys - 0.5) / f).clamp(0, 1)
        rx = ((xs - lef + 0.5) / f).clamp(0, 1) * ((rig - xs - 0.5) / f).clamp(0, 1)
        m = m * (ry.view(1, 1, h, 1) * rx.view(1, 1, 1, w))
    L, _, mag = chroma_parts(x_bchw[:1])
    hp = (L - box(L, 9)) * 255.0
    b = PAPER_BLOCK
    hb, wb = h // b, w // b
    if hb > 0 and wb > 0:
        def blocks(t, fn):
            tt = t[:, :, : hb * b, : wb * b].reshape(1, 1, hb, b, wb, b).permute(0, 1, 2, 4, 3, 5).reshape(1, 1, hb, wb, b * b)
            return fn(tt)
        flat = blocks(hp, lambda t: t.std(dim=-1)) < PAPER_HP_255
        bright = blocks(L, lambda t: t.mean(dim=-1)) > PAPER_L
        grey = blocks(mag * 255.0, lambda t: t.mean(dim=-1)) < PAPER_CHROMA_255
        paper = (flat & bright & grey).float()
        paper = box(paper, 3)
        paper = F.interpolate(paper, size=(hb * b, wb * b), mode="bilinear", align_corners=False)
        paper = F.pad(paper, (0, w - wb * b, 0, h - hb * b), mode="replicate")
        m = m * (1.0 - paper.clamp(0, 1))
    return m


def chroma_parts(x_bchw: torch.Tensor):
    """(luma [B,1,H,W], chroma vector = rgb - luma [B,3,H,W], its magnitude [B,1,H,W]).
    The chroma vector is luma-neutral by construction (its luma-weighted sum is 0)."""
    L = luma(x_bchw)
    cv = x_bchw[:, :3] - L
    mag = torch.sqrt((cv ** 2).sum(1, keepdim=True) + 1e-12)
    return L, cv, mag


def is_monochrome(x_bchw: torch.Tensor) -> tuple[bool, float]:
    """Frame-level: mean chroma magnitude under MONO_THRESHOLD_255. Returns (flag, mean /255)."""
    _, _, mag = chroma_parts(x_bchw)
    m = mag.mean().item() * 255.0
    return m < MONO_THRESHOLD_255, m


# -- the operation ---------------------------------------------------------------------


def add_grain(
    image: torch.Tensor,
    iso: str = ISO_DEFAULT,
    strength: float = STRENGTH_DEFAULT,
    seed: int = 0,
    chroma: str = "auto",
    match_texture: bool = True,
    film: str = FILM_DEFAULT,
    frame_detect: bool = True,
    scale_ref: int = SCALE_REF,
    work_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Add film grain to an image batch.

    image: ``[B, H, W, C]`` float 0..1, C >= 3; channels beyond three pass through.
    Each image in the batch uses ``seed + index`` and, with ``match_texture``, its
    own texture floor. Returns ``(out, stats)``, one stats dict per image.
    """
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"expected [B, H, W, C>=3], got {tuple(image.shape)}")
    if iso not in ISO_PRESETS:
        raise ValueError(f"unknown ISO preset {iso!r}; one of {ISO_NAMES}")
    if chroma not in CHROMA_MODES:
        raise ValueError(f"chroma must be one of {CHROMA_MODES}, got {chroma!r}")
    if film not in FILM_TYPES:
        raise ValueError(f"film must be one of {FILM_TYPES}, got {film!r}")
    preset = ISO_PRESETS[iso]
    orig_dtype = image.dtype
    dev = torch.device(work_device) if work_device is not None else image.device
    x_all = image.to(dev).float().permute(0, 3, 1, 2).contiguous()
    x, extra = (x_all[:, :3], x_all[:, 3:]) if x_all.shape[1] > 3 else (x_all, None)
    b, _, h, w = x.shape
    nominal = strength_to_amp(strength) / 255.0
    loud = ISO_LOUDNESS.get(iso, 1.0)
    sigma = blob_sigma(iso, h, w, scale_ref)

    outs, stats = [], []
    for i in range(b):
        xi = x[i : i + 1]
        L, cv, mag = chroma_parts(xi)
        frame = detect_frame(xi) if frame_detect else None
        inner = xi
        if frame is not None:
            inner = xi[:, :, frame["top"] : h - frame["bottom"], frame["left"] : w - frame["right"]]
        mono, mean_chroma_255 = is_monochrome(inner)
        floor_255 = texture_floor_255(inner) if match_texture else None
        factor = texture_factor(floor_255) if match_texture else 1.0
        amp = min(nominal * loud * factor, max(nominal * loud, AMP_CAP_255 / 255.0))
        if chroma == "off":
            chroma_mode, chroma_frac = "off", 0.0
        elif chroma == "on":
            chroma_mode, chroma_frac = "on", preset["chroma"]
        elif mono:
            chroma_mode, chroma_frac = "monochrome", 0.0
        else:
            chroma_mode, chroma_frac = "auto", preset["chroma"]

        if amp <= 0.0:
            out = xi
            applied = torch.zeros(1, 1, h, w, device=dev)
        else:
            fields = grain_field_exposure(L, sigma, preset["tail"], int(seed) + i, channels=2 if chroma_frac > 0 else 1)
            mask = grain_mask(xi, frame)
            weight = amount_weight(L, film) * amp
            weight = torch.minimum(weight, HEADROOM * torch.minimum(L, 1.0 - L) + 1.0 / 255.0) * mask
            applied = fields[:, :1] * weight  # luma grain, same value on R, G, B
            out = xi + applied
            if chroma_frac > 0:
                # multiplicative on the pixel's own chroma vector: exactly luma-neutral and
                # hue-preserving, proportional to saturation (std = chroma * amp at CHROMA_REF_255)
                out = out + cv * (fields[:, 1:2] * weight * chroma_frac * (255.0 / CHROMA_REF_255))
            out = out.clamp(0.0, 1.0)

        mid = ((L - MIDTONE_REF).abs() < 0.1) & (grain_mask(xi, frame) > 0.99) if amp > 0 else (L - MIDTONE_REF).abs() < 0.1
        delta_L = luma(out) - L
        stats.append(
            {
                "iso": iso,
                "strength": float(strength),
                "amp_255": amp * 255.0,
                "nominal_255": nominal * 255.0,
                "match_texture": bool(match_texture),
                "floor_255": floor_255,
                "texture_factor": factor,
                "sigma_px": sigma,
                "sigma_shadow_px": sigma * SIZE_SPREAD,
                "sigma_highlight_px": sigma / SIZE_SPREAD,
                "loudness": loud,
                "film": film,
                "frame": frame,
                "seed": int(seed) + i,
                "chroma_mode": chroma_mode,
                "chroma_frac": chroma_frac,
                "mean_chroma_255": mean_chroma_255,
                "applied_std_255": applied.std().item() * 255.0,
                "midtone_std_255": (delta_L[mid].std().item() * 255.0) if mid.any() else 0.0,
                "midtone_pct": mid.float().mean().item() * 100.0,
            }
        )
        outs.append(out)

    out = torch.cat(outs, dim=0)
    if extra is not None:
        out = torch.cat([out, extra], dim=1)
    return out.permute(0, 2, 3, 1).contiguous().to(image.device, orig_dtype), stats


def status_line(stats: list, seconds: float | None = None) -> str:
    """One-line verdict shared by the front ends. No colons or commas: Forge JSON-quotes
    any infotext value containing either."""
    s = stats[0]
    chroma_txt = {
        "monochrome": "monochrome (luma grain only)",
        "off": "colour grain off",
        "on": f"colour grain {s['chroma_frac'] * 100:.0f}%",
        "auto": f"colour grain {s['chroma_frac'] * 100:.0f}%",
    }[s["chroma_mode"]]
    parts = [
        f"grain {s['iso']} strength {s['strength']:g}",
        f"{s['amp_255']:.1f}/255 midtone (measured {s['midtone_std_255']:.1f})",
    ]
    if s.get("match_texture"):
        parts.append(f"x{s['texture_factor']:.2f} for texture floor {s['floor_255']:.2f}/255")
    parts += [
        f"blob {s['sigma_highlight_px']:.2f}-{s['sigma_shadow_px']:.2f} px",
        chroma_txt,
    ]
    if s.get("film") == "negative":
        parts.append("negative scan")
    fr = s.get("frame")
    if fr:
        parts.append(f"frame excluded {fr['top']}/{fr['bottom']}/{fr['left']}/{fr['right']} px")
    parts.append(f"seed {s['seed']}")
    if seconds is not None:
        parts.append(f"{seconds:.2f} s")
    line = " · ".join(parts)
    if len(stats) > 1:
        line += f" · batch of {len(stats)} (first shown)"
    return line
