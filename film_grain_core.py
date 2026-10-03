"""Pure-torch core for optional film grain, the last stage of the pipeline.

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

Why it looks like film and not like sensor noise:

* **Correlated.** White noise has zero neighbour correlation; grain clumps
  span 1-3 px. A Gaussian-blurred field at the preset sigma, renormalised,
  gives a lag-one autocorrelation of ~0.3-0.6 and reads as grain. The sigma is
  scaled with the image's long side (reference 1536 px) because grain lives on
  the frame, not on the pixel.
* **Midtone-weighted.** Grain is strongest in midtones and fades toward black
  and, faster, toward white, which is also where the eye's threshold is
  highest. The weight is a bell in luma, 1 at its peak.
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
AMP_MIN_255, AMP_MAX_255, AMP_GAMMA = 0.3, 8.0, 1.35
CHROMA_REF_255 = 40.0  # saturation at which the colour grain's std equals chroma * luma amplitude
MONO_THRESHOLD_255 = 8.0  # frame mean chroma magnitude below this = monochrome (a Krea 2 "B&W" decode carries ~5.6 of tint; colour photos 20+)
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
    """Midtone bell in luma 0..1, equal to 1 at its peak (L ~ 0.38), 0 at black and white."""
    L = L.clamp(0.0, 1.0)
    return (L ** _BELL_A) * ((1.0 - L) ** _BELL_B) / _bell_peak


def blob_sigma(iso: str, height: int, width: int, scale_ref: int = SCALE_REF) -> float:
    return ISO_PRESETS[iso]["sigma"] * max(height, width) / float(scale_ref)


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
    scale_ref: int = SCALE_REF,
    work_device: torch.device | str | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Add film grain to an image batch.

    image: ``[B, H, W, C]`` float 0..1, C >= 3; channels beyond three pass through.
    Each image in the batch uses ``seed + index``. Returns ``(out, stats)``, one
    stats dict per image.
    """
    if image.ndim != 4 or image.shape[-1] < 3:
        raise ValueError(f"expected [B, H, W, C>=3], got {tuple(image.shape)}")
    if iso not in ISO_PRESETS:
        raise ValueError(f"unknown ISO preset {iso!r}; one of {ISO_NAMES}")
    if chroma not in CHROMA_MODES:
        raise ValueError(f"chroma must be one of {CHROMA_MODES}, got {chroma!r}")
    preset = ISO_PRESETS[iso]
    orig_dtype = image.dtype
    dev = torch.device(work_device) if work_device is not None else image.device
    x_all = image.to(dev).float().permute(0, 3, 1, 2).contiguous()
    x, extra = (x_all[:, :3], x_all[:, 3:]) if x_all.shape[1] > 3 else (x_all, None)
    b, _, h, w = x.shape
    amp = strength_to_amp(strength) / 255.0
    sigma = blob_sigma(iso, h, w, scale_ref)

    outs, stats = [], []
    for i in range(b):
        xi = x[i : i + 1]
        L, cv, mag = chroma_parts(xi)
        mono, mean_chroma_255 = is_monochrome(xi)
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
            fields = grain_field(h, w, sigma, preset["tail"], int(seed) + i, dev, channels=2 if chroma_frac > 0 else 1)
            weight = luminance_weight(L) * amp
            applied = fields[:, :1] * weight  # luma grain, same value on R, G, B
            out = xi + applied
            if chroma_frac > 0:
                # multiplicative on the pixel's own chroma vector: exactly luma-neutral and
                # hue-preserving, proportional to saturation (std = chroma * amp at CHROMA_REF_255)
                out = out + cv * (fields[:, 1:2] * weight * chroma_frac * (255.0 / CHROMA_REF_255))
            out = out.clamp(0.0, 1.0)

        mid = (luminance_weight(L) > 0.8)
        delta_L = luma(out) - L
        stats.append(
            {
                "iso": iso,
                "strength": float(strength),
                "amp_255": amp * 255.0,
                "sigma_px": sigma,
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
    """One-line verdict shared by the front ends. No colons (Forge quotes them)."""
    s = stats[0]
    chroma_txt = {
        "monochrome": "monochrome, luma grain only",
        "off": "colour grain off",
        "on": f"colour grain {s['chroma_frac'] * 100:.0f}%",
        "auto": f"colour grain {s['chroma_frac'] * 100:.0f}%",
    }[s["chroma_mode"]]
    parts = [
        f"grain {s['iso']} strength {s['strength']:g}",
        f"{s['amp_255']:.1f}/255 midtone (measured {s['midtone_std_255']:.1f})",
        f"blob {s['sigma_px']:.2f} px",
        chroma_txt,
        f"seed {s['seed']}",
    ]
    if seconds is not None:
        parts.append(f"{seconds:.2f} s")
    line = " · ".join(parts)
    if len(stats) > 1:
        line += f" · batch of {len(stats)} (first shown)"
    return line
