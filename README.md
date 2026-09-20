# ComfyUI-DeGrid

One node: **VAE DeGrid (Nyquist Notch)** — removes the 2px pixel grid that the
Qwen Image VAE (and, to a lesser extent, the Wan 2.1 VAE) leaves across decoded
images. Affects Krea2, Qwen Image, **Qwen Image 2.1**, Anima and anything else
built on those VAEs.

The artifact is easy to miss at 100% zoom, but it gets amplified by any
sharpening or upscaling applied afterwards. This node erases it exactly, with
auto-calibration so there is nothing to tune.

It reads worst in flat, dark areas — a lattice of fixed amplitude has the most
local contrast to sit against where the image itself is smooth.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/lunaaispace-eng/ComfyUI-DeGrid
```

No dependencies beyond torch (no OpenGL/GLFW — works headless). Restart ComfyUI
and search for **degrid**.

> **Do not install this alongside [ComfyUI-SaveSimple](https://github.com/lunaaispace-eng/ComfyUI-SaveSimple).**
> That pack bundles the same node under the same id (`VAEDeGrid`), so having both
> installed is a node-id collision. Pick one: this repo if you only want the grid
> fix, SaveSimple if you already use the rest of that suite.

## Quick start

1. Wire **VAE Decode → VAE DeGrid → (everything else)**. It must sit *before*
   any sharpening, deconvolution, or upscaling — those amplify the grid, so
   remove it first.
2. Leave the defaults. `auto` mode measures each image and calibrates itself.
3. Run once. The node displays a status line, e.g.:

   ```
   grid 2.10/255 (checker) — removed (limit 0.019 auto) · edges protected: 1.2%
   ```

   That readout is your confirmation it worked — you don't need to pixel-peep.
   On an image that has no grid you get this instead, and nothing is changed:

   ```
   grid 0.16/255 — none detected, passed through untouched · edges protected: 0.0%
   ```

> **Order matters more than it looks.** Put this straight after `VAE Decode`,
> *before* any resize, and before a diffusion restorer/upscaler such as SeedVR2.
> A half-scale resize will hide the lattice from your eyes, but the restorer has
> already been handed it — and a restorer reads a regular lattice as detail worth
> reconstructing. Degridding after the upscaler is too late; the node will tell
> you there is nothing left to remove.

## Settings

| Widget | Default | What it does |
|---|---|---|
| `enabled` | on | Off = the image passes through completely untouched. Flip it for a quick A/B comparison. |
| `mode` | `auto` | **auto (recommended):** measures the grid strength per image and sets the removal limit itself — nothing to tune, adapts to different VAEs and content. **manual:** uses the `limit` widget instead. Only switch if auto visibly under- or over-corrects. |
| `limit` | 0.02 | **Manual mode only** (ignored in auto). Maximum per-pixel correction on the 0–1 scale. The VAE grid is usually 0.005–0.02. Too low → grid partially survives in contrasty areas. Too high → fine 2–3px texture (skin pores, fabric weave) gets slightly softened. |
| `skip_when_clean` | on | Leave the image completely untouched when no grid is actually there. The node measures the lattice directly, so anything that has already been through an upscaler or a resize is passed through **bit-for-bit**. Turn it off only to force the filter to run regardless. |
| `grid_gain` | 10 | Brightness amplification of the `removed_grid` preview **only** — never affects the cleaned image. Raise it if the preview looks like flat gray. |
| `grid_view` | `4x zoom` | Framing of the `removed_grid` preview. `full frame` shows the whole image (reads as gray noise at preview size — see below). `4x zoom` / `8x zoom` show a magnified center crop where the actual 2px lattice is visible. Preview only; the cleaned image is never cropped. |

### Status line

After each run the node shows what it measured:

- **`grid X/255 (orientation)`** — the measured lattice amplitude, peak-to-peak.
  The raw Qwen-VAE grid is typically 1–5/255 (measured on Qwen Image 2.1
  decodes: 1.6–2.0/255). Below 0.5/255 the node reports **none detected** and
  passes the image through untouched. The orientation in brackets — `checker`,
  `V-stripe` or `H-stripe` — is whichever component dominates.

  This is a *phase-locked* measurement, not a percentile of the filter's own
  output. The grid's phase is tied to the VAE's output stride, so it is constant
  across the frame: averaging the four `(y%2, x%2)` sublattices keeps the grid
  while genuine detail cancels. That matters, because "how much high-frequency
  content is in this image" is not the same question as "is there a grid", and
  answering the first one gets it backwards on a busy image.
- **`limit N (auto|manual)`** — the correction cap that was applied.
- **`edges protected: N%`** — percentage of pixels where the correction hit the
  cap. Those are real edges/detail being passed through unsoftened. A few
  percent is normal; a very high number means lots of legitimate
  high-frequency content (or a manual limit set too low).

## Reading the removed_grid preview

**"It's just gray noise — is it even doing anything?"** Yes. That is exactly
what success looks like, and here is why: the artifact is a 2-pixel pattern.
A node preview shows a 1728px image at a few hundred pixels wide, so a 2px
lattice is far below what the thumbnail can render — it aliases into uniform
gray "noise". The information is real; the zoom level just can't show it.

Two ways to actually see it:

- Set `grid_view` to `4x zoom` or `8x zoom` (default is 4x): the preview
  becomes a magnified center crop and the regular lattice pattern is plainly
  visible.
- Or open the preview image at 100%+ zoom.

What to look for:

| removed_grid shows | Meaning |
|---|---|
| Uniform fine grid / speckle, brighter over textured areas | Working correctly |
| Nearly flat gray | Little or no grid in this image (check the status line — likely `none detected`) |
| Recognizable faces, fabric, edges | Limit too high — switch to manual and lower `limit` |

A faint silhouette of the subject is normal (the artifact is slightly stronger
over detailed areas). Recognizable *detail* is not.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Grid still visible in contrasty areas after filtering | `mode: manual`, raise `limit` toward 0.03–0.04 |
| Fine texture (pores, weave) looks softened | `mode: manual`, lower `limit` toward 0.01 |
| Status says `none detected` but you see a grid | The grid may be coming from a later node (sharpener, upscaler) — this node only fixes what the VAE decode produced. Check the chain order. If the image reached this node via a resize, the resize has already scrambled the lattice; move the node earlier. |
| You want it to filter anyway | Turn `skip_when_clean` off. The status line then reads `none detected, filtered anyway`. |
| removed_grid looks like flat gray | Raise `grid_gain`, or the image simply has no grid |

## How it works

A separable Nyquist notch: 9-tap alternating-sign binomial kernel
(1D response sin⁸(ω/2)), combined as `center − Bx − By + Bxy`, which factors
into `(1 − sin⁸(ωx/2))(1 − sin⁸(ωy/2))`. That is an exact zero for any
2px-period pattern — vertical stripes, horizontal stripes, or checkerboard —
and exact unity at DC with an 8th-order flat zero, so gradients pass through
without banding.

The correction is then amplitude-clamped before subtraction, so strong real
edges and legitimate fine texture pass through unsoftened — only the
low-amplitude artifact band is removed. In `auto` mode the clamp limit is
estimated per image from a robust percentile of the extracted grid component,
so it adapts to different VAEs, LoRA stacks, and content automatically.

Separately from the clamp, the node decides *whether there is a grid at all* by
measuring the phase-locked lattice (see **Status line** above). Measured across
Qwen Image 2.1 outputs, that separates cleanly by about 10×: native VAE decodes
read 1.6–2.0/255, the same images after an upscaler read 0.10–0.20/255, and a
pure-noise control reads 0.05/255. Below the 0.5/255 threshold the filter is
skipped entirely rather than run at a small setting, because the notch is cheap
but not free — on a clean, detailed image it still shaves roughly 1/255 of real
high-frequency detail.

Based on the GLSL notch-filter approach shared by
[u/Haiku-575 on r/StableDiffusion](https://www.reddit.com/r/StableDiffusion/comments/1umwhq7/2px_pixel_grid_on_krea2_from_vae_and_how_to/),
reimplemented in pure PyTorch with a narrower 9-tap kernel, amplitude limiting,
and per-image auto-calibration.

## Changelog

### 2026-09-20

**In short:** the filter was fine — the thing deciding *when to run it* was broken.

The node used to report how much fine detail an image had and call that "the grid".
Those are not the same thing, so it got it backwards: on a real test set it called
the cleanest image the most gridded one. And because it never really knew whether a
grid was there, it filtered everything, always — quietly scraping a little genuine
texture off images that had no grid at all.

It now measures the grid itself. A VAE grid lands on the same pixel positions across
the whole frame, so averaging those positions keeps the grid while ordinary detail
cancels out. Gridded images measure about 1.6–2.0/255, grid-free ones about 0.1 —
a wide, unambiguous gap. If there is no grid, the image is now passed through
**completely untouched**, which matters most for anything that has already been
through an upscaler or a resize.

**The maths that removes the grid did not change.** On an image that really has a
grid you get exactly the same result as before.

The detail, for anyone who wants it:

- **Qwen Image 2.1 confirmed affected.** It ships a genuinely different VAE —
  `modelspec.architecture: qwen_image_2.1_vae`, a 64-channel latent and four
  spatial upsample stages, against 16 channels and three for the Qwen-Image /
  Wan 2.1 VAE — so it compresses considerably harder. The artifact is
  nonetheless the **same 2px lattice**, measured on native decodes at
  1.6–2.0/255 with checkerboard and both stripe orientations at comparable
  strength. That is expected: the period of this artifact is set by the stride
  of the *final* upsample stage, which is still 2, not by how deep the VAE is or
  how wide its latent is. There is no 16px "latent grid" to chase even though
  the VAE now compresses 16× — a phase-fold sweep over periods 5–12 shows only
  the even-period harmonics of the 2px component, and an exact-bin comb test
  reads flat at p=4/8/16/32.
- **Fixed a backwards grid detector.** The reported `grid ≈ X/255` was the 75th
  percentile of the filter's own correction, which measures how *detailed* an
  image is rather than how gridded — on real files it read higher on the
  cleanest image than on the most gridded one. It is now a phase-locked lattice
  measurement. The notch, the clamp and the auto limit are unchanged, so results
  on images that do have a grid are identical to before.
- **New `skip_when_clean` widget (default on).** An image with no lattice is now
  passed through bit-for-bit instead of being quietly filtered. Previously every
  grid-free image lost ~0.7–1.1/255 mean (up to 4.5/255 peak) of genuine
  high-frequency detail for nothing.
- Status line now names the dominant orientation and no longer claims to have
  removed a grid that was not there.

### 2026-07-04

- Initial release.

## License

Apache-2.0
