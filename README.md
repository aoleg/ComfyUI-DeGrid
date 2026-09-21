# ComfyUI-DeGrid

**VAE DeGrid (Nyquist Notch)** — a ComfyUI node and a Forge Neo extension that remove the 2px pixel grid that the
Qwen Image VAE (and, to a lesser extent, the Wan 2.1 VAE) leaves across decoded
images. Affects Krea2, Qwen Image, **Qwen Image 2.1**, Anima and anything else
built on those VAEs.

The artifact is easy to miss at 100% zoom, but it gets amplified by any
sharpening or upscaling applied afterwards. This node erases it exactly, with
auto-calibration so there is nothing to tune.

It reads worst in flat, dark areas — a lattice of fixed amplitude has the most
local contrast to sit against where the image itself is smooth.

## Front ends

One repo, one set of maths (`degrid_core.py`), several hosts:

| Host | Where it lives | Status |
|---|---|---|
| ComfyUI | `__init__.py` at the repo root (node **VAE DeGrid (Nyquist Notch)**) | shipped |
| Forge Neo | `scripts/degrid_forge.py` + `lib_degrid/` (accordion **VAE DeGrid**) | shipped, see [Forge Neo](#forge-neo) |
| SwarmUI | planned | — |

Cloning the repo into either host's extension folder is enough; each host only
loads its own entry point and ignores the others.

## Install (ComfyUI)

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
   grid 2.10/255 (checker) — removed (limit 0.019 auto) · edges protected 1.2%
   ```

   That readout is your confirmation it worked — you don't need to pixel-peep.
   On an image that has no grid you get this instead, and nothing is changed:

   ```
   grid 0.16/255 — none detected, passed through untouched · edges protected 0.0%
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
- **`partially removed, X/255 left`** — the cap was lower than the grid's own
  amplitude, so part of the lattice survived. The node measures what is left
  in the cleaned image rather than assuming the notch got it all. Raise the
  limit (or go back to `auto`).
- **`edges protected N%`** — percentage of pixels where the correction hit the
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

## Forge Neo

The same filter as a [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic/tree/neo)
extension, for the Wan-VAE models Forge Neo runs (Krea 2, Qwen Image, Wan, Anima).

```
cd <forge-neo>/extensions
git clone https://github.com/aoleg/ComfyUI-DeGrid
```

Restart the webui. A **VAE DeGrid** accordion appears on the txt2img and img2img
tabs, off by default. Tick it and generate; the console prints one line per VAE
decode, e.g.

```
DeGrid: 1536x1536 | grid 2.54/255 (V-stripe) - removed (limit 0.010 auto) | edges protected 2.9%
```

and the same verdict is written into the image's parameters as `DeGrid result`.

### Where it runs, and why that matters

In Forge the filter is not an image post-process. The extension replaces the
checkpoint's VAE, for the duration of each sampling pass, with a wrapper that
runs the notch inside every decode. Two things follow:

- **Hires fix is handled correctly.** Forge decodes the first pass and hands the
  pixels straight to the hires upscaler with no extension hook in between. An
  image-level hook would leave the grid under the upscaler, exactly the case the
  ComfyUI section above warns about. Decode-level filtering cleans the first pass
  *before* the upscaler sees it, then the final decode again. Latent-space hires
  upscalers only decode once, at the end, which is also covered.
- **Nothing else changes.** Encoding (img2img, inpainting, hires re-encode) is the
  checkpoint's own, untouched. The wrapper shares the loaded VAE weights, so
  there is no second model in memory, and tiled decoding and the out-of-memory
  fallback go through the same filter. Any extension that installs its own VAE
  subclass (e.g. an upscaling decoder) still works: its decode runs first and
  the notch runs on its output, where it will usually find nothing to remove.

A measured 2px lattice from the real Forge Neo Krea 2 decodes used to validate
this: 1.4 to 2.5/255 before, 0.05 to 0.2/255 after. The same images generated
with a Flux VAE measure 0.07 to 0.09/255 and are passed through untouched.

### Controls

| Control | Default | What it does |
|---|---|---|
| Mode | `auto` | Same as the node's `mode`: `auto` measures each image and sets the removal limit; `manual` uses the slider. |
| Limit (manual mode) | 0.02 | Same as the node's `limit`; ignored in `auto`. |
| Skip when clean | on | Same as the node's `skip_when_clean`. |
| Clean threshold (/255) | 0.5 | New in Forge: the lattice amplitude below which an image counts as clean and is left alone. Forge Neo Krea 2 decodes have measured as low as 0.47/255, just under the default, so lower it (0.3) if the console reports `none detected` on a model you know is gridded. Raise it to be more conservative. |
| Show removed grid | off | Adds a magnified centre crop of the subtracted component to the results, one per final image, so you can see the lattice without pixel-peeping. |
| Preview zoom, Preview gain | `8x`, 10 | Framing and brightness of that preview only; the cleaned image is never affected. |

Settings are saved to the image parameters as `DeGrid: mode=auto;skip=1;threshold=0.5`
(plus `limit=` in manual mode) and restore from the paste button.

### Notes

- **Order relative to other decode-side extensions.** DeGrid installs its VAE
  wrapper early (`sorting_priority = 10`), so an extension that swaps the VAE
  later, such as Neo VAE Utils, replaces the wrapper rather than being wrapped by
  it. Its decode then runs without DeGrid, which is the right outcome for an
  upscaling decoder that already averages the grid away.
- **Batches.** Each image in a batch is measured and limited on its own. The
  console line carries `[i/n]`; the parameters carry the first image's verdict.
- **Interrupted runs.** The swap is scoped to one sampling pass by Forge itself
  and is additionally undone at the end of every run and at the start of the
  next, so an interrupted generation cannot leave the wrapper installed.
- **Tests.** `python -m unittest discover -s tests -v` from the repo root, with
  any venv that has torch, numpy and Pillow. No Forge checkout or checkpoint is
  needed; the hook wiring, the decode topology (direct, tiled, OOM fallback),
  the hires double pass and the parameter round trip are all covered offline.

## Swapping the decoder instead

A recurring suggestion is to avoid the grid by not using the Qwen decoder at
all. Two versions of that idea, measured:

**A different VAE family (Flux, Flux 2) does not fit.** Krea 2, Qwen Image and
Wan read and write the 16-channel Wan latent space. Every step that samples with
one of those models must encode with the Qwen/Wan encoder and decode with a
Wan-latent decoder, so a Flux VAE can only ever be used for a pure
encode-decode round trip with no sampling in between. That round trip removes
the 2px lattice because the pattern cannot pass the bottleneck, but it also
re-synthesizes every other 2-4px detail through a different decoder. The notch
removes only the 2px band and adds nothing.

**A different decoder for the same latent space does work.** The
[Wan2.1 VAE upscale2x](https://huggingface.co/spacepxl/Wan2.1-VAE-upscale2x)
checkpoint is a decoder-only finetune whose last conv emits 12 channels, pixel-
shuffled into a 2x image (available in Forge Neo through
[Neo-VAE-Utils](https://github.com/aoleg/Neo-VAE-Utils)). Measured with the real
weights, on three 1536px inputs, lattice peak-to-peak in /255:

| input | input | 2x decode | 1x gaussian | 1x bilinear | 1x area | 1x lanczos |
|---|---|---|---|---|---|---|
| Krea 2, gridded | 2.54 | 1.30 | 0.03 | 0.04 | 0.11 | 0.06 |
| Krea 2, degridded | 0.32 | 1.22 | 0.02 | 0.03 | 0.07 | 0.05 |
| Z-Image (Flux VAE) | 0.07 | 1.06 | 0.03 | 0.04 | 0.08 | 0.06 |

- **The lattice is made by the decoder, not carried in the latent.** A grid-free
  input produces the same 2x lattice as a gridded one, and a gridded input's
  lattice does not survive the encoder. Degridding before a re-encode only
  protects whatever reads the pixels in between (an upscaler, a restorer); the
  next Qwen decode always adds a fresh grid, which is why the Forge extension
  filters every decode rather than the final image.
- **At 2x the upscale decoder is gridded too**, at about half the stock
  amplitude, vertical stripes dominant, phase-locked across the whole frame.
  Use DeGrid after it. In Forge the two extensions compose on their own.
- **Downscaled back to 1x it is clean.** Every filter leaves at most 0.11/255
  (area, a 2-tap box); gaussian and bilinear leave 0.02-0.04, and DeGrid
  reports `none detected` and passes the image through. The round trip
  reconstructs Wan-native content at 39-41 dB PSNR.

So for a plain decode the two routes reach the same place. The notch costs a
few milliseconds and covers every decode path; the upscale decoder is a full
decode at four times the pixel count, but also gives real 2x output when that
is what you want.

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

### 2026-09-21

- **Forge Neo extension.** The same node, as a `VAE DeGrid` accordion in
  [Forge Neo](#forge-neo). It filters inside the VAE decode rather than after it,
  so the hires-fix first pass is cleaned before the upscaler sees it. Adds a
  `Clean threshold` control (the node keeps its fixed 0.5/255).
- `degrid_core.degrid()` takes an optional `threshold` (default unchanged), and
  the status-line text moved into `degrid_core.status_line()` so every front end
  describes a result in the same words. Cleaned images are identical to before.
- The status line now says **partially removed** when the clamp limit was too
  low to subtract the whole grid, with the residual amplitude, instead of
  claiming "removed" for whatever the detector saw. `stats` gains
  `residual_255`. The `edges protected` figure lost its colon so Forge does not
  quote the value in the parameters.
- README: measured the Wan2.1 VAE upscale2x decoder as an alternative to
  degridding (see [Swapping the decoder instead](#swapping-the-decoder-instead)):
  gridded at 2x, clean after its own downscale to 1x; the lattice comes from
  the decoder, not the latent.

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
