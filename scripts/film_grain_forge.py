"""Film grain for Forge Neo (sd-webui-forge-classic, neo branch).

Optional, off by default, the last stage of the pipeline: a seeded, spatially
correlated, midtone-weighted noise field on each final image. Same maths as the
ComfyUI node in this repo; ``film_grain_core.py`` at the extension root is
loaded as-is.

Integration: ``postprocess_image_after_composite``, once per final image, with
``sorting_priority = 30`` so it runs after VAE DeGrid (10) and VAE Enhance (20).
It never touches the hires-fix first pass: grain added before a resize or an
upscaler is resampled into blobs, so it has to be the last thing that happens
to the pixels. It does not depend on either other accordion being enabled.

The seed is the image's own generation seed (``p.all_seeds[index]``), so a batch
gets different grain per image and a re-run gives the same grain.
"""

from __future__ import annotations

import inspect
import time

import gradio as gr
import numpy as np
import torch
from PIL import Image

from modules import scripts
from modules.infotext_utils import PasteField
from modules.processing import logger
from modules.ui_components import InputAccordion

from lib_degrid.loader import load_grain_core

_EXTENSION_ROOT = scripts.basedir()

INFOTEXT_KEY = "Film grain"
INFOTEXT_RESULT_KEY = "Film grain result"


def _ascii(text: str) -> str:
    return text.replace("—", "-").replace("·", "|").replace("→", "->")


def _to_tensor(image: Image.Image) -> tuple[torch.Tensor, str]:
    mode = image.mode if image.mode in ("RGB", "RGBA") else "RGB"
    if image.mode != mode:
        image = image.convert(mode)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0), mode


def _to_image(t: torch.Tensor, mode: str = "RGB") -> Image.Image:
    arr = t.detach().clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    return Image.fromarray(arr, mode=mode)


def infotext(iso, strength, colour_grain, match_texture, film_type) -> str:
    """Compact ``k=v;k=v`` form; no colons or commas so Forge writes it unquoted."""
    return f"iso={str(iso).replace(' ', '')};strength={float(strength):g};chroma={colour_grain};match={int(bool(match_texture))};film={film_type}"


def parse_infotext(text: str | None) -> dict[str, str] | None:
    if not text or not isinstance(text, str):
        return None
    out: dict[str, str] = {}
    for part in text.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out or None


def _paste(name: str, cast):
    def read(params: dict):
        parsed = parse_infotext(params.get(INFOTEXT_KEY))
        if not parsed or name not in parsed:
            return None
        try:
            return cast(parsed[name])
        except (TypeError, ValueError):
            return None

    return read


def _paste_enabled(params: dict):
    return INFOTEXT_KEY in params


def _paste_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "on")


def _paste_iso(v: str) -> str:
    """'ISO200' (as written) or 'ISO 200' back to the dropdown's label."""
    digits = "".join(ch for ch in str(v) if ch.isdigit())
    return f"ISO {digits}"


def seed_for(p, index: int) -> int:
    seeds = getattr(p, "all_seeds", None)
    if isinstance(seeds, (list, tuple)) and seeds and 0 <= index < len(seeds):
        try:
            return int(seeds[index])
        except (TypeError, ValueError):
            pass
    try:
        return int(getattr(p, "seed", 0)) + int(index)
    except (TypeError, ValueError):
        return int(index)


class FilmGrainScript(scripts.Script):
    sorting_priority = 30  # after VAE DeGrid (10) and VAE Enhance (20): grain is the last thing that touches the pixels

    def title(self):
        return "Film grain"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        core = load_grain_core(_EXTENSION_ROOT)
        with InputAccordion(False, label=self.title()) as enabled:
            gr.HTML(
                "Adds film grain to each final image as the last step. <b>ISO</b> sets the grain's "
                "character (blob size, clumpiness, colour), never its amount; ISO 200 matches the grain "
                "the model draws itself when prompted for it. <b>Strength</b> sets the amount alone: "
                "1 is invisible, 50 reads as texture rather than noise, 100 is a bit noisy. Grey stays "
                "exactly grey, colour grain never adds a hue and is off on monochrome images. "
                "Judge the result at 100%."
            )
            with gr.Row():
                iso = gr.Dropdown(
                    value=core.ISO_DEFAULT, choices=list(core.ISO_NAMES), label="ISO",
                    info="grain character, not amount; 100 blends into skin as fine texture, 200 is the model's own grain, 400+ sits on the image as film",
                )
                strength = gr.Slider(
                    minimum=0, maximum=100, step=1, value=int(core.STRENGTH_DEFAULT), label="Strength",
                    info="1 = 0.3/255 invisible, 50 = 3.3/255 texture not noise, 100 = 8/255 grainy; highlights and shadows get less",
                )
                colour_grain = gr.Radio(
                    choices=list(core.CHROMA_MODES), value="auto", label="Colour grain",
                    info="auto: on saturated colours of colour images only; off: luma only; on: even on a tinted frame",
                )
            with gr.Row():
                match_texture = gr.Checkbox(
                    value=True, label="Same visible grain on textured images",
                    info="textured or already grainy images hide part of the grain, so they get more (up to 1.4x); clean renders get exactly the strength's amount; never less",
                )
                film_type = gr.Radio(
                    choices=list(core.FILM_TYPES), value=core.FILM_DEFAULT, label="Film type",
                    info="print: a print or slide, most grain in the darks; negative: a negative scan, more in the highlights; grain is coarser in the shadows either way",
                )

        self.infotext_fields = [
            PasteField(enabled, _paste_enabled),
            PasteField(iso, _paste("iso", _paste_iso)),
            PasteField(strength, _paste("strength", float)),
            PasteField(colour_grain, _paste("chroma", str)),
            PasteField(match_texture, _paste("match", _paste_bool)),
            PasteField(film_type, _paste("film", str)),
        ]
        # Positional order == the hooks' parameter order.
        return [enabled, iso, strength, colour_grain, match_texture, film_type]

    def process(self, p, enabled, iso, strength, colour_grain, match_texture, film_type, **kwargs):
        if not enabled:
            return
        p.extra_generation_params[INFOTEXT_KEY] = infotext(iso, strength, colour_grain, match_texture, film_type)

    @torch.inference_mode()
    def postprocess_image_after_composite(self, p, pp, enabled, iso, strength, colour_grain, match_texture, film_type, **kwargs):
        if not enabled:
            return
        image = getattr(pp, "image", None)
        if not isinstance(image, Image.Image):
            return
        core = load_grain_core(_EXTENSION_ROOT)
        iso = str(iso) if str(iso) in core.ISO_NAMES else core.ISO_DEFAULT
        colour_grain = str(colour_grain) if str(colour_grain) in core.CHROMA_MODES else "auto"
        film_type = str(film_type) if str(film_type) in core.FILM_TYPES else core.FILM_DEFAULT
        index = int(getattr(pp, "index", 0) or 0)
        seed = seed_for(p, index)

        x, mode = _to_tensor(image)
        work = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        t0 = time.perf_counter()
        out, stats = core.add_grain(x, iso=iso, strength=float(strength), seed=seed, chroma=colour_grain, match_texture=bool(match_texture), film=film_type, work_device=work)
        line = core.status_line(stats, seconds=time.perf_counter() - t0)

        logger.info(_ascii(f"Film grain: {image.width}x{image.height} [{index + 1}] | {line}"))
        p.extra_generation_params[INFOTEXT_RESULT_KEY] = _ascii(line)
        pp.image = _to_image(out, mode)


_UI_PARAMS = ["enabled", "iso", "strength", "colour_grain", "match_texture", "film_type"]
for _hook in (FilmGrainScript.process, FilmGrainScript.postprocess_image_after_composite):
    _params = [n for n in inspect.signature(_hook).parameters if n not in ("self", "p", "pp", "kwargs")]
    assert _params == _UI_PARAMS, (_hook.__name__, _params)
