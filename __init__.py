"""ComfyUI-DeGrid — removes the 2px VAE pixel grid (Qwen Image / Wan 2.1 VAEs)."""

import torch
from typing_extensions import override
from comfy_api.latest import ComfyExtension, io, ui

from .degrid_core import NEGLIGIBLE_AMP, degrid, status_line

DEFAULT_THRESHOLD_255 = round(NEGLIGIBLE_AMP * 255.0, 2)  # 0.5


def _status_line(mode: str, stats: list, threshold_255: float = DEFAULT_THRESHOLD_255) -> str:
    return status_line(mode, stats, threshold=threshold_255 / 255.0)


def _console(line: str) -> None:
    """One line per run in the backend log (SwarmUI users only see the node text there)."""
    try:
        print(f"[DeGrid] {line}")
    except UnicodeEncodeError:
        print("[DeGrid] " + line.encode("ascii", "replace").decode("ascii"))


class VAEDeGrid(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VAEDeGrid",
            display_name="VAE DeGrid (Nyquist Notch)",
            category="image/postprocessing",
            description=(
                "Removes the 2px pixel grid left by the Qwen Image / Qwen Image 2.1 "
                "/ Wan 2.1 VAEs (Krea2, Qwen Image, Anima...). Wire directly after "
                "VAE Decode, before any resize, sharpening or upscaling — and before "
                "a restorer like SeedVR2, which will otherwise treat the lattice as "
                "detail worth reconstructing.\n\n"
                "Defaults are the zero-config path: leave mode on 'auto' and the node "
                "measures each image and calibrates itself. It measures the lattice "
                "itself, not just how detailed the image is, so an image that never "
                "had a grid is reported as clean and passed through untouched. After "
                "a run, the node shows the measured grid strength and which "
                "orientation dominates.\n\n"
                "The removed_grid output shows WHAT was subtracted. The artifact is "
                "only 2px, so in 'full frame' view it looks like faint gray noise — "
                "that is correct behavior, not a failure. Switch grid_view to 4x/8x "
                "zoom to see the actual lattice pattern."
            ),
            search_aliases=["degrid", "notch", "grid artifact", "qwen vae", "krea2", "pixel grid"],
            inputs=[
                io.Image.Input("image", tooltip="Wire straight from VAE Decode."),
                io.Boolean.Input(
                    "enabled", default=True,
                    tooltip="Off = the image passes through completely untouched. "
                            "Use it to A/B compare with and without degrid.",
                ),
                io.Combo.Input(
                    "mode", options=["auto", "manual"], default="auto",
                    tooltip="auto (recommended): measures the grid strength of each "
                            "image and sets the removal limit itself — nothing to tune. "
                            "manual: uses the 'limit' value below instead; use it only "
                            "if auto visibly under- or over-corrects.",
                ),
                io.Float.Input(
                    "limit", default=0.02, min=0.0, max=0.10, step=0.001,
                    tooltip="MANUAL MODE ONLY (ignored in auto). Maximum per-pixel "
                            "correction on the 0-1 scale. The VAE grid is usually "
                            "0.005-0.02, so 0.02 is a good start. Too low = grid "
                            "partially survives in contrasty areas. Too high = fine "
                            "2-3px texture (pores, fabric) gets slightly softened.",
                ),
                io.Boolean.Input(
                    "skip_when_clean", default=True,
                    tooltip="Leave the image completely untouched when no grid is "
                            "actually there. The node measures the lattice directly "
                            "(its phase is locked to the VAE's output stride, so it "
                            "survives averaging while real detail cancels), and an "
                            "image with none — anything that has been through an "
                            "upscaler or a resize — is passed through bit-for-bit. "
                            "Turn this off only to force the filter to run regardless.",
                ),
                io.Float.Input(
                    "grid_gain", default=10.0, min=1.0, max=50.0, step=1.0,
                    tooltip="Brightness amplification of the removed_grid preview ONLY "
                            "— it never affects the cleaned image. Raise it if the "
                            "preview looks like flat gray and you want to see the "
                            "removed pattern more clearly.",
                ),
                io.Combo.Input(
                    "grid_view", options=["full frame", "4x zoom", "8x zoom"],
                    default="4x zoom",
                    tooltip="Framing of the removed_grid preview. The artifact is only "
                            "2px, so 'full frame' aliases into gray noise at node-preview "
                            "size — normal, but hard to read. '4x zoom' / '8x zoom' show "
                            "a magnified center crop where the actual 2px lattice is "
                            "visible. Preview only; the cleaned image is never cropped.",
                ),
                io.Float.Input(
                    "threshold", default=DEFAULT_THRESHOLD_255, min=0.05, max=5.0, step=0.05,
                    optional=True,
                    tooltip="Lattice amplitude, in /255 units, below which an image counts "
                            "as clean (see the status line's 'grid X/255'). Native Qwen-VAE "
                            "decodes measure about 0.5-2.5, images that went through an "
                            "upscaler or a resize about 0.1-0.2. Lower it (0.3) if a model "
                            "you know is gridded reports 'none detected'.",
                ),
            ],
            outputs=[
                io.Image.Output(
                    "cleaned", display_name="image",
                    tooltip="The degridded image — same size as the input. "
                            "Send this onward to sharpening/upscaling/save.",
                ),
                io.Image.Output(
                    "removed_grid", display_name="removed_grid",
                    tooltip="Visualization of what was subtracted (amplified by "
                            "grid_gain, centered on gray). Healthy result: a uniform "
                            "fine grid/noise texture. If you can recognize faces or "
                            "fabric here, the limit is too high. Not meant for further "
                            "processing — preview only.",
                ),
            ],
        )

    @classmethod
    def execute(cls, image, enabled, mode, limit, grid_gain, grid_view,
                skip_when_clean=True, threshold=DEFAULT_THRESHOLD_255):
        if not enabled:
            return io.NodeOutput(
                image, torch.full_like(image, 0.5),
                ui=ui.PreviewText("bypassed (enabled = off)"),
            )
        threshold = float(threshold) if threshold is not None else DEFAULT_THRESHOLD_255
        cleaned, vis, stats = degrid(
            image, mode=mode, limit=limit, grid_gain=grid_gain, grid_view=grid_view,
            skip_when_clean=skip_when_clean, threshold=threshold / 255.0,
        )
        line = _status_line(mode, stats, threshold)
        _console(line)
        return io.NodeOutput(cleaned, vis, ui=ui.PreviewText(line))


class DeGridExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [VAEDeGrid]


async def comfy_entrypoint() -> DeGridExtension:
    return DeGridExtension()
