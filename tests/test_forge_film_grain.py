"""Offline tests for the Forge Neo film grain script (hook wiring, parameters, seed, paste round trip)."""

from __future__ import annotations

import inspect
import logging
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_stubs as stubs  # noqa: E402

stubs.install_fakes()

from lib_degrid.loader import load_grain_core  # noqa: E402

fg = load_grain_core(str(stubs.REPO_ROOT))
script_mod = stubs.load_script_module(stubs.REPO_ROOT / "scripts" / "film_grain_forge.py", "film_grain_forge_under_test")

H, W = 96, 128


def pil_image(value: float = 0.4, rgb=(1.0, 1.0, 1.0)) -> Image.Image:
    arr = np.full((H, W, 3), value, np.float32) * np.array(rgb, np.float32)
    return Image.fromarray((arr * 255 + 0.5).astype(np.uint8), "RGB")


def as_tensor(image: Image.Image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)[None]


class ScriptTests(unittest.TestCase):
    UI_ORDER = ["enabled", "iso", "strength", "colour_grain"]
    DEFAULTS = dict(enabled=False, iso="ISO 200", strength=50, colour_grain="auto")

    def setUp(self):
        self.script = script_mod.FilmGrainScript()
        self.pp_cls = stubs.fake_forge_modules(str(stubs.REPO_ROOT))["modules.scripts"].PostprocessImageArgs

    def args(self, **overrides):
        values = dict(self.DEFAULTS, **overrides)
        return [values[name] for name in self.UI_ORDER]

    def processing(self, seeds=(1234, 1235)):
        p = stubs.FakeProcessing(stubs.FakeVAE(stubs.FakeDecoder(torch.zeros(1, 3, 1, H, W))))
        p.all_seeds = list(seeds)
        p.seed = seeds[0]
        return p

    def test_ui_order_matches_hooks_and_defaults(self):
        controls = self.script.ui(False)
        for hook in (self.script.process, self.script.postprocess_image_after_composite):
            params = [n for n in inspect.signature(hook).parameters if n not in ("p", "pp", "kwargs")]
            self.assertEqual(params, self.UI_ORDER, hook.__name__)
        self.assertEqual(len(controls), len(self.UI_ORDER))
        for control, name in zip(controls, self.UI_ORDER):
            self.assertEqual(control.value, self.DEFAULTS[name], name)
        self.assertEqual(controls[1].choices, list(fg.ISO_NAMES))
        self.assertEqual(controls[3].choices, list(fg.CHROMA_MODES))
        self.assertEqual(self.script.sorting_priority, 30)
        self.assertEqual(len(self.script.infotext_fields), 4)

    def test_disabled_is_a_no_op(self):
        p = self.processing()
        image = pil_image()
        pp = self.pp_cls(image, 0)
        self.script.process(p, *self.args())
        self.script.postprocess_image_after_composite(p, pp, *self.args())
        self.assertIs(pp.image, image)
        self.assertNotIn(script_mod.INFOTEXT_KEY, p.extra_generation_params)

    def test_full_run_uses_the_image_seed_and_writes_parameters(self):
        p = self.processing(seeds=(7, 8))
        image = pil_image(0.38)
        pp = self.pp_cls(image, 1)  # second image of the batch
        self.script.process(p, *self.args(enabled=True))
        self.assertEqual(p.extra_generation_params[script_mod.INFOTEXT_KEY], "iso=ISO200;strength=50;chroma=auto")
        with self.assertLogs("processing", level="INFO") as logs:
            self.script.postprocess_image_after_composite(p, pp, *self.args(enabled=True))
        self.assertEqual(len(logs.output), 1)
        self.assertIn(f"Film grain: {W}x{H} [2]", logs.output[0])
        self.assertIn("seed 8", logs.output[0])
        self.assertTrue(logs.output[0].isascii(), logs.output[0])
        result = p.extra_generation_params[script_mod.INFOTEXT_RESULT_KEY]
        self.assertTrue(result.startswith("grain ISO 200 strength 50"))
        self.assertNotIn(":", result)
        self.assertIn("monochrome, luma grain only", result)  # a grey card

        self.assertIsNot(pp.image, image)
        x_in, x_out = as_tensor(image), as_tensor(pp.image)
        d = (x_out - x_in)[..., 0] * 255.0
        self.assertAlmostEqual(d.std().item(), fg.strength_to_amp(50), delta=0.5)  # midtone grey at the bell's peak
        self.assertTrue(torch.equal(x_out[..., 0], x_out[..., 1]))  # grey stays grey
        # exactly what the core would produce with that seed
        expected, _ = fg.add_grain(x_in, "ISO 200", 50, seed=8)
        self.assertTrue(torch.allclose(x_out, expected, atol=1.0 / 255))

    def test_seed_fallbacks(self):
        p = self.processing(seeds=(5,))
        self.assertEqual(script_mod.seed_for(p, 0), 5)
        self.assertEqual(script_mod.seed_for(p, 3), 5 + 3)  # beyond all_seeds: base seed plus index

        class Bare:
            pass

        self.assertEqual(script_mod.seed_for(Bare(), 2), 2)

    def test_tolerates_odd_inputs_and_rgba(self):
        p = self.processing()
        image = pil_image(0.5, rgb=(1.0, 0.8, 0.6)).convert("RGBA")
        pp = self.pp_cls(image, 0)
        self.script.postprocess_image_after_composite(p, pp, True, "ISO 12800", "75", "maybe")
        self.assertEqual(pp.image.mode, "RGBA")
        self.assertTrue(np.array_equal(np.asarray(pp.image)[..., 3], np.asarray(image)[..., 3]))
        self.assertIn("grain ISO 200 strength 75", p.extra_generation_params[script_mod.INFOTEXT_RESULT_KEY])
        self.assertIn("colour grain 8%", p.extra_generation_params[script_mod.INFOTEXT_RESULT_KEY])

    def test_paste_fields(self):
        self.script.ui(False)
        readers = {name: pf.function for pf, name in zip(self.script.infotext_fields, ["enabled", "iso", "strength", "chroma"])}
        params = {script_mod.INFOTEXT_KEY: "iso=ISO1600;strength=35;chroma=off"}
        self.assertIs(readers["enabled"](params), True)
        self.assertEqual(readers["iso"](params), "ISO 1600")
        self.assertAlmostEqual(readers["strength"](params), 35.0)
        self.assertEqual(readers["chroma"](params), "off")
        self.assertIs(readers["enabled"]({}), False)
        self.assertIsNone(readers["iso"]({}))
        self.assertIsNone(script_mod.parse_infotext("garbage"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    unittest.main(verbosity=2)
