"""Offline tests for film_emulation_core.py.

    python -m unittest discover -s tests -v

Needs torch only. The statistics (lag-one autocorrelation, kurtosis, axis ratio)
are the ones the grain was calibrated with; the thresholds here are loose enough
to survive a reseed and tight enough to catch a broken blur, bell or gate.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forge_stubs as stubs  # noqa: E402

from lib_degrid.loader import load_film_core  # noqa: E402

fg = load_film_core(str(stubs.REPO_ROOT))


def flat(value: float, h: int = 512, w: int = 512, rgb=None) -> torch.Tensor:
    x = torch.full((1, h, w, 3), float(value))
    if rgb is not None:
        x = x * torch.tensor(rgb).view(1, 1, 1, 3)
    return x


def luma_np(x_bhwc: torch.Tensor) -> np.ndarray:
    return (x_bhwc[0, ..., :3].numpy() @ np.array([0.299, 0.587, 0.114], np.float32)) * 255.0


def acf(h: np.ndarray, dy: int, dx: int) -> float:
    a = h - h.mean(); n = a.std() ** 2 + 1e-9; H, W = a.shape
    return float((a[dy:, dx:] * a[:H - dy, :W - dx]).mean() / n)


def kurt(h: np.ndarray) -> float:
    a = h - h.mean()
    return float((a ** 4).mean() / (a ** 2).mean() ** 2)


def axis_ratio(h: np.ndarray) -> float:
    a = h - h.mean(); Fq = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    H, W = Fq.shape; yy, xx = np.mgrid[-H // 2:H - H // 2, -W // 2:W - W // 2]
    r = np.hypot(yy / H, xx / W); ang = np.degrees(np.arctan2(yy / H, xx / W)) % 180
    band = (r > 0.15) & (r < 0.5)
    ax = band & ((ang < 10) | (ang > 170) | (np.abs(ang - 90) < 10)); di = band & ((np.abs(ang - 45) < 10) | (np.abs(ang - 135) < 10))
    return float(Fq[ax].mean() / Fq[di].mean())


class MappingTests(unittest.TestCase):
    def test_strength_anchors(self):
        self.assertAlmostEqual(fg.strength_to_amp(1), 0.3, delta=0.02)
        self.assertAlmostEqual(fg.strength_to_amp(50), 3.3, delta=0.15)
        self.assertAlmostEqual(fg.strength_to_amp(100), 8.0, delta=0.01)
        self.assertEqual(fg.strength_to_amp(0), 0.0)
        self.assertEqual(fg.strength_to_amp(150), fg.strength_to_amp(100))
        # monotone
        vals = [fg.strength_to_amp(s) for s in range(0, 101)]
        self.assertTrue(all(b > a for a, b in zip(vals, vals[1:])))

    def test_luminance_bell(self):
        L = torch.linspace(0, 1, 1001)
        w = fg.luminance_weight(L)
        self.assertAlmostEqual(w.max().item(), 1.0, places=3)
        self.assertEqual(w[0].item(), 0.0)
        self.assertEqual(w[-1].item(), 0.0)
        peak = L[w.argmax()].item()
        self.assertTrue(0.3 < peak < 0.45, peak)
        self.assertGreater(w[100].item(), w[950].item())  # fades faster toward white than toward black

    def test_presets_and_scale(self):
        self.assertEqual(tuple(fg.ISO_PRESETS), fg.ISO_NAMES)
        sig = [fg.ISO_PRESETS[n]["sigma"] for n in fg.ISO_NAMES]
        self.assertTrue(all(b > a for a, b in zip(sig, sig[1:])))
        s400 = fg.ISO_PRESETS["ISO 400"]["sigma"]
        self.assertAlmostEqual(fg.blob_sigma("ISO 400", 1536, 1024), s400)
        # square-root scaling: twice the long side gives sqrt(2) times the blob, half gives 1/sqrt(2)
        self.assertAlmostEqual(fg.blob_sigma("ISO 400", 3072, 2048), s400 * 2 ** 0.5)
        self.assertAlmostEqual(fg.blob_sigma("ISO 400", 768, 512), s400 / 2 ** 0.5)
        # the grain-2 pair: 1728 vs 1152 long side differ by 1.22x instead of 1.5x
        r = fg.blob_sigma("ISO 200", 1728, 1280) / fg.blob_sigma("ISO 200", 1152, 896)
        self.assertAlmostEqual(r, 1.5 ** 0.5, places=6)


class FieldTests(unittest.TestCase):
    def test_unit_std_and_statistics_per_iso(self):
        prev_acf = -1.0
        for name in fg.ISO_NAMES:
            p = fg.ISO_PRESETS[name]
            f = fg.grain_field(512, 512, p["sigma"], p["tail"], seed=3, device="cpu")[0, 0].numpy()
            self.assertAlmostEqual(float(f.std()), 1.0, places=3, msg=name)
            self.assertAlmostEqual(float(f.mean()), 0.0, places=3, msg=name)
            a = acf(f, 1, 0)
            self.assertGreater(a, prev_acf - 0.02, name)  # correlation rises with ISO
            prev_acf = a
            self.assertTrue(0.85 < axis_ratio(f) < 1.15, (name, axis_ratio(f)))  # isotropic
            k = kurt(f)
            if name == "ISO 100":
                self.assertTrue(2.8 < k < 3.2, (name, k))
            if name == "ISO 3200":
                self.assertGreater(k, 3.5, (name, k))
        f100 = fg.grain_field(512, 512, fg.ISO_PRESETS["ISO 100"]["sigma"], 0.0, 3, "cpu")[0, 0].numpy()
        f3200 = fg.grain_field(512, 512, fg.ISO_PRESETS["ISO 3200"]["sigma"], 0.0, 3, "cpu")[0, 0].numpy()
        self.assertLess(acf(f100, 1, 0), 0.5)
        self.assertGreater(acf(f3200, 1, 0), 0.7)

    def test_seed_reproducible_and_distinct(self):
        a = fg.grain_field(64, 64, 0.8, 0.1, 7, "cpu")
        b = fg.grain_field(64, 64, 0.8, 0.1, 7, "cpu")
        c = fg.grain_field(64, 64, 0.8, 0.1, 8, "cpu")
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.allclose(a, c))


class GrainTests(unittest.TestCase):
    def test_midtone_amplitude_matches_strength(self):
        x = flat(fg.MIDTONE_REF)  # where strength is defined
        for s, expect in ((1, 0.3), (50, 3.3), (100, 8.0)):
            out, st = fg.add_grain(x, "ISO 200", s, seed=1, match_texture=False)  # loudness 1
            d = luma_np(out) - luma_np(x)
            self.assertAlmostEqual(float(d.std()), expect, delta=expect * 0.08 + 0.05, msg=s)
            self.assertAlmostEqual(st[0]["midtone_std_255"], float(d.std()), delta=0.05)
            self.assertEqual(st[0]["chroma_mode"], "monochrome")  # a grey card is monochrome
        self.assertAlmostEqual(st[0]["sigma_px"], fg.ISO_PRESETS["ISO 200"]["sigma"] * (512 / fg.SCALE_REF) ** 0.5)
        self.assertIsNone(st[0]["floor_255"])
        self.assertEqual(st[0]["texture_factor"], 1.0)

    def test_black_white_and_headroom(self):
        for v in (0.0, 1.0):  # pure black and pure white: nothing
            out, _ = fg.add_grain(flat(v), "ISO 400", 100, seed=1)
            self.assertEqual(float((out - flat(v)).abs().max()), 0.0, v)
        # near black: grain, but never more than the headroom (half the distance to black + 1/255)
        for v in (0.01, 0.02, 0.04):
            out, _ = fg.add_grain(flat(v), "ISO 3200", 100, seed=1, match_texture=False)
            d = luma_np(out) - luma_np(flat(v))
            self.assertLess(float(d.std()), (fg.HEADROOM * v * 255 + 1.0) * 1.05, v)
            self.assertGreater(float(d.std()), 0.2, v)
        # near white: the print curve is 0 from 0.95
        out, _ = fg.add_grain(flat(0.97), "ISO 400", 100, seed=1, match_texture=False)
        self.assertLess(float((luma_np(out) - luma_np(flat(0.97))).std()), 0.3)

    def test_grey_stays_exactly_grey(self):
        x = flat(0.4)
        for iso in fg.ISO_NAMES:
            out, st = fg.add_grain(x, iso, 100, seed=5)
            self.assertTrue(torch.equal(out[..., 0], out[..., 1]), iso)
            self.assertTrue(torch.equal(out[..., 1], out[..., 2]), iso)
            self.assertEqual(st[0]["chroma_mode"], "monochrome")
        # forced on: still grey, because the per-pixel gate sees no saturation
        out, st = fg.add_grain(x, "ISO 3200", 100, seed=5, chroma="on")
        self.assertTrue(torch.equal(out[..., 0], out[..., 2]))
        self.assertEqual(st[0]["chroma_mode"], "on")

    def test_sepia_keeps_its_hue(self):
        x = flat(0.5, rgb=(1.0, 0.85, 0.65))  # saturated enough to pass the gate
        out, st = fg.add_grain(x, "ISO 3200", 100, seed=2)
        self.assertEqual(st[0]["chroma_mode"], "auto")
        self.assertGreater(st[0]["chroma_frac"], 0.3)
        # chroma vector direction unchanged where it is defined: angle in the (R-L, B-L) plane
        def angle(t):
            L = (t[..., :3] * torch.tensor([0.299, 0.587, 0.114])).sum(-1, keepdim=True)
            cv = t[..., :3] - L
            return torch.atan2(cv[..., 2], cv[..., 0])
        da = (angle(out) - angle(x)).abs().rad2deg()
        self.assertLess(da.quantile(0.99).item(), 1.0)
        # ...and the colour grain is actually there: per-channel deltas are not identical
        d = out - x
        self.assertFalse(torch.allclose(d[..., 0], d[..., 2], atol=1e-4))
        # off: identical deltas on every channel
        out2, st2 = fg.add_grain(x, "ISO 3200", 100, seed=2, chroma="off")
        d2 = out2 - x
        self.assertTrue(torch.allclose(d2[..., 0], d2[..., 2], atol=1e-6))
        self.assertEqual(st2[0]["chroma_mode"], "off")

    def test_neutral_patch_in_colour_image_gets_no_colour_grain(self):
        x = flat(0.5, h=256, w=512)
        x[:, :, 256:] = torch.tensor([0.8, 0.3, 0.3])  # right half saturated red
        out, st = fg.add_grain(x, "ISO 1600", 100, seed=4)
        self.assertEqual(st[0]["chroma_mode"], "auto")
        d = out - x
        left = d[:, :, :200]; right = d[:, :, 300:]
        self.assertTrue(torch.allclose(left[..., 0], left[..., 2], atol=1e-6))  # grey half: neutral grain
        self.assertFalse(torch.allclose(right[..., 0], right[..., 2], atol=1e-4))  # red half: colour grain

    def test_monochrome_detection_on_tinted_frame(self):
        # a cast of a few levels is still monochrome (a Krea 2 "B&W" decode carries ~5.6/255); 11/255 is not
        faint = flat(0.5) + torch.tensor([0.012, 0.0, -0.012]).view(1, 1, 1, 3)
        strong = flat(0.5) + torch.tensor([0.03, 0.0, -0.03]).view(1, 1, 1, 3)
        self.assertTrue(fg.is_monochrome(faint.permute(0, 3, 1, 2))[0])
        self.assertFalse(fg.is_monochrome(strong.permute(0, 3, 1, 2))[0])

    def test_batch_rgba_dtype_and_zero_strength(self):
        rgb = torch.cat([flat(0.4, 64, 64), flat(0.6, 64, 64)], dim=0)
        alpha = torch.rand(2, 64, 64, 1)
        x = torch.cat([rgb, alpha], dim=-1).half()
        out, st = fg.add_grain(x, "ISO 800", 60, seed=10)
        self.assertEqual(out.dtype, torch.float16)
        self.assertEqual(tuple(out.shape), (2, 64, 64, 4))
        self.assertTrue(torch.equal(out[..., 3], x[..., 3]))
        self.assertEqual([s["seed"] for s in st], [10, 11])
        self.assertFalse(torch.equal(out[0, ..., :3] - x[0, ..., :3], out[1, ..., :3] - x[1, ..., :3]))
        out0, st0 = fg.add_grain(x, "ISO 800", 0, seed=10)
        self.assertTrue(torch.equal(out0, x))
        self.assertEqual(st0[0]["amp_255"], 0.0)

    def test_equal_loudness_across_iso(self):
        x = flat(fg.MIDTONE_REF)
        def std(iso):
            out, _ = fg.add_grain(x, iso, 50, seed=1, match_texture=False)
            return float((luma_np(out) - luma_np(x)).std())
        ref = std("ISO 200")
        for iso, mul in fg.ISO_LOUDNESS.items():
            self.assertAlmostEqual(std(iso) / ref, mul, delta=0.06, msg=iso)
        louds = [fg.ISO_LOUDNESS[n] for n in fg.ISO_NAMES]
        self.assertTrue(all(b < a for a, b in zip(louds, louds[1:])))  # coarser grain, less amplitude

    def test_status_line_and_validation(self):
        _, st = fg.add_grain(flat(0.4, 64, 64), "ISO 200", 50, seed=42, match_texture=False)
        line = fg.status_line(st, seconds=0.05)
        self.assertTrue(line.startswith("grain ISO 200 strength 50 · 3.3/255 midtone"))
        self.assertIn("monochrome (luma grain only)", line)
        self.assertIn("seed 42", line)
        self.assertNotIn("texture floor", line)
        _, st = fg.add_grain(flat(0.4, 64, 64), "ISO 400", 50, seed=42)
        line = fg.status_line(st)
        self.assertIn("x1.00 for texture floor 0.00/255", line)  # a clean card gets exactly the strength's amount
        self.assertTrue(line.startswith("grain ISO 400 strength 50 · 3.0/255 midtone"))  # 3.3 x loudness 0.9
        for ln in (line, fg.status_line(st, seconds=0.1)):
            self.assertNotIn(":", ln)  # Forge JSON-quotes infotext values containing a colon...
            self.assertNotIn(",", ln)  # ...or a comma
        with self.assertRaises(ValueError):
            fg.add_grain(flat(0.4, 64, 64), "ISO 12800", 50)
        with self.assertRaises(ValueError):
            fg.add_grain(flat(0.4, 64, 64), "ISO 400", 50, chroma="maybe")


class ToneAndSizeTests(unittest.TestCase):
    def test_amount_curves(self):
        L = torch.tensor([0.0, 0.03, 0.10, 0.25, 0.45, 0.65, 0.80, 0.90, 0.95, 1.0])
        pr = fg.amount_weight(L, "print")
        self.assertAlmostEqual(float(fg.amount_weight(torch.tensor([fg.MIDTONE_REF]), "print")), 1.0, places=6)
        self.assertAlmostEqual(float(fg.amount_weight(torch.tensor([fg.MIDTONE_REF]), "negative")), 1.0, places=6)
        self.assertEqual(float(pr[0]), 0.0)
        self.assertEqual(float(pr[-1]), 0.0)
        self.assertEqual(float(pr[-2]), 0.0)
        self.assertGreater(float(pr[2]), 1.5)  # darks get the most
        self.assertLess(float(pr[6]), 0.4)  # brights little
        ng = fg.amount_weight(L, "negative")
        self.assertGreater(float(ng[6]), 1.1)  # a negative scan: more in the highlights
        self.assertLess(float(ng[2]), 0.8)
        mid = torch.linspace(0, 1, 501)
        self.assertTrue(bool((fg.amount_weight(mid, "print") >= 0).all()))

    def test_film_type_moves_the_grain_between_darks_and_brights(self):
        dark, bright = flat(0.12), flat(0.80)
        def std(x, film):
            out, _ = fg.add_grain(x, "ISO 200", 50, seed=4, film=film, match_texture=False)
            return float((luma_np(out) - luma_np(x)).std())
        self.assertGreater(std(dark, "print"), std(bright, "print") * 3)
        self.assertGreater(std(bright, "negative"), std(dark, "negative"))
        with self.assertRaises(ValueError):
            fg.add_grain(dark, "ISO 200", 50, film="slide")

    def test_field_has_unit_variance_at_every_exposure(self):
        for v in (0.05, 0.3, 0.5, 0.9):
            L = torch.full((1, 1, 384, 384), v)
            f = fg.grain_field_exposure(L, 0.8, 0.02, seed=3)[0, 0].numpy()
            self.assertAlmostEqual(float(f.std()), 1.0, delta=0.04, msg=v)
            self.assertAlmostEqual(float(f.mean()), 0.0, delta=0.03, msg=v)

    def test_grain_is_coarser_in_the_shadows(self):
        x = torch.cat([torch.full((1, 512, 256, 3), 0.10), torch.full((1, 512, 256, 3), 0.70)], dim=2)
        out, st = fg.add_grain(x, "ISO 800", 50, seed=6, match_texture=False)
        d = luma_np(out) - luma_np(x)
        dark, bright = d[:, 16:240], d[:, 272:496]
        self.assertGreater(acf(dark, 1, 0), acf(bright, 1, 0) + 0.15)  # larger blobs
        self.assertGreater(kurt(dark), kurt(bright))  # clumpier
        self.assertGreater(st[0]["sigma_shadow_px"], st[0]["sigma_highlight_px"] * 1.7)


class PaperTests(unittest.TestCase):
    @staticmethod
    def polaroid(h=900, w=700, top=60, bottom=200, side=45, paper=0.91, seed=0):
        """Off-white paper border with a faint gradient around a busy picture."""
        g = torch.Generator().manual_seed(seed)
        img = torch.full((1, h, w, 3), paper) + torch.linspace(-0.01, 0.01, w).view(1, 1, w, 1)
        pic = torch.rand(1, h - top - bottom, w - 2 * side, 3, generator=g) * 0.8 + 0.1
        pic = torch.nn.functional.avg_pool2d(pic.permute(0, 3, 1, 2), 3, 1, 1).permute(0, 2, 3, 1)
        img[:, top : h - bottom, side : w - side] = pic
        return img

    def test_frame_detected_and_left_bit_exact(self):
        x = self.polaroid()
        fr = fg.detect_frame(x.permute(0, 3, 1, 2))
        self.assertIsNotNone(fr)
        self.assertEqual((fr["top"], fr["bottom"], fr["left"], fr["right"]), (60, 200, 45, 45))
        out, st = fg.add_grain(x, "ISO 400", 100, seed=2)
        self.assertEqual(st[0]["frame"]["bottom"], 200)
        self.assertTrue(torch.equal(out[:, :60], x[:, :60]))
        self.assertTrue(torch.equal(out[:, -200:], x[:, -200:]))
        self.assertTrue(torch.equal(out[:, :, :45], x[:, :, :45]))
        self.assertTrue(torch.equal(out[:, :, -45:], x[:, :, -45:]))
        self.assertFalse(torch.equal(out[:, 100:600, 100:600], x[:, 100:600, 100:600]))
        self.assertIn("frame excluded 60/200/45/45 px", fg.status_line(st))

    def test_no_frame_on_a_sky_band_or_a_plain_photo(self):
        x = self.polaroid()[:, 60:-200, 45:-45]  # the picture alone
        self.assertIsNone(fg.detect_frame(x.permute(0, 3, 1, 2)))
        sky = x.clone()
        sky[:, :120] = 0.75  # a clear sky along one edge only
        self.assertIsNone(fg.detect_frame(sky.permute(0, 3, 1, 2)))
        _, st = fg.add_grain(sky, "ISO 200", 50, seed=1)
        self.assertIsNone(st[0]["frame"])
        _, st = fg.add_grain(self.polaroid(), "ISO 200", 50, seed=1, frame_detect=False)
        self.assertIsNone(st[0]["frame"])

    def test_flat_paper_white_inside_the_picture_gets_no_grain(self):
        x = flat(0.5, 256, 512)
        x[:, 64:192, 300:460] = 0.9  # flat paper white
        g = torch.Generator().manual_seed(1)
        x[:, 64:192, 40:200] = 0.88 + torch.randn(1, 128, 160, 1, generator=g) * (3 / 255)  # bright but textured
        out, _ = fg.add_grain(x, "ISO 200", 100, seed=3, match_texture=False)
        d = luma_np(out) - luma_np(x)
        self.assertLess(float(np.abs(d[104:152, 344:424]).max()), 0.05)  # paper core: nothing (the gate ramps over two blocks at its edge)
        self.assertGreater(float(d[90:166, 60:180].std()), 0.3)  # textured bright area: a little grain


class TextureMatchTests(unittest.TestCase):
    @staticmethod
    def textured(level_255: float, h: int = 512, w: int = 512, seed: int = 0) -> torch.Tensor:
        """Mid-grey with white noise of a given std, /255: a known texture floor."""
        g = torch.Generator().manual_seed(seed)
        n = torch.randn(1, h, w, 1, generator=g) * (level_255 / 255.0)
        return (torch.full((1, h, w, 3), 0.4) + n).clamp(0, 1)

    def test_floor_measures_known_texture(self):
        for level in (0.5, 1.1, 3.0):
            x = self.textured(level).permute(0, 3, 1, 2)
            f = fg.texture_floor_255(x)
            # white noise of std s keeps ~s*sqrt(1-1/81) after a 9 px box high-pass; the 10th
            # percentile of 64 blocks sits a little under that
            self.assertAlmostEqual(f, level, delta=level * 0.12 + 0.02, msg=level)
        self.assertEqual(fg.texture_floor_255(flat(0.4).permute(0, 3, 1, 2)), 0.0)
        # blocks near black or white do not count: a clean midtone half decides, not a noisy white half
        x = self.textured(4.0)
        x[:, :, 256:] = 1.0
        x[:, :, :256] = 0.4
        self.assertLess(fg.texture_floor_255(x.permute(0, 3, 1, 2)), 0.1)

    def test_factor_curve(self):
        self.assertEqual(fg.texture_factor(0.0), 1.0)  # clean: never less than the strength asks for
        self.assertEqual(fg.texture_factor(0.45), 1.0)
        self.assertEqual(fg.texture_factor(1.1), 1.0)
        self.assertAlmostEqual(fg.texture_factor(1.65), 1.5 ** 0.6, places=6)
        self.assertEqual(fg.texture_factor(50.0), fg.MASK_MAX_FACTOR)
        vals = [fg.texture_factor(f / 10) for f in range(0, 80)]
        self.assertTrue(all(b >= a for a, b in zip(vals, vals[1:])))  # monotone

    def test_clean_image_gets_the_full_amount_textured_image_more(self):
        clean, rough = self.textured(0.45, seed=1), self.textured(3.0, seed=2)
        out_c, st_c = fg.add_grain(clean, "ISO 200", 50, seed=7)
        out_r, st_r = fg.add_grain(rough, "ISO 200", 50, seed=7)
        self.assertEqual(st_c[0]["texture_factor"], 1.0)
        self.assertAlmostEqual(st_c[0]["amp_255"], fg.strength_to_amp(50), places=6)
        self.assertGreater(st_r[0]["texture_factor"], 1.3)
        self.assertAlmostEqual(st_r[0]["amp_255"], st_r[0]["nominal_255"] * st_r[0]["texture_factor"], places=6)
        added_c = (luma_np(out_c) - luma_np(clean)).std()
        added_r = (luma_np(out_r) - luma_np(rough)).std()
        self.assertGreater(added_r, added_c * 1.25)
        # off: the nominal amount on both
        _, st_off = fg.add_grain(rough, "ISO 200", 50, seed=7, match_texture=False)
        self.assertAlmostEqual(st_off[0]["amp_255"], fg.strength_to_amp(50), places=6)

    def test_boost_is_capped(self):
        rough = self.textured(20.0, seed=3)
        _, st = fg.add_grain(rough, "ISO 200", 50, seed=7)
        self.assertEqual(st[0]["texture_factor"], fg.MASK_MAX_FACTOR)
        _, st100 = fg.add_grain(rough, "ISO 200", 100, seed=7)
        self.assertAlmostEqual(st100[0]["amp_255"], fg.strength_to_amp(100) * fg.MASK_MAX_FACTOR, places=5)
        _, st100 = fg.add_grain(rough, "ISO 100", 100, seed=7)  # 8 x 1.15 x 1.4 = 12.9 -> the absolute cap
        self.assertAlmostEqual(st100[0]["amp_255"], fg.AMP_CAP_255, places=5)
        _, st0 = fg.add_grain(rough, "ISO 200", 0, seed=7)
        self.assertEqual(st0[0]["amp_255"], 0.0)  # strength 0 stays off

    def test_strength_still_controls_a_matched_image(self):
        rough = self.textured(3.0, seed=1)
        amps = [fg.add_grain(rough, "ISO 200", s, seed=7)[1][0]["amp_255"] for s in (25, 50, 75)]
        self.assertTrue(amps[0] < amps[1] < amps[2], amps)

    def test_batch_images_are_matched_individually(self):
        x = torch.cat([self.textured(0.4, 256, 256, 1), self.textured(3.0, 256, 256, 2)], dim=0)
        _, st = fg.add_grain(x, "ISO 200", 50, seed=3)
        self.assertEqual(st[0]["texture_factor"], 1.0)
        self.assertGreater(st[1]["texture_factor"], 1.3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
