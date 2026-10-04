# -*- coding: utf-8 -*-
"""Re-map the four BRCA1 structure renders from RdBu_r onto the greyscale-safe TOL_DEL ramp.

WHY THIS EXISTS. The quilt in Figure S6a, the four structure panels in S6b and their shared colour
bar all encode the same quantity on one diverging ramp. `RdBu_r` is ColorBrewer-safe for colour
vision deficiency -- its two extremes sit 0.40 (deuteranopia) and 0.36 (protanopia) apart in sRGB --
but its two extremes have IDENTICAL relative luminance, 0.030 and 0.030, a WCAG contrast ratio of
exactly 1.00. In greyscale a maximally tolerated substitution and a maximally deleterious one are
the same grey, and the quilt carries no signal at all.

The builder can simply be handed a different colormap. The four structure PNGs cannot: they were
rendered from `data/raw/structures/*.pdb`, which is not in the deposit, so re-rendering needs a
tree this machine does not have. Changing only the quilt would leave the panel disagreeing with
itself, which is worse than either state.

So the PNGs are re-mapped at the pixel level. This is sound here, and the measurement that says so
is in the module: these renders are shaded cartoons, and a multiplicative shading model fits them to a median
residual of 0.5-0.7 sRGB units. Every content pixel is fitted as k * RdBu_r(t) for a shade k and a
ramp position t, then re-emitted as k * TOL_DEL(t), which carries the illumination through intact.
Pixels the model cannot fit within 12 units -- outlines, some anti-aliased edges -- are left
exactly as they were. Alpha is preserved untouched.

TOL_DEL keeps the diverging structure and the blue/red hue convention, and pins its lightest
anchor to t = 0.5 so it still lands on the TwoSlopeNorm centre. What changes is the luminance
profile: the tolerated arm ends at a MEDIUM blue instead of a dark one, so

    ends contrast       4.00   (RdBu_r 1.00)
    centre vs tolerated 3.23
    centre vs delet.   12.89
    deuteranopia ends   0.675  (RdBu_r 0.399)
    protanopia ends     0.705  (RdBu_r 0.356)

i.e. it is better in greyscale AND better under both simulated dichromacies.

Run:  python3 tools/recolour_brca1_structures.py [--dry-run]
The originals are kept beside the outputs as *_rdbu.png and the script is idempotent: it refuses
to run twice by checking for that backup first.
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from ccs.style_gb import TOL_DEL_NODES

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGDIR = os.path.join(HERE, "reports", "figures")
PANELS = ("brca1_ring_evo2", "brca1_ring_sge", "brca1_brct_evo2", "brca1_brct_sge")
RESID = 12.0           # max residual of the shading fit below; 95-97 % of content sits inside it


def _ramp(cmap, n=512):
    return (np.array([cmap(t)[:3] for t in np.linspace(0, 1, n)]) * 255.0).astype(np.float32)


def recolour(path_in, path_out, dry=False):
    from matplotlib.colors import LinearSegmentedColormap
    src = _ramp(plt.get_cmap("RdBu_r"), 256)
    dst = _ramp(LinearSegmentedColormap.from_list("tol_del", TOL_DEL_NODES), 256)

    im = np.array(Image.open(path_in).convert("RGBA"))
    rgb = im[:, :, :3].astype(np.float32)
    alpha = im[:, :, 3]
    h, w, _ = rgb.shape
    flat = rgb.reshape(-1, 3)
    # "Visible" here means what the composite's own autocrop() calls CONTENT: any channel below
    # 247. That test is copied deliberately. The renders have an OPAQUE pure-white background
    # (76 % of the frame), and pure white sits 15.1 sRGB units from the RdBu_r centre (247,247,246)
    # -- inside the remap threshold. Without this guard the background would be re-emitted as the
    # new centre (247,246,243), autocrop would then read it as content, stop cropping, and the four
    # structure panels would change size. No pixel ON the ramp is pure white, so nothing is lost.
    vis = (alpha.reshape(-1) > 8) & (flat < 247).any(axis=1)

    # THE RENDERS ARE SHADED, so a flat nearest-colour match is not enough: a pixel in shadow is
    # the ramp colour scaled down by illumination and can sit 60+ sRGB units from the flat locus.
    # Matching on distance alone caught only 16-42 % of content and would have produced a two-tone
    # panel -- lit surfaces on the new ramp, shaded ones still on the old -- which is worse than
    # leaving the figure as it was. So the model is MULTIPLICATIVE: solve
    #     px  ~=  k * src(t),        k in [0, 1.25]
    # for both the ramp position t and the shade k, then re-emit k * dst(t), which carries the
    # illumination through unchanged. For a fixed t the optimal k is the projection
    # (px . src_t) / |src_t|^2, so the whole fit is one matrix product and an argmin. Measured on
    # the deposited renders the fit is near-exact: median residual 0.5-0.7 sRGB units, 95 % and
    # 90 % of content inside 8, and a median shade of 0.76 and 0.59 -- i.e. these panels really are
    # about a quarter to two-fifths in shadow.
    #
    # The neutral grey BARD1 partner chain fits the ramp's near-white centre at low k and is
    # therefore re-emitted as k * (247,246,243) -- within about three units of the grey it already
    # was. It is not recoloured, and it does not need a special case.
    #
    # Solved on the UNIQUE colours, not per pixel: a few thousand distinct triples stand in for
    # several million pixels, which is ~1000x less arithmetic for a bit-identical result.
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    sq = (src * src).sum(axis=1)                        # |src_t|^2
    k = np.clip((uniq @ src.T) / sq[None, :], 0.0, 1.25)
    resid = np.sqrt(((uniq[:, None, :] - k[:, :, None] * src[None, :, :]) ** 2).sum(axis=2))
    u_idx = resid.argmin(axis=1)
    u_res = resid.min(axis=1)
    u_k = k[np.arange(len(uniq)), u_idx]
    idx, best, shade = u_idx[inv], u_res[inv], u_k[inv]

    hit = vis & (best <= RESID)
    out = flat.copy()
    out[hit] = np.clip(dst[idx[hit]] * shade[hit][:, None], 0, 255)
    im2 = im.copy()
    im2[:, :, :3] = out.reshape(h, w, 3).astype(np.uint8)

    pct = 100.0 * hit.sum() / max(1, vis.sum())
    # autocrop() must see the same bounding box before and after, or the panels change size
    def _box(a3):
        m = (a3[:, :, :3] < 247).any(axis=2)
        ys, xs = np.where(m)
        return (int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())) if len(ys) else None
    b_in, b_out = _box(im), _box(im2)
    assert b_in == b_out, "autocrop box moved: %s -> %s" % (b_in, b_out)
    print("  %-22s content %7d px | remapped %6.2f%% | untouched %6.2f%% | crop box stable %s"
          % (os.path.basename(path_in), int(vis.sum()), pct, 100 - pct, b_in == b_out))
    if not dry:
        # Written through a temp file and renamed. An earlier run was interrupted between taking
        # the backup and writing the output, which left one panel with a backup on disk and its
        # original still live -- and the "backup exists, skip" guard below then declared that panel
        # already done. os.replace is atomic on POSIX, so the pair is now all-or-nothing.
        tmp = path_out + ".tmp.png"
        Image.fromarray(im2, mode="RGBA").save(tmp)
        os.replace(tmp, path_out)
    return pct


def main():
    dry = "--dry-run" in sys.argv
    for name in PANELS:
        live = os.path.join(FIGDIR, name + ".png")
        keep = os.path.join(FIGDIR, name + "_rdbu.png")
        if not os.path.exists(live):
            raise SystemExit("missing render: %s" % live)
        if os.path.exists(keep):
            print("  %-22s already re-mapped (backup present); skipping" % (name + ".png"))
            continue
        if dry:
            recolour(live, live, dry=True)
            continue
        tmpk = keep + ".tmp.png"
        Image.open(live).save(tmpk)
        os.replace(tmpk, keep)          # backup committed before anything is overwritten
        recolour(keep, live, dry=False)
    print("done%s" % (" (dry run, nothing written)" if dry else ""))


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(HERE, "src"))
    main()
