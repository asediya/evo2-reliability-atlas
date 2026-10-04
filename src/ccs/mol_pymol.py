"""Publication-grade molecular renders for the figures, via ray-traced open-source PyMOL.

Why this exists: the earlier assets came from 3Dmol.js (WebGL rasterisation), which has no ray tracing,
no ambient occlusion and no silhouettes — at hero size a cartoon degenerates into a "ribbon hairball".
PyMOL ray-tracing gives real occlusion shading, ink outlines and depth, so a structure reads as a solid OBJECT.

Run with the dedicated PyMOL environment (NOT the project venv):
    .pymol/python.exe -m src.ccs.mol_pymol            # renders every structure
    .pymol/python.exe src/ccs/mol_pymol.py p53 nucl   # or a subset

Outputs assets/mol/<name>_px.png (full frame, transparent) and <name>_px_crop.png (cropped to content).
"""
import os, sys

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "mol")
FETCH = os.environ.get("PYMOL_FETCH_DIR", os.path.join(OUT, "_pdb"))
SIZE = int(os.environ.get("MOL_SIZE", "2600"))

# ---- palette shared with the figures (fig3_style) ----
CODING = "0x0072B2"   # blue
FAIL = "0xC1443E"     # red  — reserved for the variant/eQTL, the only saturated accent
DNA = "0x2E6E9E"      # DNA blue
HIST = "0xC9C2B6"     # histone tan
PROT = "0x8FA8BF"     # neutral protein


def _base_settings(cmd):
    """The publication render recipe: occlusion + ink outline + orthographic + transparent."""
    cmd.set("ray_opaque_background", 0)
    cmd.set("antialias", 2)
    cmd.set("orthoscopic", 1)
    cmd.set("ray_trace_mode", 1)          # silhouette / ink outline
    cmd.set("ray_trace_color", "grey20")
    cmd.set("ray_trace_gain", 0.12)
    cmd.set("ambient_occlusion_mode", 1)  # the thing WebGL cannot do
    cmd.set("ambient_occlusion_scale", 18)
    cmd.set("ambient", 0.12)
    cmd.set("direct", 0.45)
    cmd.set("reflect", 0.50)
    cmd.set("specular", 0.15)
    cmd.set("spec_power", 200)
    cmd.set("ray_shadows", 0)
    cmd.set("cartoon_fancy_helices", 1)
    cmd.set("cartoon_highlight_color", "grey40")
    cmd.set("depth_cue", 0)
    cmd.set("surface_quality", 1)


def _crop(path):
    """Trim transparent margin -> <stem>_crop.png (needs Pillow; skipped silently if absent)."""
    try:
        from PIL import Image
    except ImportError:
        return None
    im = Image.open(path).convert("RGBA")
    bb = im.getbbox()
    if bb:
        pad = 10
        bb = (max(0, bb[0] - pad), max(0, bb[1] - pad),
              min(im.width, bb[2] + pad), min(im.height, bb[3] + pad))
        im = im.crop(bb)
    out = os.path.splitext(path)[0] + "_crop.png"
    im.save(out)
    return im.size


# ---------------------------------------------------------------- per-structure recipes
def r_p53(cmd):
    """p53 DNA-binding domain + the three pathogenic hotspot residues as the only saturated colour."""
    cmd.fetch("2OCJ", async_=0)
    cmd.remove("solvent or not polymer")
    cmd.hide("everything"); cmd.show("cartoon", "chain A")
    cmd.color("grey70", "chain A")
    cmd.color("slate", "chain A and ss S")
    cmd.color("lightblue", "chain A and ss H")
    cmd.select("hot", "chain A and resi 175+248+273")
    cmd.show("spheres", "hot and not name N+C+O")
    cmd.color("firebrick", "hot"); cmd.set("sphere_scale", 0.85, "hot")
    cmd.orient("chain A"); cmd.zoom("chain A", 2.0)


def r_p53_dna(cmd):
    """p53 bound to its DNA response element — shows WHY a coding change matters (1TUP)."""
    cmd.fetch("1TUP", async_=0)
    cmd.remove("solvent or resn ZN")
    cmd.hide("everything")
    cmd.show("cartoon", "polymer.protein and chain A")
    cmd.show("cartoon", "polymer.nucleic")
    cmd.color("grey70", "polymer.protein")
    cmd.color("slate", "polymer.protein and ss S")
    cmd.color(DNA, "polymer.nucleic")
    cmd.set("cartoon_ring_mode", 3, "polymer.nucleic")
    cmd.set("cartoon_ring_finder", 1)
    cmd.select("hot", "chain A and resi 248+273")
    cmd.show("spheres", "hot and not name N+C+O")
    cmd.color("firebrick", "hot"); cmd.set("sphere_scale", 0.8, "hot")
    cmd.orient("chain A or polymer.nucleic"); cmd.zoom("chain A or polymer.nucleic", 2.0)


def r_nucleosome(cmd):
    """Nucleosome: solid histone core (surface) + bold DNA cartoon — reads as a spool at any size."""
    cmd.fetch("1KX5", async_=0)
    cmd.remove("solvent")
    cmd.hide("everything")
    cmd.show("surface", "polymer.protein")
    cmd.show("cartoon", "polymer.nucleic")
    cmd.color(HIST, "polymer.protein")
    cmd.color(DNA, "polymer.nucleic")
    cmd.set("cartoon_nucleic_acid_mode", 4)
    cmd.set("cartoon_tube_radius", 1.6, "polymer.nucleic")
    cmd.set("transparency", 0.0)
    cmd.orient(); cmd.zoom("all", 2.0)


def r_cohesin(cmd):
    """Cohesin/SMC — the loop-extruder; elongated coiled-coil silhouette."""
    cmd.fetch("6WG3", async_=0)
    cmd.remove("solvent")
    cmd.hide("everything"); cmd.show("cartoon", "polymer")
    cmd.color(PROT, "polymer.protein")
    cmd.color(DNA, "polymer.nucleic")
    cmd.orient(); cmd.zoom("all", 2.0)


def r_tf_dna(cmd):
    """A transcription factor gripping DNA (glucocorticoid receptor, 1GLU) — the enhancer-binding motif."""
    cmd.fetch("1GLU", async_=0)
    cmd.remove("solvent")
    cmd.hide("everything"); cmd.show("cartoon", "polymer")
    cmd.color("grey70", "polymer.protein")
    cmd.color("lightblue", "polymer.protein and ss H")
    cmd.color(DNA, "polymer.nucleic")
    cmd.set("cartoon_ring_mode", 3, "polymer.nucleic")
    cmd.orient(); cmd.zoom("all", 2.0)


def r_ctcf(cmd):
    """CTCF zinc fingers on DNA — the insulator/loop-anchor protein."""
    cmd.fetch("5T00", async_=0)
    cmd.remove("solvent")
    cmd.hide("everything"); cmd.show("cartoon", "polymer")
    cmd.color("grey70", "polymer.protein")
    cmd.color(DNA, "polymer.nucleic")
    cmd.show("spheres", "resn ZN"); cmd.color("grey40", "resn ZN")
    cmd.set("sphere_scale", 0.4, "resn ZN")
    cmd.set("cartoon_ring_mode", 3, "polymer.nucleic")
    cmd.orient(); cmd.zoom("all", 2.0)


def r_bdna(cmd):
    """Canonical B-DNA duplex (1BNA) — the raw substrate the model actually reads."""
    cmd.fetch("1BNA", async_=0)
    cmd.remove("solvent")
    cmd.hide("everything"); cmd.show("cartoon", "polymer.nucleic")
    cmd.color(DNA, "polymer.nucleic")
    cmd.set("cartoon_ring_mode", 3); cmd.set("cartoon_ring_finder", 1)
    cmd.set("cartoon_nucleic_acid_mode", 4)
    cmd.orient(); cmd.zoom("all", 2.0)


def r_p53_tsr(cmd):
    """THE HERO. p53 core domain bound to its DNA response element (1TSR, Cho et al. Science 1994) —
    the structure in which R248/R273 are visibly DNA-CONTACT residues, so the annotation is true.
    Mono grey value-ramp for the protein and a desaturated DNA; ALL chroma reserved for the hotspots."""
    cmd.fetch("1TSR", async_=0)
    cmd.remove("solvent")
    cmd.hide("everything")
    # chain B is the DNA-engaging monomer in this entry; keep only it plus both DNA strands
    cmd.show("cartoon", "chain B or chain E or chain F")
    cmd.color("grey80", "chain B")
    cmd.color("grey55", "chain B and ss S")
    cmd.color("grey35", "chain B and ss H")
    cmd.color("grey65", "chain E or chain F")
    cmd.set("cartoon_ring_mode", 3, "chain E or chain F")
    cmd.set("cartoon_ring_finder", 1)
    cmd.set("cartoon_nucleic_acid_mode", 4)
    cmd.show("spheres", "resn ZN and chain B"); cmd.color("grey45", "resn ZN and chain B")
    cmd.set("sphere_scale", 0.35, "resn ZN")
    cmd.select("hot", "chain B and resi 175+248+273")
    cmd.show("spheres", "hot and not name N+C+O")
    cmd.color("firebrick", "hot"); cmd.set("sphere_scale", 0.9, "hot")
    # camera: orient on the DNA duplex so the helix lies horizontal, then tip slightly so the
    # protein sits beside (not behind) it and the hotspot-to-DNA contacts stay unoccluded
    cmd.orient("chain E or chain F")
    cmd.turn("x", -20)
    cmd.zoom("chain B or chain E or chain F", 1.2)


RECIPES = {
    "p53tsr": ("p53_tsr_pm", r_p53_tsr),
    "p53": ("p53_pm", r_p53),
    "p53dna": ("p53_dna_pm", r_p53_dna),
    "nucl": ("nucleosome_pm", r_nucleosome),
    "cohesin": ("cohesin_pm", r_cohesin),
    "tfdna": ("tf_dna_pm", r_tf_dna),
    "ctcf": ("ctcf_pm", r_ctcf),
    "bdna": ("bdna_pm", r_bdna),
}


def main(keys):
    import pymol
    pymol.finish_launching(["pymol", "-qc"])
    from pymol import cmd
    os.makedirs(OUT, exist_ok=True); os.makedirs(FETCH, exist_ok=True)
    cmd.set("fetch_path", FETCH)
    for k in keys:
        if k not in RECIPES:
            print("skip unknown:", k); continue
        name, fn = RECIPES[k]
        cmd.reinitialize()
        cmd.set("fetch_path", FETCH)
        _base_settings(cmd)
        try:
            fn(cmd)
            _base_settings(cmd)                      # recipe may have overridden; reassert
            cmd.ray(SIZE, SIZE)
            p = os.path.join(OUT, name + ".png")
            cmd.png(p, dpi=600)
            sz = _crop(p)
            print(f"OK  {name:16s} {os.path.getsize(p):>9,d}B  crop={sz}", flush=True)
        except Exception as e:
            print(f"FAIL {name}: {e}", flush=True)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    main(args if args else list(RECIPES))
