"""Extract Evo2-40B internal embeddings for the atlas + eQTL panels (for the representational
hooks: H1 cross-species coding-manifold alignment, H3 eQTL probe-vs-likelihood).

Runs AFTER the scoring run on the same resident 40B (one extra forward pass each for ref + alt).
To keep it cheap we center-crop each 8192bp window to `--crop` bp around the variant (var_off) --
embedding geometry does not need the full context, and a 2048bp crop is ~4x cheaper than 8192.

For each variant, at two mid/late layers, we save:
  - {layer}_ref_mean    : mean-pooled hidden state over the ref crop  -> LOCUS / manifold feature (H1)
  - {layer}_delta_center: alt(center) - ref(center) hidden state      -> VARIANT-EFFECT feature (H3 probe)
fp16, np.savez_compressed per panel -> data/processed/emb/{name}_emb.npz
[variant_id, label, {layer}_ref_mean [N,D], {layer}_delta_center [N,D]]

  python src/ccs/score_evo2_embeddings.py --crop 2048 --batch 16
"""
import argparse, glob, os, re, sys, time
import numpy as np
import polars as pl

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WIN = 8192


def pick_layers(model):
    """Introspect the resident model for two mid/late `blocks.N.mlp.l3` layers (~55%, ~80% depth)."""
    mods = [n for n, _ in model.model.named_modules()]
    l3 = sorted((int(re.search(r"blocks\.(\d+)", n).group(1)), n)
                for n in mods if re.fullmatch(r"blocks\.\d+\.mlp\.l3", n))
    if l3:
        idx = [i for i, _ in l3]
        return [f"blocks.{idx[int(0.55 * len(idx))]}.mlp.l3",
                f"blocks.{idx[int(0.80 * len(idx))]}.mlp.l3"]
    # fallback: whole-block outputs
    bl = sorted(set(int(re.search(r"blocks\.(\d+)", n).group(1))
                    for n in mods if re.match(r"blocks\.\d+", n)))
    return [f"blocks.{bl[int(0.55 * len(bl))]}", f"blocks.{bl[int(0.80 * len(bl))]}"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--outdir", default="data/processed/emb")
    ap.add_argument("--atlas-glob", default="data/interim/atlas8192/*_windows_8192.parquet")
    ap.add_argument("--extra", nargs="*", default=["data/interim/ablation/eqtl_abl_8192.parquet"])
    a = ap.parse_args()

    import torch
    from evo2 import Evo2

    print("loading Evo2-40B (resident) ...", flush=True)
    model = Evo2("evo2_40b")
    tok = model.tokenizer
    layers = pick_layers(model)
    print("embedding layers:", layers, flush=True)
    os.makedirs(a.outdir, exist_ok=True)
    half = a.crop // 2

    def reduce_batch(seqs, ci_local):
        ids = torch.stack([torch.tensor(tok.tokenize(s), dtype=torch.int) for s in seqs]).to("cuda:0")
        with torch.inference_mode():
            _, emb = model(ids, return_embeddings=True, layer_names=layers)
        B = ids.shape[0]
        ar = torch.arange(B, device=ids.device)
        ci = torch.tensor(ci_local, device=ids.device)
        means, centers = {}, {}
        for ly in layers:
            h = emb[ly].float()                      # [B, L, D]
            means[ly] = h.mean(dim=1).half().cpu().numpy()
            centers[ly] = h[ar, ci, :].half().cpu().numpy()
        del emb
        return means, centers

    panels = sorted(glob.glob(a.atlas_glob)) + list(a.extra)
    for p in panels:
        if not os.path.exists(p):
            print("MISSING panel, skip:", p, flush=True); continue
        name = os.path.basename(p).replace("_windows_8192", "").replace(".parquet", "")
        out = os.path.join(a.outdir, f"{name}_emb.npz")
        if os.path.exists(out):
            print("skip (done):", out, flush=True); continue

        df = pl.read_parquet(p)
        vids = df["variant_id"].to_list()
        labs = df["label"].to_list() if "label" in df.columns else [-1] * df.height
        offs = df["var_off"].to_list() if "var_off" in df.columns else [WIN // 2] * df.height
        refs = df["ref_seq"].to_list()
        alts = df["alt_seq"].to_list()
        N = len(vids)

        acc = {}
        for ly in layers:
            acc[f"{ly}_ref_mean"] = []
            acc[f"{ly}_delta_center"] = []
        t0 = time.time()
        for s in range(0, N, a.batch):
            e = min(s + a.batch, N)
            starts = [min(max(0, offs[i] - half), WIN - a.crop) for i in range(s, e)]
            ci = [offs[i] - st for i, st in zip(range(s, e), starts)]
            rc = [refs[i][st:st + a.crop] for i, st in zip(range(s, e), starts)]
            ac = [alts[i][st:st + a.crop] for i, st in zip(range(s, e), starts)]
            rmean, rcen = reduce_batch(rc, ci)
            amean, acen = reduce_batch(ac, ci)
            for ly in layers:
                acc[f"{ly}_ref_mean"].append(rmean[ly])
                acc[f"{ly}_delta_center"].append(acen[ly] - rcen[ly])
            print(f"  {name} {e}/{N}  {e/(time.time()-t0):.2f}/s", flush=True)

        out_d = {"variant_id": np.array(vids), "label": np.array(labs)}
        for k, v in acc.items():
            out_d[k.replace(".", "_")] = np.concatenate(v)
        np.savez_compressed(out, **out_d)
        print(f"wrote {out}: {N} vars in {time.time()-t0:.0f}s", flush=True)

    print("ALL EMB DONE", flush=True)


if __name__ == "__main__":
    main()
