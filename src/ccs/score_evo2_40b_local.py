"""Run Evo2-40B on a single GPU by streaming its 50 StripedHyena2 blocks one-at-a-time
from 256GB CPU RAM -> GPU -> evict. Works because vortex's stateless_forward threads only the
hidden tensor block->block (inference_params=None => no cross-block state), so blocks are evictable.

Design (validated by reading vortex source):
 - Build StripedHyena with torch.cuda.is_available()==False so ALL blocks construct on CPU (no 80GB
   GPU OOM at construction; model.py:686 branches on cuda availability).
 - Load 40B bf16 weights into RAM (fits 256GB; the process needs >=200GB of host memory).
 - Move only embedding + final norm + unembed (tiny: vocab 512 x 8192) to cuda:0.
 - Monkeypatch StripedHyena.stateless_forward: for each block -> move_to_device(block,'cuda') +
   fixup_fp8_extra_states(block) [preserves TE FP8 scale/amax across the move] -> run -> evict to cpu.
 - Batch the left-context prefixes so the ~80GB/forward weight stream amortizes over the batch.
 - VALIDATE against the NIM-API oracle (score_evo2_40b_api.py output) before trusting the full run.

Run in the evo2-emb container, HF cache mounted, --gpus all:
  python score_evo2_40b_local.py --in windows.parquet --out scores.parquet --batch 64 [--oracle nim.parquet --limit 20]
"""
import argparse, sys, time, math, types
import numpy as np
import polars as pl
import torch


_SAFE_FIXUP = None


def _install_safe_fp8_fixup():
    """Replace vortex.fixup_fp8_extra_states with a CPU-safe version: skip modules whose params are
    on CPU (their FP8 extra-state is re-fixed at stream time on the GPU); do the real fixup for
    modules already on CUDA. Avoids torch.cuda.device('cpu') crashing on CPU-resident blocks."""
    import vortex.model.utils as _vu
    import vortex.model.model as _vm

    def safe(module):
        if not hasattr(module, "fp8_meta"):
            for c in module.children():
                safe(c)
            return
        for c in module.children():
            safe(c)
        try:
            device = next(module.parameters()).device
        except StopIteration:
            return
        if device.type == "cpu":
            return                                    # skip on CPU; redone on GPU at stream time
        module.fp8_meta_tensors_initialized = False
        with torch.cuda.device(device):
            module.set_extra_state(module.get_extra_state())

    _vu.fixup_fp8_extra_states = safe
    _vm.fixup_fp8_extra_states = safe
    return safe


def build_cpu_model(model_name="evo2_40b"):
    """Load an Evo 2 checkpoint with each block evicted to CPU RAM immediately after it's built on
    the GPU (TE FP8 Linear needs CUDA at construction, so blocks build on cuda:0 then are evicted ->
    peak GPU is one ~1.6GB block, never 80GB).

    model_name is a parameter so the smaller checkpoints can be scored through THIS path. That
    matters for comparability rather than for speed: the deposited 1B and 7B scores for the splicing
    and ClinVar panels come from the BioNeMo window-sum scorer, while this one is a single-position
    next-token readout. Reading a 40B number from here against a 1B number from there changes the
    readout and the capacity together, which is why Additional file 1 compares checkpoints only at a shared readout.
    A ladder built entirely through this function is a capacity ladder; a ladder assembled across
    the two scorers is not."""
    global _SAFE_FIXUP
    import evo2 as _evo2
    import vortex.model.model as _vm
    _SAFE_FIXUP = _install_safe_fp8_fixup()           # cpu-safe fixup used at load AND stream time
    _orig_mtd = _vm.move_to_device
    def _to_cpu(module, device):                      # force each freshly-built block off the GPU
        return _orig_mtd(module, "cpu")
    _vm.move_to_device = _to_cpu
    try:
        m = _evo2.Evo2(model_name)                    # each block built on cuda:0 then evicted to CPU
    finally:
        _vm.move_to_device = _orig_mtd
    sh = m.model                                      # vortex StripedHyena
    for i in range(len(sh.blocks)):
        _orig_mtd(sh.blocks[i], "cpu")                # ensure every block resides in CPU RAM
        sh.block_idx_to_device[i] = "cpu"
    for name in ("embedding_layer", "norm", "unembed"):   # tiny modules -> GPU
        mod = getattr(sh, name, None)
        if mod is not None:
            _orig_mtd(mod, "cuda:0")
    sh.block_idx_to_device[0] = "cuda:0"              # forward() returns/norms on device[0]
    return m, sh


def patch_streaming(sh):
    """Replace stateless_forward with a block-streaming version (CPU<->GPU per block)."""
    from vortex.model.utils import move_to_device, fixup_fp8_extra_states
    def streaming_stateless_forward(self, x, padding_mask=None):
        if isinstance(padding_mask, torch.Tensor):
            x = x * padding_mask[..., None]
        x = x.to("cuda:0")
        for block in self.blocks:
            move_to_device(block, "cuda:0")
            fixup_fp8_extra_states(block)             # keep TE FP8 scale/amax valid after the move
            x, _ = block(x, inference_params=None, padding_mask=padding_mask)
            move_to_device(block, "cpu")              # evict; caching allocator reuses the buffer
        return x, None
    sh.stateless_forward = types.MethodType(streaming_stateless_forward, sh)


def score_col(model_name):
    """Column name for a checkpoint: evo2_1b_base -> 'evo2_1b_base_neg'.

    The column carried the model in its name from the start, and it has to keep doing so now that
    this scorer can run more than one checkpoint. A 1B file with a column called `evo2_40b_neg` is
    the metric-key collision this repository already had to go and fix elsewhere: the file name says
    one thing, the column says another, and whichever a reader joins on wins silently.
    """
    return "%s_neg" % model_name


def score_panel(inp, out, batch, tok, sh, limit=0, oracle=None, model_name="evo2_40b"):
    """Score one windows panel with the already-built streaming model. Checkpoint/resume-safe:
    skips variant_ids already in `out`, flushes after every streaming forward."""
    import os
    col = score_col(model_name)
    df = pl.read_parquet(inp)
    if limit:
        df = df.head(limit)
    scored = {}
    if os.path.exists(out) and not limit:
        prev = pl.read_parquet(out)
        scored = dict(zip(prev["variant_id"].to_list(), prev[col].to_list()))
        df = df.filter(~pl.col("variant_id").is_in(list(scored)))
        print(f"  resume: {len(scored)} already scored, {df.height} remaining", flush=True)
    vids = df["variant_id"].to_list(); ref = df["ref_seq"].to_list()
    alt = df["alt_seq"].to_list(); voff = df["var_off"].to_list()

    # var_off must index the substituted base. It is the one thing this scorer cannot infer and
    # cannot survive getting wrong: the score is lp[alt_base] - lp[ref_base] at that offset, so if
    # var_off points anywhere else the two bases are identical and EVERY delta is exactly 0.0. That
    # happened -- the splicing and ClinVar panels were built 1,001 bp with the variant at index 500
    # but carried var_off = 501, copied from the atlas windows, which are 1,002 bp. Seven hours of
    # 40B scoring produced 7,626 identical zeros and exited 0. Checked here, before the model runs.
    for i in range(len(vids)):
        o = int(voff[i])
        if not (0 <= o < len(ref[i])):
            raise SystemExit("var_off %d out of range for a %d-bp window (%s)"
                             % (o, len(ref[i]), vids[i]))
        if ref[i][o].upper() == alt[i][o].upper():
            d = [k for k, (x, y) in enumerate(zip(ref[i], alt[i])) if x != y]
            raise SystemExit(
                "var_off does not point at the substitution in %s: ref and alt both carry %r at "
                "offset %d, so every delta would be exactly zero. The bases differ at %s. Fix the "
                "panel builder's var_off, do not adjust it here."
                % (vids[i], ref[i][o], o, d if len(d) < 5 else "%d positions" % len(d)))

    def flush(pairs):
        d = dict(scored); d.update(pairs)
        pl.DataFrame({"variant_id": list(d), col: list(d.values())}).write_parquet(out)

    if not vids:
        print(f"  nothing to score ({len(scored)} already in {out})", flush=True)
        return

    def score_batch(prefixes):
        ids = [torch.tensor(tok.tokenize(p), dtype=torch.int) for p in prefixes]
        Lmax = max(len(t) for t in ids)
        # The pad below is LEFT padding with token id 0, which is a real byte in Evo 2's vocabulary
        # rather than a designated pad token, and no attention mask is passed. On a variable-length
        # panel real tokens would convolve over those bytes and every row in the batch would be
        # silently altered. It does not happen here: extract_windows_local.py fixes var_off at a
        # constant HALF and drops any window whose length differs, so every prefix in every batch is
        # the same length and the padding is zero-width. This assertion is what makes that a
        # guarantee instead of a coincidence, and it is why the FP8 batch-composition ablation
        # (bit-identical across batch sizes) is sound but says nothing about padding.
        assert all(len(t) == Lmax for t in ids), (
            "variable-length prefixes in one batch: left padding with token id 0 and no attention "
            "mask would corrupt every row. Pass an attention mask before scoring a ragged panel.")
        batch_t = torch.stack([torch.nn.functional.pad(t, (Lmax - len(t), 0)) for t in ids]).to("cuda:0")
        with torch.inference_mode():
            logits, _ = sh(batch_t)                   # [B, L, 512]
        return torch.log_softmax(logits[:, -1, :].float(), dim=-1).cpu().numpy()

    new = {}; t0 = time.time()
    for s in range(0, len(vids), batch):
        e = min(s + batch, len(vids))
        prefixes = [ref[i][:int(voff[i])].upper() for i in range(s, e)]
        lp = score_batch(prefixes)
        for j, i in enumerate(range(s, e)):
            rb, ab = ref[i][int(voff[i])].upper(), alt[i][int(voff[i])].upper()
            new[vids[i]] = -float(lp[j][ord(ab)] - lp[j][ord(rb)])
        flush(new)                                    # checkpoint after every streaming forward
        print(f"  {e}/{len(vids)}  {e/(time.time()-t0):.2f}/s", flush=True)

    flush(new)
    # A panel of one distinct score is not a result, it is a broken run that happened to exit 0.
    # Nothing downstream can tell the difference -- an AUROC on a constant column is 0.5 and looks
    # like a finding -- so the failure has to be raised here, where the cause is still visible.
    vals = set(new.values())
    if len(new) > 1 and len(vals) == 1:
        raise SystemExit(
            "degenerate output: all %d new scores are %r. The model was asked the same question "
            "twice per variant; check var_off and the ref/alt bases before rescoring."
            % (len(new), vals.pop()))
    print(f"  wrote {out}: {len(scored)+len(new)} variants ({len(new)} new) in {time.time()-t0:.0f}s", flush=True)

    if oracle:
        orc = pl.read_parquet(oracle)
        d = pl.DataFrame({"variant_id": list(new), "local": list(new.values())}).join(orc, on="variant_id").drop_nulls()
        if d.height:
            r = float(np.corrcoef(d["local"].to_numpy(), d[col].to_numpy())[0, 1])
            print(f"  VALIDATION vs NIM oracle: n={d.height}  Pearson r={r:.3f}  "
                  f"(>0.99 = streaming FP8 path matches hosted 40B)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", help="single windows parquet")
    ap.add_argument("--out", help="single output parquet")
    ap.add_argument("--manifest", help="TSV of 'in_parquet<TAB>out_parquet' lines; model loads once, all panels scored")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--oracle", help="NIM-API scores parquet to validate against (single-panel only)")
    ap.add_argument("--limit", type=int, default=0, help="only score N variants (for validation)")
    # Exposed so a capacity ladder can be built entirely through THIS readout. The deposited 1B and
    # 7B scores for the splicing and ClinVar panels come from the BioNeMo window-sum scorer, so a
    # 40B number from here cannot be read against them as a capacity rung: the readout moves too.
    ap.add_argument("--model", default="evo2_40b",
                    help="checkpoint to stream: evo2_40b (default), evo2_7b, evo2_1b_base")
    a = ap.parse_args()

    panels = []
    if a.manifest:
        with open(a.manifest) as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                ip, op = ln.split("\t")[:2]
                panels.append((ip, op))
    else:
        if not (a.inp and a.out):
            ap.error("provide --in/--out or --manifest")
        panels.append((a.inp, a.out))

    print(f"building {a.model} on CPU ; {len(panels)} panel(s) to score ...", flush=True)
    m, sh = build_cpu_model(a.model)
    patch_streaming(sh)
    tok = m.tokenizer

    for ip, op in panels:
        print(f"\n===== panel {ip} -> {op} =====", flush=True)
        score_panel(ip, op, a.batch, tok, sh, limit=a.limit, oracle=a.oracle, model_name=a.model)


if __name__ == "__main__":
    main()
