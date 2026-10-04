"""Week-1 NT-500M probe: verify GPU load, introspect the 6-mer tokenizer, compute a
masked-marginal LLR for a synthetic ref/alt variant, and measure latency + VRAM.
This establishes the exact scoring mechanics for score_nt.py."""
import os, time, torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

MID = "InstaDeepAI/nucleotide-transformer-v2-500m-multi-species"
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")

tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(
    MID, trust_remote_code=True, torch_dtype=torch.float32
).to("cuda").eval()
print("weights VRAM MB:", round(torch.cuda.memory_allocated() / 1e6))

# --- tokenizer introspection ---
print("mask token:", tok.mask_token, tok.mask_token_id, "| cls:", tok.cls_token, tok.cls_token_id,
      "| pad:", tok.pad_token_id, "| vocab:", tok.vocab_size)
seq = "ACGT" * 250  # 1000 bp
enc = tok(seq, return_tensors="pt")
ids = enc["input_ids"][0]
print("1kb -> n_tokens:", ids.shape[0])
print("first 6 token strs:", tok.convert_ids_to_tokens(ids[:6].tolist()))
# a single 6-mer token id
mer = tok.convert_tokens_to_ids("ACGTAC")
print("token id of 'ACGTAC':", mer)

# --- masked-marginal LLR on a synthetic variant ---
# put variant in the centre; build a window that is a clean multiple of 6.
W = 1002  # 167 hexamers
half = W // 2
import random
random.seed(0)
bases = "ACGT"
left = "".join(random.choice(bases) for _ in range(half))
right = "".join(random.choice(bases) for _ in range(W - half - 1))
ref_center, alt_center = "A", "T"
ref_win = left + ref_center + right
alt_win = left + alt_center + right
var_off = len(left)                 # 0-based offset of the variant base
tok_idx_seq = var_off // 6          # which 6-mer (in raw sequence, pre-CLS)

def masked_llr(ref_seq, alt_seq, var_off):
    enc = tok(ref_seq, return_tensors="pt").to("cuda")
    ids = enc["input_ids"].clone()
    # +1 for CLS at position 0 if present
    offset = 1 if tok.cls_token_id is not None and ids[0, 0] == tok.cls_token_id else 0
    tpos = var_off // 6 + offset
    ref_mer = ref_seq[(var_off // 6) * 6:(var_off // 6) * 6 + 6]
    alt_mer = alt_seq[(var_off // 6) * 6:(var_off // 6) * 6 + 6]
    ref_id = tok.convert_tokens_to_ids(ref_mer)
    alt_id = tok.convert_tokens_to_ids(alt_mer)
    ids[0, tpos] = tok.mask_token_id
    with torch.inference_mode():
        logits = model(input_ids=ids, attention_mask=enc["attention_mask"]).logits
    logp = torch.log_softmax(logits[0, tpos].float(), dim=-1)
    return (logp[alt_id] - logp[ref_id]).item(), ref_mer, alt_mer, ref_id, alt_id

llr, rm, am, rid, aid = masked_llr(ref_win, alt_win, var_off)
print(f"masked-marginal LLR(alt-ref): {llr:.4f}  ref_mer={rm}({rid}) alt_mer={am}({aid})")

# --- latency / VRAM ---
enc = tok(ref_win, return_tensors="pt").to("cuda")
torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    for _ in range(5):
        model(**enc)
    torch.cuda.synchronize(); t0 = time.time(); N = 40
    for _ in range(N):
        out = model(**enc)
    torch.cuda.synchronize()
dt = (time.time() - t0) / N
print(f"per-forward({W}bp): {dt*1000:.1f} ms => ~{1/dt:.0f} win/s | peak VRAM MB:",
      round(torch.cuda.max_memory_allocated() / 1e6), "| logits", tuple(out.logits.shape))
print("SMOKE_OK")
