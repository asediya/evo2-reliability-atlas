"""Download the Myotis lucifugus white-nose-syndrome WGS reads (BioProject PRJNA624023, ~252 GB, 20
FASTQ files) from ENA, resumably. Skips already-complete files; resumes partial ones via HTTP Range.
Updates the JARVIS dashboard (logs/status/bat.status + events) as it goes. Runs detached; safe to
re-launch (it continues where it left off).
"""
import urllib.request, json, os, time

OUT = "data/raw/bat/reads"
ST = "logs/status/bat.status"
EV = "logs/status/events.log"
os.makedirs(OUT, exist_ok=True)


def st(m):
    with open(ST, "w") as f: f.write(m + "\n")
def ev(m):
    with open(EV, "a", encoding="utf-8") as f: f.write(f"[{time.strftime('%H:%M:%S')}] supervisor: {m}\n")


def main():
    api = ("https://www.ebi.ac.uk/ena/portal/api/filereport?accession=PRJNA624023"
           "&result=read_run&fields=fastq_ftp,fastq_bytes&format=json&limit=0")
    d = json.load(urllib.request.urlopen(urllib.request.Request(api, headers={"User-Agent": "Mozilla/5.0"}), timeout=90))
    files = []
    total = 0
    for r in d:
        us = (r.get("fastq_ftp") or "").split(";")
        bs = (r.get("fastq_bytes") or "").split(";")
        for u, b in zip(us, bs):
            if not u:
                continue
            url = u if u.startswith("http") else "https://" + u
            sz = int(b) if b.strip().isdigit() else 0
            files.append((url, sz)); total += sz
    ev(f"bat download: {len(files)} FASTQ files, {total/1e9:.0f} GB total")

    done = 0
    for i, (url, sz) in enumerate(files, 1):
        fn = os.path.join(OUT, url.split("/")[-1])
        have = os.path.getsize(fn) if os.path.exists(fn) else 0
        if sz and have >= sz:
            done += have; continue
        st(f"RUNNING | bat reads {done/1e9:.0f}/{total/1e9:.0f} GB  (file {i}/{len(files)})")
        hdr = {"User-Agent": "Mozilla/5.0"}
        if have:
            hdr["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=120) as r, \
                    open(fn, "ab" if have else "wb") as f:
                last = time.time()
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk); done += len(chunk)
                    if time.time() - last > 20:
                        st(f"RUNNING | bat reads {done/1e9:.1f}/{total/1e9:.0f} GB  (file {i}/{len(files)})")
                        last = time.time()
        except Exception as e:
            ev(f"bat file {i} ({os.path.basename(fn)}) error: {e} — will resume on relaunch")
    st(f"DONE | bat reads downloaded ({done/1e9:.0f} GB, {len(files)} files)")
    ev(f"bat WGS reads download complete ({done/1e9:.0f} GB) — ready for mapping to Myoluc2.0")


if __name__ == "__main__":
    main()
