"""Sample biallelic ACGT SNVs from the in-hand cattle pvar files into a coordinate-sorted VCF
for snpEff consequence annotation. Also emits a parquet sidecar (variant_id, chrom, pos, ref, alt, af).
Low-memory: duckdb streams the ~30M-variant pvars and takes a uniform SAMPLE.

  python sample_vcf.py --pvar-dir geno_pgen --n 3000000 --out-vcf sample.vcf --out-parq sample_vars.parquet
"""
import argparse, glob, os
import duckdb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pvar-dir", required=True)
    ap.add_argument("--n", type=int, default=3_000_000)
    ap.add_argument("--out-vcf", required=True)
    ap.add_argument("--out-parq", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=16)
    a = ap.parse_args()

    pvars = sorted(glob.glob(os.path.join(a.pvar_dir, "Chr*.pvar")))
    assert pvars, "no pvar files"
    files = "[" + ",".join("'" + p.replace("\\", "/") + "'" for p in pvars) + "]"
    con = duckdb.connect()
    con.execute(f"SET threads={a.threads}")
    cols = ("{'CHROM':'VARCHAR','POS':'BIGINT','ID':'VARCHAR','REF':'VARCHAR',"
            "'ALT':'VARCHAR','FILTER':'VARCHAR','INFO':'VARCHAR'}")
    # AF parsed from INFO (…;AF=x;…); uniform reservoir sample of clean biallelic SNVs
    con.execute(f"""
        CREATE TEMP TABLE s AS
        SELECT CHROM, POS, ID, REF, ALT,
               TRY_CAST(regexp_extract(INFO, 'AF=([0-9.eE-]+)', 1) AS DOUBLE) AS af
        FROM read_csv({files}, delim='\t', comment='#', columns={cols}, ignore_errors=true)
        WHERE length(REF)=1 AND length(ALT)=1
          AND REF IN ('A','C','G','T') AND ALT IN ('A','C','G','T')
        USING SAMPLE {a.n} ROWS (reservoir, {a.seed});
    """)
    n = con.execute("SELECT count(*) FROM s").fetchone()[0]
    print(f"sampled {n} biallelic ACGT SNVs", flush=True)

    # parquet sidecar (variant_id = CHROM_POS_REF_ALT to match Evo2/phyloP joins)
    con.execute(f"""
        COPY (SELECT (CHROM || '_' || POS || '_' || REF || '_' || ALT) AS variant_id,
                     CHROM AS chrom, POS AS pos, REF AS ref, ALT AS alt, af
              FROM s ORDER BY CHROM, POS)
        TO '{a.out_parq.replace(os.sep, "/")}' (FORMAT parquet);
    """)
    # minimal VCF, coordinate-sorted; ID carries our variant_id for round-trip
    con.execute("""
        CREATE TEMP TABLE v AS
        SELECT CHROM, POS, (CHROM || '_' || POS || '_' || REF || '_' || ALT) AS ID, REF, ALT
        FROM s ORDER BY CHROM, POS;
    """)
    rows = con.execute("SELECT CHROM, POS, ID, REF, ALT FROM v").fetchall()
    with open(a.out_vcf, "w") as f:
        f.write("##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")
        for c, p, i, r, al in rows:
            f.write(f"{c}\t{p}\t{i}\t{r}\t{al}\t.\t.\t.\n")
    print(f"wrote {a.out_vcf} ({len(rows)} records) + {a.out_parq}", flush=True)


if __name__ == "__main__":
    main()
