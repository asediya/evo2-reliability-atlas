# -*- coding: utf-8 -*-
"""Build Additional file 2 (the code deposit) deterministically from the working tree.

The manuscript's Availability statement commits the deposit to contain "all scripts under src/ and
tools/, together with the recompute-layer JSON files that carry all 211 traced published
values", under the MIT License. This assembles exactly that, so the zip cannot silently drift from what the paper
promises (an earlier hand-built zip was missing new scripts and every JSON). Sorted entries and a
fixed timestamp make the archive byte-reproducible.

    python tools/build_code_zip.py
    -> reports/submission/Additional_file_2_Code.zip
"""
import glob
import os
import sys
import zipfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# reports/submission/ is frozen while the rewrite is in progress, so the destination is an
# argument. It also follows $CCS_SUBMISSION_OUT, which is what every other stage of
# finalize_submission.sh honours. Leaving that out made the documented rebuild command write into
# the frozen v1 directory and fail check_submission_frozen on its own output: the one gate whose
# whole job is to prove the previous submission has not moved was broken by running the build.
OUT = (sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(".zip")
       else os.path.join(os.environ.get("CCS_SUBMISSION_OUT", "reports/submission"),
                         "Additional_file_2_Code.zip"))
FIXED = (2026, 1, 1, 0, 0, 0)   # constant mtime -> reproducible archive


def collect():
    files = set()
    for pat in ("src/**/*.py", "tools/**/*.py",
                # the analysis layer written for this version: the new panels, the coverage
                # accounting and the assembly. Absent, a reader cannot rebuild Results 4, 5 or 7.
                "analyses/scripts/**/*.py",
                # the deposited glmtrust package: source, tests, examples, benchmarks, docs, packaging
                "glmtrust/**/*.py", "glmtrust/**/*.md", "glmtrust/**/*.toml",
                "glmtrust/**/*.cff", "glmtrust/**/*.yml",
                # The benchmark fixture (119 kB): without it reproduce_paper_trust_layer.py
                # cannot run from the deposit, because the study score tree is not deposited.
                "glmtrust/benchmarks/fixtures/*.parquet",
                # and the trust-layer summary it checks against, so the benchmark also runs
                # from a glmtrust checkout outside this archive
                "glmtrust/benchmarks/fixtures/*.json",
                # six of eight figures could not be rebuilt from the deposit because
                # the assets/ image tree was absent and named in no manifest. Only these ten files
                # are read by any builder -- nine PhyloPic silhouettes and one structure crop, 1.1 MB
                # against the 51 MB full tree, so the deposit gains reproducibility without the bulk.
                "assets/silhouettes/*.png",
                # the archive README references five paths the archive did not
                # contain, and one of them is where the README says the CC BY 4.0 attribution for
                # four PhyloPic silhouettes is discharged. A licence obligation cannot be satisfied
                # by a file the reader does not receive. 151 kB in total.
                "assets/silhouettes/credits.json",
                "docs/REPRODUCING.md", "docs/img/banner.png", "docs/img/concept.png",
                "notebooks/quickstart.ipynb",
                # the recipe of the container the Evo 2 scores were produced in, and the download
                # script for the human scorer releases behind Additional file 5
                "container/*", "tools/fetch_human_panel.sh",
                ):
        files.update(glob.glob(pat, recursive=True))
    files.update(glob.glob("reports/*.json"))          # recompute layers holding published numbers
    # The recompute layer for this version's new arms. Every number in Results 4, 5 and 7 is read
    # from one of these, and without them the analysis scripts above have nothing to check against:
    # 224 kB against a 4 MB archive, so there is no size argument for leaving them out.
    files.update(glob.glob("analyses/results/*.json"))
    files.update(glob.glob("analyses/results/*.parquet"))
    files.update(glob.glob("reports/*.tsv"))           # the cattle sample table the Methods cite
    files.update(glob.glob("reports/*.parquet"))       # small recompute parquets (fig4_pervariant, etc.) — R4
    # 40 scripts write their results to logs/, which was absent from the deposit, so
    # Notes S5 and S6 quoted values a reader could not open. Ship the log outputs that carry reported
    # numbers, and the superseded-artifact directory the README and COMPILED_RESULTS point at.
    # those two globs also swept in material that is not an analysis
    # output -- working notes rather than results. (An earlier tree held reports/_superseded/; it is
    # this manuscript plus an old abstract, ~800 kB, and those drafts state older declarations than
    # the manuscript now makes, which an editor would read as a discrepancy.
    #
    # Excluded by name, not by directory. logs/*.md is present so Notes S5 and S6
    # would resolve, and the other 34 files there are genuine analysis output; that directory is
    # the COMPILED_RESULTS snapshot that README.md and COMPILED_RESULTS.md point at by name to say "do
    # not quote it", so dropping the directory would break two live pointers.
    INTERNAL = ("logs/DISCUSSION_REWRITE_BRIEF.md",)
    files.update(f for f in glob.glob("logs/*.md")
                 if f.replace("\\", "/") not in INTERNAL)
    for top in ("README.md", "LICENSE", "glmtrust/LICENSE",
                "requirements.txt",                                    # R2: README's install step needs it
                ".zenodo.json",                                        # Zenodo release metadata
                "FIGURES.md",                                          # E107: figure-to-script manifest
                "reports/COMPILED_RESULTS.md", "reports/tables.md",    # R2: README references these directly
                                # E66: the manuscript-integrity tools need these to run without tripping their guards
                "reports/supplementary_prose.md", "reports/supplementary_tables.md",
                "reports/DATA_MANIFEST.md"):                           # R6: what data/ is needed and where it lives
        if os.path.exists(top):
            files.add(top)
    # Exclusions:
    #  - __pycache__/.venv: never ship caches
    #  - _superseded_*: shipped-but-superseded figure code, ambiguous provenance (E86)
    #  - repro_state.json: the reproduction log is the stale 3,506-variant record and carries internal
    #    planning notes; Note S7 documents the harness instead (E48/E61/E65)
    #  - PROJECT_ROADMAP.md: planning doc with positioning strategy + external agent-memory refs (E87)
    #  - figure-screening/QC JSONs: internal design records that describe earlier figure versions, carry
    #    a working-directory absolute path, and are not authoritative for the submitted figures
    #    (E99/E100/E101/E106/E107); FIGURES.md documents the live builders instead
    SCREEN = ("fig4_panelA_screening.json", "fig4_panelA_research.json", "fig3_fleet_result.json",
              "fig6_screen_raw.json", "fig2_rework_spec.json")
    # the submitted manuscript and supplementary were shipped INSIDE the code deposit,
    # which guarantees a future divergence between the two copies. The in-deposit cross-references
    # that needed them are satisfied by COMPILED_RESULTS.md, tables.md and supplementary_tables.md.
    # tools/ shipped 47 files including a bundled snpEff distribution and the
    # submission-tooling directory, neither of which is analysis code.
    # three reports/ artefacts are read by no deposited script, written by no
    # deposited script and quoted in no text, table or figure legend (verified by grep over
    # src/, tools/, glmtrust/ and the three prose sources). The deposit's stated principle is that
    # these files ARE the recompute layer the builders read, so shipping orphans weakens it.
    # fig5_data.json in particular invites a reader to check Figure S5 against the wrong file --
    # its live builder reads fig5_reach.json.
    DROP = ("reports/manuscript.md", "reports/gb_supplementary.md",
            # superseded writer: emits Tables S29-S32 under an earlier numbering that
            # conflicts with the submitted S29-S31; the submitted document is its own
            # authority (tools/export_from_submission.py), so this must not ship
            "analyses/scripts/build_new_supp_tables.py",
            "reports/_recon_pervariant_trust_8192.parquet",
            "reports/_recon_abstain_dissect.parquet",
            "reports/fig5_data.json")

    # Packaging output directories. `python -m build` writes a verbatim copy of every module into
    # glmtrust/build/lib/glmtrust/, and because collect() globs the filesystem rather than asking git,
    # those copies were swept into the deposit even though glmtrust/.gitignore lists build/. A referee
    # unzipping it found two audit.py files with no way to tell which one produced the results, and
    # the build copy is a snapshot that goes stale the moment the source is edited. Anything gitignored
    # is by definition not source and must not ship.
    BUILD_DIRS = ("/build/", "/dist/", ".egg-info/")

    def keep(f):
        return ("__pycache__" not in f and not f.startswith(".venv")
                and not any(d in "/" + f for d in BUILD_DIRS)
                and "_superseded_" not in f          # superseded figure code; the _superseded/ dir stays
                and f not in DROP
                and not f.startswith("tools/se52/")
                # tools/snpEff/ is the same third-party distribution under its other directory
                # name. Twelve Python-2 scripts were shipping: they fail `compileall` on any
                # modern interpreter, they are not analysis code, and redistributing them under
                # this deposit's blanket MIT with no attribution is not ours to do. The comment
                # above about removing "a bundled snpEff distribution" described a fix that had
                # only ever been applied to se52/.
                and not f.startswith("tools/snpEff/")
                # Note S7 names consistency_check.py as one of three legs certifying
                # that the manuscript carries no contradictions, and calls that certification
                # "deposited". The blanket tools/manuscript/ exclusion meant a referee could not run the very
                # check the paper cites as its guarantee. The rest of that plugin is authoring
                # toolchain and stays out; the cited checker ships.
                and not (f.startswith("tools/manuscript/")
                         and not f.endswith("scripts/consistency_check.py"))
                and not f.endswith("reports/repro_state.json")
                and not any(f.endswith("reports/" + s) for s in SCREEN))
    return sorted(f.replace("\\", "/") for f in files if keep(f.replace("\\", "/")))


def main():
    files = collect()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            info = zipfile.ZipInfo(f, date_time=FIXED)
            info.compress_type = zipfile.ZIP_DEFLATED
            # 26 deposited builders read data/, write a deposited artefact, and carry
            # no exit/raise/assert, so running one without the (undeposited) data/ tree could write a
            # degenerate file over the very artefact the paper's numbers are checked against -- at
            # exit 0, with nothing shown. Shipping the recompute layer read-only turns that silent
            # corruption into a loud PermissionError. A reader who genuinely wants to regenerate an
            # artefact clears the bit first, which is a deliberate act rather than an accident.
            # The bit covers .md as well as .json and .parquet: without it, running the
            # documented `build_supplementary.py` on a clean extraction would rewrite
            # reports/supplementary_tables.md in "w" mode with its own tables only, destroying the
            # appended S14-S24 -- including S22, S23 and S24, the three tables carrying the Abstract's
            # human-scale claim -- and exit 0.
            # logs/ is inside the rule because it holds 34 shipped, MANIFESTED notes: without it,
            # running build_decision_panel.py on a clean extract would overwrite logs/decision_panel.md
            # before the read-only bit on reports/decision_panel.json stopped it, so the guard would
            # fire one file too late and the archive would fail its own manifest check.
            # ".tsv" is included because reports/cattle_sample_table.tsv is a deposited artefact
            # with a producing script (src/ccs/build_cattle_sample_table.py writes exactly that
            # path). The predicate is an extension whitelist rather than a directory rule, because a
            # directory rule would also make the eight reports/figures/*.png read-only, which they
            # should not be.
            protected = ((f.startswith("reports/") or f.startswith("analyses/results/")
                          or f.startswith("logs/"))
                         and f.rsplit(".", 1)[-1] in ("json", "parquet", "md", "tsv"))
            info.external_attr = (0o444 if protected else 0o644) << 16
            if protected:
                info.external_attr |= 0x01                  # FILE_ATTRIBUTE_READONLY, for Windows
            with open(f, "rb") as fh:
                z.writestr(info, fh.read())
    n_py = sum(f.endswith(".py") for f in files)
    n_json = sum(f.endswith(".json") for f in files)
    print("wrote %s" % OUT)
    print("  %d files total: %d .py (src+tools+glmtrust), %d recompute-layer .json, plus docs/packaging"
          % (len(files), n_py, n_json))
    print("  %.0f KB" % (os.path.getsize(OUT) / 1024))
    # sanity: the scripts behind the newest table, the per-variant parquet the trust-layer tables are
    # built from, and the data manifest must all be present.
    #
    # The glmtrust audit modules are asserted here because their absence is exactly the failure this
    # list exists to catch: the manuscript names glmtrust as its deliverable, and a zip built in a
    # worktree pinned to a commit that predates a module matches every glob above and still lacks
    # that module. A glob cannot notice an absence; an assertion can.
    for must in ("src/ccs/analyze_2x2_readout.py", "src/ccs/build_supplementary_extra.py",
                 "reports/readout_2x2_decomposition.json", "reports/fig4_pervariant.parquet",
                 "requirements.txt", "reports/DATA_MANIFEST.md",
                 "glmtrust/src/glmtrust/audit.py",
                 "glmtrust/src/glmtrust/card.py",
                 "glmtrust/src/glmtrust/delong.py",
                 "glmtrust/tests/test_audit.py",
                 "glmtrust/tests/test_card.py",
                 "glmtrust/tests/test_delong.py",
                 "tools/export_from_submission.py",
                 "tools/capture_scoring_environment.py",
                 ".zenodo.json"):
        assert must in files, ("MISSING from deposit: %s\n"
                               "  If this is a glmtrust module, check which working tree you are "
                               "building from: `git worktree list`." % must)
    n_parq = sum(f.endswith(".parquet") for f in files)
    print("  verified: readout JSON, fig4_pervariant.parquet, requirements.txt, DATA_MANIFEST.md present"
          " (%d recompute parquets in all)" % n_parq)


if __name__ == "__main__":
    main()
