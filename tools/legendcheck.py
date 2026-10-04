"""Check each figure legend in 01_Manuscript_GigaScience_v2.0.docx against the plate it describes.

    python tools/legendcheck.py /path/to/upload_folder      # exit 0 = clean, 1 = defect

Neither existing suite crosses this boundary. verify_package.py checks the manuscript against the
deposited aggregates; pdfcheck.py and inkunder.py check the plates against themselves. A legend
that names a panel the figure no longer draws, or prints a value the figure contradicts, passes
both. That gap has bitten this package before: Note S43 described panels Figures 3 and 5 had
stopped drawing, and it was found by reading, not by a gate.

Two checks, both STRUCTURAL rather than existential. The distinction decides whether a gate is
worth having: "does this string appear anywhere in the corpus" gets cheaper as the corpus grows
and eventually passes regardless of truth, whereas "are these two sets equal" and "is this value
a near miss of that one" do not get cheaper with size.

  PANELS   the "(a) (b) (c)" markers in the legend against the bold lower-case letters actually
           drawn on the plate at panel size. A set difference either way is a real defect: a
           legend describing a deleted panel, or a panel nothing is said about.

  NUMBERS  every number the legend prints, against every number the plate prints. Exact match is
           corroboration. No match at all is ordinary legend prose and is not reported. The
           reportable case is the NEAR MISS -- the plate carries a number at the same printed
           precision differing only in the last place or two. That is the shape of a
           transcription slip; an unrelated coincidence is not near.

WHAT IT CANNOT DO, stated so a passing run is not over-read:

  * It cannot see DUPLICATION. A legend sentence and a plate key stating the same fact agree
    with each other, so this reports corroboration where the truth is redundancy. Contradiction
    and redundancy are different questions and only the first is asked here.

  * It cannot certify itself. nums() asserts that it is handed a list, because a bare str
    iterates CHARACTERS and would turn the legend's "numbers" into the digits 0-9. Trust a run
    only once it has been validated by injected faults: put a ghost panel in a legend and a
    last-place typo in a value, and confirm both fail.
"""
import os, re, sys, zipfile
import pymupdf
from xml.etree import ElementTree as ET

W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
PAIRS = [(i, "fig%d.pdf" % i) for i in range(1, 11)]   # the ten figures, by their upload names
# the size the ten plates set their panel letters in, read from the house style so the check follows it
STYLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "src", "ccs", "style_main.py")
PANEL_PT = float(re.search(r"^PANEL, AXIS, TICK, ANNOT, FLOOR = ([\d.]+)", open(STYLE).read(), re.M).group(1))
NUM = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")


def legends(docx):
    z = zipfile.ZipFile(docx)
    body = ET.fromstring(z.read('word/document.xml')).find(W + 'body')
    out = {}
    for p in body:
        if p.tag != W + 'p': continue
        t = "".join(e.text or "" for e in p.iter(W + 't'))
        m = re.match(r'^\s*Figure (\d+)\.', t)
        if m and len(t) > 200: out[int(m.group(1))] = t.strip()
    return out


def plate(path):
    """Every span on the plate, plus the bold single letters that mark its panels."""
    pg = pymupdf.open(path)[0]
    spans, letters = [], set()
    for b in pg.get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                t = s["text"].strip()
                if not t: continue
                spans.append(t)
                # a panel letter is a lone lower-case letter set in bold Arial at PANEL size (style_main.save
                # allows nothing else at that size)
                if re.fullmatch(r"[a-z]", t) and "Bold" in s["font"] and abs(s["size"] - PANEL_PT) < 0.05:
                    letters.add(t)
    return spans, letters


def nums(strings):
    """Numeric literals in a LIST of strings. Passing a bare str here iterates its CHARACTERS and
    silently yields the digits 0-9, which gives every figure a plausible but meaningless
    corroboration count and misses an injected transcription error. Guarded rather than commented."""
    assert not isinstance(strings, str), "nums() takes a list of strings, not one string"
    out = {}
    for s in strings:
        for m in NUM.finditer(s):
            raw = m.group(1).rstrip(",")          # a trailing comma is punctuation, not a separator
            if raw: out.setdefault(raw.replace(",", ""), raw)
    return out


def near(a, b):
    """True if a and b are the same quantity mistyped: same precision, last places differ."""
    if a == b: return False
    da, db = (a.split(".") + [""])[1], (b.split(".") + [""])[1]
    if len(da) != len(db) or len(da) == 0: return False        # same printed precision only
    try: fa, fb = float(a), float(b)
    except ValueError: return False
    if fa == 0 or fb == 0: return False
    step = 10.0 ** -len(da)
    d = abs(fa - fb)
    return 0 < d <= 3 * step + 1e-12                            # within three of the last place


def main(sub):
    L = legends(f"{sub}/01_Manuscript_GigaScience_v2.0.docx")
    bad = 0
    print(f"{'fig':>4}  {'panels legend':>13}  {'panels plate':>12}  {'nums corrob.':>12}  near-miss")
    for n, f in PAIRS:
        leg = L[n]
        spans, drawn = plate(f"{sub}/{f}")
        said = set(re.findall(r"\((\w)\)", leg)) & set("abcdefgh")
        pn, ln_ = nums(spans), nums([leg])
        corrob = sorted(set(pn) & set(ln_), key=float)
        miss = [(ln_[a], pn[b]) for a in ln_ if a not in pn for b in pn if near(a, b)]
        flag = ""
        if said != drawn:
            flag += f"  PANELS legend={sorted(said)} plate={sorted(drawn)}"; bad += 1
        if miss:
            flag += "  NEAR-MISS " + ", ".join(f"legend {a} vs plate {b}" for a, b in miss); bad += 1
        print(f"{n:>4}  {''.join(sorted(said)):>13}  {''.join(sorted(drawn)):>12}  {len(corrob):>12}{flag}")
    print()
    print("FAIL" if bad else "PASS", f"— {bad} figure(s) with a legend/plate discrepancy")
    print("NOT checked: prose claims with no number, values the legend states that the plate")
    print("never prints (normal), and whether a corroborated number means the same thing in both.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
