"""Build overleaf.zip: a self-contained copy of the report for Overleaf.

Contains main.tex, sections/, bib.bib, the .sty files and the figures that the
report actually includes (copied into figures/). Upload it with
Overleaf -> New Project -> Upload Project.   Usage:  python make_overleaf.py
"""

import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent
REPORT, FIGS = ROOT / "report", ROOT / "figures"

tex_files = [REPORT / "main.tex", *sorted((REPORT / "sections").glob("*.tex"))]
used = set()
for f in tex_files:
    used |= set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", f.read_text(encoding="utf-8")))

with zipfile.ZipFile(ROOT / "overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in tex_files + [REPORT / "bib.bib", REPORT / "theme.sty", REPORT / "shortcuts.sty"]:
        z.write(f, f.relative_to(REPORT))
    for name in sorted(used):
        path = FIGS / name
        if not path.exists():
            raise FileNotFoundError(f"figure used in the report but missing: {path}")
        z.write(path, f"figures/{name}")
        print("figure:", name)
print("written", ROOT / "overleaf.zip")
