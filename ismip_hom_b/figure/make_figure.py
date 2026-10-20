#!/usr/bin/env python3

"""Build the ISMIP-HOM experiment B figure (Appendix D, Fig. D1).

The figure is the pgfplots source used in the paper (ismip.tex), compiled in a
standalone document. It reads the CSV files of this folder:
  exp_b_unified_<L>km_identity_lbfgs_ref.csv   Elmer/Ice full-Stokes ('oga')
  exp_b_unified_<L>km_identity_lbfgs_igm.csv   IGM surface speed
The PDF is written next to this script. Requires pdflatex (with pgfplots and
sansmath) or tectonic.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

TIKZ = "ismip.tex"
OUTPUT = "ismip_hom_b.pdf"

WRAPPER = r"""\documentclass[border=2pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{iftex}
\ifPDFTeX\else\usepackage{newunicodechar}\newunicodechar{°}{\textdegree}\fi
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
\usepackage{sansmath}
\definecolor{blue_t}{RGB}{41,128,185}
\begin{document}
\input{%s}
\end{document}
"""


def compile_latex(build_dir, tex_name):
    if shutil.which("pdflatex") is not None:
        command = ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_name]
    elif shutil.which("tectonic") is not None:
        command = ["tectonic", tex_name]
    else:
        sys.exit(
            "Neither pdflatex nor tectonic was found. Install a LaTeX distribution "
            "with pgfplots and sansmath, or: conda install -c conda-forge tectonic"
        )
    result = subprocess.run(command, cwd=build_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout[-3000:], result.stderr[-3000:])
        sys.exit(f"{command[0]} failed on {TIKZ}")


def main():
    here = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory() as tmp:
        build_dir = Path(tmp)
        data_dir = build_dir / "data"
        data_dir.mkdir()
        for path_csv in here.glob("*.csv"):
            shutil.copy(path_csv, data_dir)
        shutil.copy(here / TIKZ, build_dir)
        (build_dir / "figure.tex").write_text(WRAPPER % TIKZ, encoding="utf-8")
        compile_latex(build_dir, "figure.tex")
        shutil.copy(build_dir / "figure.pdf", here / OUTPUT)
    print(f"Written {here / OUTPUT}")


if __name__ == "__main__":
    main()
