# Med-Self-Preference LaTeX manuscript

`main.tex` is a self-contained, upload-ready LaTeX manuscript. It uses only
standard TeX packages and contains the references directly, so no `.bib` file
or local data files are required.

The canonical Markdown experiment record is
[`../../experiment.md`](../../experiment.md). Keep substantive result changes
synchronized between that document and `main.tex`.

Compile locally with:

```bash
tectonic main.tex
```

Before submission, verify the author list, affiliations, and corresponding
author metadata. The manuscript includes the completed 1,200-judgment
identity-revealed experiment and its paired comparison with the blinded
condition, plus the fixed-effects token-length sensitivity for model scoring
and matched self-preference.
