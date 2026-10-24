# Paper draft (IEEE conference template)

- `main.tex`: the paper (IEEEtran, conference mode).
- `supplement.tex`: supplementary material: split sizes, the full feature list, all baselines and models, interval calibration, dropped observations, the new-league check, and four extra figures.
- `references.bib`: the bibliography. The related-work entries were checked against publisher or repository pages.
- `generated/`: **do not edit by hand.** `numbers.tex` holds every number quoted in the text, and the `tab_*.tex` files hold the result tables. They are rebuilt from `results/` by:

  ```bash
  ~/miniforge3/envs/football/bin/python scripts/make_paper_assets.py
  ```

  `./run_all.sh` also runs this step at the end.

- Figures are read from `../figures/` (see `\graphicspath` in `main.tex`).

## Compile locally

The `football` environment includes `tectonic`, a small LaTeX engine. It downloads the needed LaTeX packages on first use.

```bash
cd paper && ~/miniforge3/envs/football/bin/tectonic main.tex && ~/miniforge3/envs/football/bin/tectonic supplement.tex
```

`./run_all.sh` also compiles both PDFs at the end.

## Compile on Overleaf

1. Upload the `paper/` folder and the `figures/` folder, keeping both at the same level. Alternatively, copy the figures used into `paper/figures/`: `main.tex` uses `fig07` and `fig09`, and `supplement.tex` uses `fig08`, `fig12`, `fig13` and `fig15`.
2. Set the compiler to pdfLaTeX. BibTeX runs automatically.

## Still to do before submission

- The PDFs are anonymous for double-blind review. `anonymous.tex` hides the author block and the GitHub link (https://github.com/t-raisa/football-market-value, public, MIT license) in both `main.tex` and `supplement.tex`. For the camera-ready version, change `\anonymoustrue` to `\anonymousfalse` in `anonymous.tex` and recompile.
- The main paper is 6 pages including references. Check the venue's exact limit (some count references separately).
