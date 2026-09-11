# Empirical networks

Two real-network external-validation case studies (Section 5.2 of the
paper). Each subfolder contains the export script that builds a
`network_for_llm.json`-schema file from the source data, plus that
script's actual output.

**Raw survey/response data is intentionally not redistributed here.**
Both sources are already public elsewhere; re-hosting a copy under this
repo's license would blur who owns what. See each subfolder for exact
provenance.

## occupational_wellbeing/

The BWAS-7/UWES-9/MBI-GS/PSS-10 network from Bereznowski, Atroszko &
Konarski (2023), "Work addiction, work engagement, job burnout, and
perceived stress: A network analysis," *Frontiers in Psychology* 14:1130069
(https://doi.org/10.3389/fpsyg.2023.1130069).

- `export_published_network.R` -- reuses the paper's own `script.R`
  logic verbatim for data preparation (composite construction, listwise
  deletion) and calls the paper's own estimation function
  (`bootnet::estimateNetwork(default="EBICglasso", threshold=TRUE)`).
  Community membership is hardcoded from the paper's own reported
  4-cluster spin-glass result (Results 3.2) rather than re-detected.
- `network_for_llm.json` -- this script's verified output: 14 nodes, 27
  edges, N=676 -- an exact match to the paper's own reported density
  (27/91 edges), and to every edge weight the paper's Discussion section
  cites by name.
- `occupational_wellbeing_network.pdf` / `.png` -- the corresponding
  qgraph figure, community-colored, with a fixed layout seed for
  reproducibility.
- `sessionInfo_bereznowski.txt` -- exact R/package versions used to
  produce the above.

**To reproduce:** download `dataset.csv` from the paper's own OSF
supplementary materials (https://osf.io/jvqfa/) and place it in a `data/`
subdirectory next to `export_published_network.R` (i.e.
`data/dataset.csv`), then run the script -- it locates that file
automatically regardless of your working directory, or you can point it
elsewhere via the `GRAPH2NL_BEREZNOWSKI_DATA` environment variable.
Requires R packages `bootnet`, `dplyr`, `igraph`, `qgraph`, `jsonlite`.
The original `script.R` is not needed to run this reproduction (its
data-prep logic is already reused verbatim above) -- download it from the
same OSF deposit only if you want to independently verify that fidelity
yourself.

## personality/

A 25-item Big Five Inventory network estimated from the public `bfi`
dataset built into R's `psych` package (Goldberg, 1999; Revelle, 2024,
https://CRAN.R-project.org/package=psych).

- `export_bfi_network.R` -- loads `psych::bfi` directly (no external file
  needed), reverse-codes the 7 items the package's own documented scoring
  key (`bfi.keys`) marks as reverse-scored, estimates the network via
  `qgraph::cor_auto()` (polychoric) + `qgraph::EBICglasso(gamma=0.5,
  threshold=TRUE)` -- the same correlation+regularization combination
  `bootnet::estimateNetwork(default="EBICglasso")` uses internally.
  Checks (and logs) whether `cor_auto`'s `forcePD` correction was actually
  needed, rather than applying it silently. Community membership is the
  theoretical five-factor structure from the item design, not data-driven
  detection. Also produces a qgraph figure of the estimated network, with
  a fixed layout seed for reproducibility.
- `network_for_llm_bfi.json` -- this script's verified output: 25 nodes,
  89 edges, N=2,436 complete cases (of 2,800 total).
- `bfi_personality_network.pdf` -- the corresponding qgraph figure.
- `sessionInfo_bfi.txt` -- exact R/package versions used to produce the
  above.

**To reproduce:** run `export_bfi_network.R` directly -- no external data
file needed, since `psych::bfi` is bundled with the package (requires R
packages `psych`, `qgraph`, `igraph`, `jsonlite`).
