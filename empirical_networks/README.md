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
  verbatim for data preparation (composite construction, listwise
  deletion) and calls the paper's own estimation function
  (`bootnet::estimateNetwork(default="EBICglasso", threshold=TRUE)`).
  Community membership is hardcoded from the paper's own reported
  4-cluster spin-glass result (Results 3.2) rather than re-detected.
- `network_for_llm.json` -- this script's verified output: 14 nodes, 27
  edges, N=676 -- an exact match to the paper's own reported density
  (27/91 edges), and to every edge weight the paper's Discussion section
  cites by name.

**To reproduce:** download `dataset.csv` and the original `script.R` from
the paper's own OSF supplementary materials
(https://osf.io/jvqfa/), place `dataset.csv` next to
`export_published_network.R`, and run it (requires R packages `bootnet`,
`dplyr`, `igraph`, `qgraph`, `jsonlite`).

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
  Community membership is the theoretical five-factor structure from the
  item design, not data-driven detection. Also produces a qgraph figure
  of the estimated network.
- `network_for_llm_bfi.json` -- this script's verified output: 25 nodes,
  89 edges, N=2,436 complete cases (of 2,800 total).

**To reproduce:** run `export_bfi_network.R` directly -- no external data
file needed, since `psych::bfi` is bundled with the package (requires R
packages `psych`, `qgraph`, `igraph`, `jsonlite`).
