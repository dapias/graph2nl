#!/usr/bin/env Rscript
# export_bfi_network.R -- builds network_for_llm.json from the public,
# built-in psych::bfi (25-item Big Five Inventory) dataset, for use as a
# second, genuinely public real-network external-validation case study
# alongside Bereznowski et al. 2023 (see case_studies/bereznowski/).
#
# DATA SOURCE: psych::bfi is bundled directly in the R psych/psychTools
# package (Revelle, W. psych: Procedures for Psychological, Psychometric,
# and Personality Research. Northwestern University. https://CRAN.R-
# project.org/package=psych) -- no external data file, no access request.
#
# GROUND TRUTH CHOICE: the five-factor item groupings (Agreeableness,
# Conscientiousness, Extraversion, Neuroticism, Openness) are used as the
# THEORETICAL/a priori community structure -- baked into the item design
# itself (bfi's own documented scoring key), NOT walktrap-detected. This is
# a firmer ground truth than Bereznowski's stochastic spin-glass communities,
# and it asks a sharper question of the LLM: can it recover a community
# structure it may already "know" from prior training (Big Five is
# extremely well-known), or does the same collapsing/merging failure found
# on Bereznowski's less culturally salient clusters show up regardless?
#
# REVERSE-KEYING: A1, C4, C5, E1, E2, O2, O5 are reverse-scored on the 1-6
# scale, per the psych package's own documented scoring key (bfi.keys;
# https://rdrr.io/cran/psych/man/bfi.html, verified 2026-08-25). Applied
# BEFORE estimation so all edges reflect "higher = more of the named
# construct," matching network_for_llm.json's schema requirement.
#
# ESTIMATION: qgraph::cor_auto() (polychoric, appropriate for 6-point
# ordinal items) + qgraph::EBICglasso(gamma=0.5, threshold=TRUE) -- the
# same correlation + regularization combination bootnet::estimateNetwork(
# default="EBICglasso") uses internally. 
#
# expected_influence/strength_centrality are derived directly from the
# estimated edge matrix by THIS script, not sourced from any published
# paper -- flagged as such in meta.method, same convention as
# export_published_network.R used for Bereznowski.
#
# REPRODUCIBILITY: package versions used to produce the delivered
# network_for_llm_bfi.json and figure are captured to sessionInfo_bfi.txt
# at the end of this script (step 8) -- see that file for exact pins.

suppressMessages({
  if (requireNamespace("psychTools", quietly = TRUE)) {
    library(psychTools)
  } else {
    library(psych)
  }
  library(qgraph)
  library(jsonlite)
})

set.seed(42)  # fixes the spring-layout figure's node positions across reruns;
              # has no effect on estimation (cor_auto/EBICglasso are deterministic
              # given the same data)

data(bfi)

item_prefixes <- c("A", "C", "E", "N", "O")
items <- unlist(lapply(item_prefixes, function(p) paste0(p, 1:5)))
stopifnot(all(items %in% colnames(bfi)))

df <- bfi[, items]

# Reverse-code on the 1-6 scale: reversed = 7 - x
reverse_items <- c("A1", "C4", "C5", "E1", "E2", "O2", "O5")
for (it in reverse_items) df[[it]] <- 7 - df[[it]]

# Listwise deletion, same discipline as used for
# Bereznowski
df_complete <- df[complete.cases(df), ]
n_total <- nrow(bfi)
n_complete <- nrow(df_complete)
cat(sprintf("N: %d total, %d complete cases (listwise deletion)\n", n_total, n_complete))

# --- 2. Correlation + admissibility check -----------------------------------
# Check whether the raw polychoric matrix is already positive-definite
# BEFORE applying cor_auto's forcePD correction, so a correction is logged
# rather than silently applied.
cor_mat_raw <- tryCatch(
  qgraph::cor_auto(df_complete, forcePD = FALSE),
  error = function(e) NULL
)
pd_needed <- TRUE
if (!is.null(cor_mat_raw)) {
  min_eig <- min(eigen(cor_mat_raw, symmetric = TRUE, only.values = TRUE)$values)
  pd_needed <- min_eig <= 0
  cat(sprintf("Raw polychoric matrix: min eigenvalue = %.6f (%s)\n",
              min_eig, if (pd_needed) "NOT positive-definite -- forcePD correction applied below" else "already positive-definite -- forcePD correction is a no-op"))
} else {
  cat("Raw (uncorrected) polychoric matrix could not be computed directly; proceeding with forcePD=TRUE only.\n")
}
cor_mat <- qgraph::cor_auto(df_complete, forcePD = TRUE)

# Polychoric correlation (items are 6-point ordinal) + EBICglasso
# regularization, gamma=0.5 (qgraph/bootnet default)
net <- qgraph::EBICglasso(cor_mat, n = n_complete, gamma = 0.5, threshold = TRUE)
colnames(net) <- rownames(net) <- items

n_edges_check <- sum(net[upper.tri(net)] != 0)
cat(sprintf("Estimated network: %d nonzero edges (of %d possible)\n", n_edges_check, length(items) * (length(items) - 1) / 2))

# Theoretical five-factor community assignment (ground truth, not detected)
factor_of <- setNames(
  rep(c("Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness"), each = 5),
  items
)
community_id <- setNames(rep(1:5, each = 5), items)  # 1=A,2=C,3=E,4=N,5=O -- fixed, arbitrary numbering

# Plain-language item descriptions, POST-reversal polarity (so "higher =
# more of the construct" reads correctly for every item, matching the
# schema's polarity requirement) -- original bfi wording is preserved in a
# trailing note for the 7 reverse-scored items so the reversal is auditable.
item_descriptions <- c(
  A1 = "Attends to and is considerate of others' feelings (reverse-coded from original item 'Am indifferent to the feelings of others'; higher = more considerate)",
  A2 = "Inquires about others' well-being",
  A3 = "Knows how to comfort others",
  A4 = "Loves children",
  A5 = "Makes people feel at ease",
  C1 = "Is exacting/careful in their work",
  C2 = "Continues until everything is perfect",
  C3 = "Does things according to a plan",
  C4 = "Completes tasks thoroughly, not half-way (reverse-coded from 'Do things in a half-way manner'; higher = more thorough)",
  C5 = "Uses time efficiently, does not waste it (reverse-coded from 'Waste my time'; higher = less time-wasting)",
  E1 = "Talks a lot (reverse-coded from 'Don't talk a lot'; higher = more talkative)",
  E2 = "Finds it easy to approach others (reverse-coded from 'Find it difficult to approach others'; higher = more approachable/sociable)",
  E3 = "Knows how to captivate people",
  E4 = "Makes friends easily",
  E5 = "Takes charge",
  N1 = "Gets angry easily",
  N2 = "Gets irritated easily",
  N3 = "Has frequent mood swings",
  N4 = "Often feels blue",
  N5 = "Panics easily",
  O1 = "Is full of ideas",
  O2 = "Engages with difficult reading material (reverse-coded from 'Avoid difficult reading material'; higher = more intellectually curious)",
  O3 = "Carries the conversation to a higher level",
  O4 = "Spends time reflecting on things",
  O5 = "Probes deeply into a subject (reverse-coded from 'Will not probe deeply into a subject'; higher = more intellectually thorough)"
)

# Only nonzero edges, per schema
edge_idx <- which(upper.tri(net) & net != 0, arr.ind = TRUE)
edges <- lapply(seq_len(nrow(edge_idx)), function(k) {
  i <- edge_idx[k, 1]; j <- edge_idx[k, 2]
  w <- net[i, j]
  list(
    source = items[i],
    target = items[j],
    weight = unname(round(w, 4)),
    sign = if (w > 0) "positive" else "negative"
  )
})

strength <- apply(abs(net), 1, sum)
expected_influence <- apply(net, 1, sum)

nodes <- lapply(items, function(it) {
  list(
    id = it,
    description = unname(item_descriptions[it]),
    theme = unname(factor_of[it]),
    community = unname(community_id[it]),
    strength_centrality = unname(round(strength[it], 4)),
    expected_influence = unname(round(expected_influence[it], 4))
  )
})

modularity_val <- tryCatch({
  if (!requireNamespace("igraph", quietly = TRUE)) stop("igraph not available")
  igraph_g <- igraph::graph_from_adjacency_matrix(abs(net), mode = "undirected", weighted = TRUE, diag = FALSE)
  igraph::modularity(igraph_g, membership = community_id[items])
}, error = function(e) {
  cat("modularity computation skipped:", conditionMessage(e), "\n")
  NA
})

meta <- list(
  dataset = "psych::bfi (public, built into the R psych/psychTools package)",
  run_name = "bfi_five_factor_external_validation",
  sample_description = sprintf(
    "N=%d complete cases (of %d total) from the public bfi dataset (25-item Big Five Inventory, 6-point Likert scale, 'Very Inaccurate' to 'Very Accurate')",
    n_complete, n_total),
  n = n_complete,
  method = paste(
    "Polychoric correlation (qgraph::cor_auto) + EBICglasso regularization",
    "(qgraph::EBICglasso, gamma=0.5, threshold=TRUE) -- same correlation+",
    "regularization combination as bootnet::estimateNetwork(default=",
    "'EBICglasso'), called directly. Community membership is the",
    "THEORETICAL five-factor structure from the item design (bfi's own",
    "documented scoring key), NOT walktrap-detected -- so the",
    "'community_detection_modularity' field below is the modularity of a",
    "FIXED, theory-supplied partition, not a data-driven detection result",
    "(contrast with production networks elsewhere in this project, where",
    "the same field name is computed from actual walktrap detection).",
    "strength_centrality/expected_influence are derived directly from the",
    "estimated edge matrix by this export script, not sourced from any",
    "published paper.",
    if (pd_needed) "The raw polychoric matrix required forcePD correction (see console log at generation time)." else "The raw polychoric matrix was already positive-definite; forcePD had no effect."
  ),
  community_detection_modularity = modularity_val,
  caveat = paste(
    "Cross-sectional self-report data -- no causal interpretation warranted.",
    "Community labels reflect the Big Five's well-established theoretical",
    "structure, not blind data-driven detection."
  )
)

out <- list(meta = meta, nodes = nodes, edges = edges)
out_path <- "network_for_llm_bfi.json"
write(jsonlite::toJSON(out, auto_unbox = TRUE, pretty = TRUE, na = "null"), out_path)
cat(sprintf("Wrote %s: %d nodes, %d edges\n", out_path, length(nodes), length(edges)))


# ---------------------------------------------------------------------------
# Figure: qgraph plot of the estimated network, styled consistently with the
# Occupational Well-Being network figure (community-colored nodes, blue/red
# edges for sign, edge width by |weight|). Reuses `net`, `items`, and
# `community_id` from the estimation above rather than re-reading the JSON.
# set.seed() above fixes the spring layout so this figure is reproducible.
# ---------------------------------------------------------------------------

factor_names <- c("Agreeableness", "Conscientiousness", "Extraversion", "Neuroticism", "Openness")
palette <- c("#66C2A5", "#8DA0CB", "#FC8D62", "#E78AC3", "#A6D854")
group_factor <- factor(factor_names[community_id[items]], levels = factor_names)

fig_path <- "bfi_personality_network.pdf"
pdf(fig_path, width = 8, height = 8)
qgraph(net,
       layout = "spring",
       groups = group_factor,
       color = palette,
       labels = items,
       label.cex = 1.1,
       vsize = 6,
       posCol = "#4A7EBB",
       negCol = "#D9534F",
       edge.width = 1.1,
       fade = TRUE,
       legend = FALSE,
       #legend.cex = 0.35,
       layoutScale = c(0.9, 0.9),
       border.width = 1.2,
       label.color = "black")
dev.off()

cat(sprintf("Wrote %s\n", fig_path))

# ---------------------------------------------------------------------------
# 8. Reproducibility: pin the exact package/R versions used for this run.
# ---------------------------------------------------------------------------
sessioninfo_path <- "sessionInfo_bfi.txt"
writeLines(capture.output(sessionInfo()), sessioninfo_path)
cat(sprintf("Wrote %s\n", sessioninfo_path))
