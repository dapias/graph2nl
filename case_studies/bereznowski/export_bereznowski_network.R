# =========================================================================
# export_published_network.R
#
# Builds network_for_llm.json from the PAPER'S OWN analysis, not a
# reimplementation. Sections 1-2 below reuse Bereznowski et al. (2023)'s
# own `script.R` verbatim (composite construction, listwise deletion),
# and the network estimation call in section 3 is bootnet's own wrapper
# (`estimateNetwork(..., default = "EBICglasso", threshold = TRUE)`),
# which is what the paper itself used.
#
# DATA/SCRIPT SOURCE: dataset.csv and the original script.R are NOT
# included in this repo. Both are the original authors' own materials,
# openly deposited by them at https://osf.io/jvqfa/ (see the paper's Data
# availability statement: "The analytic code for all analyses performed in
# this study and the Supplementary material are available at
# https://osf.io/jvqfa/."). To reproduce this export, download dataset.csv
# from that OSF deposit and place it at the path set in DATA_PATH below (or
# pass a different path via the GRAPH2NL_BEREZNOWSKI_DATA environment
# variable). Citation: Bereznowski, P., Atroszko, P. A., & Konarski, R.
# (2023). Work addiction, work engagement, job burnout, and perceived
# stress: A network analysis. Frontiers in Psychology, 14, 1130069.
# https://doi.org/10.3389/fpsyg.2023.1130069 (CC BY 4.0).
#
# Community membership is NOT re-detected here. `igraph::cluster_spinglass`
# is stochastic and gave an unstable, negative-modularity partition in an
# earlier attempt at this reproduction. Instead the four clusters are
# hardcoded exactly as reported in the paper's Results section 3.2 (the
# actual published finding, not a re-run of the algorithm that produced
# it) -- this is the more faithful choice for an external-validation
# comparison against the paper's own Discussion.
#
# REPRODUCIBILITY: package versions used to produce the delivered
# network_for_llm.json and figures are captured to sessionInfo_bereznowski.txt
# at the end of this script (section 10) -- see that file for exact pins.
#
# Requires: bootnet, qgraph, igraph, dplyr, jsonlite.
# =========================================================================

library(bootnet)
library(dplyr)
library(igraph)
library(qgraph)
library(jsonlite)

# --- 0. Locate the data ------------------------------------------------------
# Not bundled with this repo -- see the header note above. Defaults to a
# `data/` subdirectory next to this script; override with an environment
# variable if you keep the OSF download elsewhere.
# 1. `Rscript path/to/file.R` from a terminal
.get_script_dir <- function() {
  # 1. `Rscript path/to/file.R` from a terminal
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- sub("^--file=", "", grep("^--file=", args, value = TRUE))
  if (length(file_arg) == 1) return(dirname(normalizePath(file_arg)))
  
  # 2. base::source("path/to/file.R") -- sets `ofile` in the sourcing frame
  ofile <- tryCatch(sys.frames()[[1]]$ofile, error = function(e) NULL)
  if (!is.null(ofile) && nzchar(ofile)) return(dirname(normalizePath(ofile)))
  
  # 3. RStudio's "Source" button, or running interactively with this file
  #    open as the active editor tab -- neither of the above sets anything,
  #    but the file's own path is still recoverable via rstudioapi.
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      tryCatch(rstudioapi::isAvailable(), error = function(e) FALSE)) {
    ctx <- tryCatch(rstudioapi::getSourceEditorContext(), error = function(e) NULL)
    if (!is.null(ctx) && nzchar(ctx$path)) return(dirname(normalizePath(ctx$path)))
  }
  
  # 4. Give up -- caller falls back to the working directory, which is only
  #    correct if you setwd()'d here (or launched R/RStudio from here) first.
  getwd()
}

env_path <- Sys.getenv("GRAPH2NL_BEREZNOWSKI_DATA", unset = NA)
candidates <- c(
  env_path,
  file.path(.get_script_dir(), "data", "dataset.csv"),
  file.path("data", "dataset.csv")
)
candidates <- candidates[!is.na(candidates)]
found <- candidates[file.exists(candidates)]

if (length(found) == 0) {
  stop(sprintf(
    "Could not find dataset.csv in any of:\n  %s\nDetected script directory: %s\nCurrent working directory: %s\nIf the script directory above is wrong, you likely ran this via source()/console-paste rather than `Rscript file.R`, and rstudioapi (if installed) also couldn't resolve it -- setwd() to the script's folder first, or set GRAPH2NL_BEREZNOWSKI_DATA to the full path of dataset.csv. Download the file itself from https://osf.io/jvqfa/ (Bereznowski, Atroszko & Konarski, 2023) if you don't already have it.",
    paste(candidates, collapse = "\n  "), .get_script_dir(), getwd()
  ))
}
DATA_PATH <- found[1]
cat(sprintf("Using data file: %s\n", DATA_PATH))

# --- 1. Load and prep data (verbatim from the paper's script.R, sections 1-2) ----

data <- read.csv(DATA_PATH)

data <- data %>%
  mutate(PSS4R = case_when(PSS4 == 1 ~ 5, PSS4 == 2 ~ 4, PSS4 == 3 ~ 3, PSS4 == 4 ~ 2, PSS4 == 5 ~ 1),
         PSS5R = case_when(PSS5 == 1 ~ 5, PSS5 == 2 ~ 4, PSS5 == 3 ~ 3, PSS5 == 4 ~ 2, PSS5 == 5 ~ 1),
         PSS7R = case_when(PSS7 == 1 ~ 5, PSS7 == 2 ~ 4, PSS7 == 3 ~ 3, PSS7 == 4 ~ 2, PSS7 == 5 ~ 1),
         PSS8R = case_when(PSS8 == 1 ~ 5, PSS8 == 2 ~ 4, PSS8 == 3 ~ 3, PSS8 == 4 ~ 2, PSS8 == 5 ~ 1)) %>%
  mutate(UWES1 = UWES1 + 1, UWES2 = UWES2 + 1,
         UWES3 = UWES3 + 1, UWES4 = UWES4 + 1,
         UWES5 = UWES5 + 1, UWES6 = UWES6 + 1,
         UWES7 = UWES7 + 1, UWES8 = UWES8 + 1,
         UWES9 = UWES9 + 1,
         Vigor = UWES1 + UWES2 + UWES5,
         Dedication = UWES3 + UWES4 + UWES7,
         Absorption = UWES6 + UWES8 + UWES9,
         Exhaustion = MBI1 + MBI2 + MBI3 + MBI4 + MBI6,
         Cynicism = MBI8 + MBI9 + MBI13 + MBI14 + MBI15,
         Efficacy = MBI5 + MBI7 + MBI10 + MBI11 + MBI12 + MBI16,
         Stress = PSS1 + PSS2 + PSS3 + PSS4R + PSS5R + PSS6 + PSS7R + PSS8R + PSS9 + PSS10)

data <- data %>%
  mutate(removed = ifelse(is.na(BWAS1) | is.na(BWAS2) | is.na(BWAS3) | is.na(BWAS4) |
                            is.na(BWAS5) | is.na(BWAS6) | is.na(BWAS7) | is.na(Vigor) |
                            is.na(Dedication) | is.na(Absorption) | is.na(Exhaustion) | is.na(Cynicism) |
                            is.na(Efficacy) | is.na(Stress), 1, 0))

data.without.na <- data %>% filter(removed == 0)

data.network <- data.without.na %>%
  select(BWAS1:BWAS7, Vigor:Absorption, Exhaustion:Efficacy, Stress)

n_obs <- nrow(data.network)  # should be 676

# --- 2. Network estimation (paper's own method: bootnet + EBICglasso) -----------

network <- estimateNetwork(data.network, default = "EBICglasso", threshold = TRUE)
W <- getWmat(network)  # 14x14 signed partial-correlation matrix

n_edges <- sum(W != 0) / 2
mean_abs_weight <- mean(abs(W[upper.tri(W)]))
max_weight <- max(W)
cat(sprintf("Sanity check vs. paper: edges=%d (paper: 27), mean|w|=%.3f (paper: .06), max=%.3f (paper: .42)\n",
            n_edges, mean_abs_weight, max_weight))

# --- 3. Node metadata: id, description, theme, published community -------------

node_order <- c("salience", "tolerance", "mood_modification", "relapse", "withdrawal",
                 "conflict", "problems", "vigor", "dedication", "absorption",
                 "exhaustion", "cynicism", "professional_efficacy", "stress")

node_description <- c(
  salience              = "Salience -- work dominates thoughts, feelings, and behavior (Bergen Work Addiction Scale item 1).",
  tolerance             = "Tolerance -- needing to work increasingly more to achieve the same satisfaction (BWAS item 2).",
  mood_modification     = "Mood modification -- working to escape from or improve a negative mood (BWAS item 3).",
  relapse               = "Relapse -- attempts to cut down on work fail and old patterns resume (BWAS item 4).",
  withdrawal            = "Withdrawal -- becoming stressed or distressed when prevented from working (BWAS item 5).",
  conflict              = "Conflict -- work interferes with relationships, hobbies, or other activities (BWAS item 6).",
  problems              = "Problems -- health or other problems caused by working too much (BWAS item 7).",
  vigor                 = "Vigor -- high energy and mental resilience while working (UWES-9 subscale).",
  dedication            = "Dedication -- sense of significance, enthusiasm, pride, and challenge in one's work (UWES-9 subscale).",
  absorption            = "Absorption -- being fully concentrated and engrossed in one's work, with difficulty detaching (UWES-9 subscale).",
  exhaustion            = "Exhaustion -- feelings of energy depletion from work (MBI-GS subscale).",
  cynicism              = "Cynicism -- increased mental distance from and negativism toward one's work (MBI-GS subscale).",
  professional_efficacy = "Professional efficacy -- sense of competence and accomplishment at work (MBI-GS subscale; higher score = more efficacy, NOT reverse-coded).",
  stress                = "Perceived stress -- general perceived stress over the past month (PSS-10 total score)."
)

# researcher-assigned theme = the instrument/construct each item was designed to measure
node_theme <- c(
  salience = "work_addiction", tolerance = "work_addiction", mood_modification = "work_addiction",
  relapse = "work_addiction", withdrawal = "work_addiction", conflict = "work_addiction", problems = "work_addiction",
  vigor = "work_engagement", dedication = "work_engagement", absorption = "work_engagement",
  exhaustion = "job_burnout", cynicism = "job_burnout", professional_efficacy = "job_burnout",
  stress = "perceived_stress"
)

# DATA-DRIVEN community = the paper's own 4 spin-glass clusters (Results, section 3.2),
# hardcoded from the published text rather than re-run, since spinglass is stochastic
# and produced an unstable/negative-modularity result when re-run on this reproduction.
# Note professional_efficacy (theme = job_burnout) lands in community 3 with engagement,
# and stress (theme = perceived_stress) lands in community 4 with the burnout core --
# exactly the theme != community divergence the paper's Discussion calls out as notable.
node_community <- c(
  salience = 1, mood_modification = 1, withdrawal = 1,                          # "peripheral" addiction symptoms
  tolerance = 2, relapse = 2, conflict = 2, problems = 2,                        # "core" addiction symptoms
  vigor = 3, dedication = 3, absorption = 3, professional_efficacy = 3,          # engagement + efficacy
  exhaustion = 4, cynicism = 4, stress = 4                                       # burnout core + stress
)

# --- 4. Centrality: strength (paper's own metric) + expected influence (derived) --

# Standard node strength, exactly as in the paper's script section 8
strength_standard <- centrality(network)$InDegree
names(strength_standard) <- node_order

# Expected influence is NOT computed anywhere in the paper's own script/text --
# it is derived here directly from the published edge matrix (signed row sums,
# Robinaugh et al. 2016 definition), not re-estimated. Flagged in meta.method below.
expected_influence <- rowSums(W)
names(expected_influence) <- node_order

# --- 5. Modularity of the hardcoded published partition (for meta, informational) --

ig <- graph_from_adjacency_matrix(abs(W), mode = "undirected", weighted = TRUE, diag = FALSE)
membership_vec <- node_community[node_order]
modularity_value <- modularity(ig, membership = membership_vec, weights = E(ig)$weight)

# --- 6. Build nodes array ---------------------------------------------------------

nodes <- lapply(node_order, function(id) {
  list(
    id = id,
    description = unname(node_description[id]),
    theme = unname(node_theme[id]),
    community = unname(node_community[id]),
    strength_centrality = round(unname(strength_standard[id]), 4),
    expected_influence = round(unname(expected_influence[id]), 4)
  )
})

# --- 7. Build edges array (nonzero upper-triangle entries only) -------------------

edges <- list()
idx <- 1
for (i in 1:13) {
  for (j in (i + 1):14) {
    w <- W[i, j]
    if (w != 0) {
      edges[[idx]] <- list(
        source = node_order[i],
        target = node_order[j],
        weight = round(w, 4),
        sign = ifelse(w > 0, "positive", "negative")
      )
      idx <- idx + 1
    }
  }
}

# --- 8. Assemble network_for_llm.json ---------------------------------------------

network_for_llm <- list(
  meta = list(
    dataset = "Bereznowski, Atroszko & Konarski (2023), Frontiers in Psychology 14:1130069 (data/code: https://osf.io/jvqfa/, CC BY 4.0)",
    run_name = "rq4_published_network_faithful",
    sample_description = "676 working Poles (mean age 36.12, SD 11.23), listwise-complete on BWAS-7/UWES-9/MBI-GS/PSS-10",
    n = n_obs,
    method = paste0(
      "qgraph::cor_auto() correlations + bootnet::estimateNetwork(default='EBICglasso', ",
      "threshold=TRUE), gamma=0.5 -- the paper's own estimation call. Reproduced here: ",
      n_edges, " edges (paper: 27), mean|weight|=", round(mean_abs_weight, 3), " (paper: .06), ",
      "max=", round(max_weight, 3), " (paper: .42). Community assignment is NOT re-detected ",
      "(spin-glass is stochastic and was unstable on re-estimation); it is hardcoded from the ",
      "paper's own reported 4-cluster spin-glass result (Results 3.2). expected_influence is ",
      "derived here from the published edge matrix (signed row sums) -- the paper's own script ",
      "does not report this metric, only standard node strength and a modified/bridge strength."
    ),
    community_detection_modularity = round(modularity_value, 3),
    community_detection_note = "Modularity computed here on |edge weights| for the hardcoded published partition; the paper does not report a modularity value for its spin-glass solution.",
    caveat = "Cross-sectional data -- no causal interpretation warranted. EBIC-glasso regularization shrinks small partial correlations to exactly zero; an absent edge is not evidence the population association is zero."
  ),
  nodes = nodes,
  edges = edges
)

write_json(network_for_llm, "network_for_llm.json", pretty = TRUE, auto_unbox = TRUE, digits = 4)
cat("Wrote network_for_llm.json:", length(nodes), "nodes,", length(edges), "edges.\n")


# =========================================================================
# 9. Network visualization (qgraph)
# =========================================================================

set.seed(42)  # fixes the spring-layout figure's node positions across
              # reruns; has no effect on the estimation above

# Short labels for plotting -- readable derivatives of the same construct
# names used in prose and Table 11 (just without the underscore), so a
# reader can map figure <-> table <-> text at a glance without learning a
# separate abbreviation scheme.
node_labels <- c(
  salience = "salience", tolerance = "tolerance", mood_modification = "mood mod.",
  relapse = "relapse", withdrawal = "withdrawal", conflict = "conflict",
  problems = "problems", vigor = "vigor", dedication = "dedication",
  absorption = "absorption", exhaustion = "exhaustion", cynicism = "cynicism",
  professional_efficacy = "efficacy", stress = "stress"
)

community_names <- c(
  "1" = "Work addiction (peripheral)",
  "2" = "Work addiction (core)",
  "3" = "Work engagement",
  "4" = "Burnout & stress"
)

groups_factor <- factor(
  node_community[node_order],
  levels = names(community_names),
  labels = community_names
)

# Colorblind-safe qualitative palette (Set2), consistent across your two
# network figures if you reuse this for the Personality network too --
# just extend to 5 colors there for the five factor communities.
community_colors <- c("#66C2A5", "#FC8D62", "#8DA0CB", "#E78AC3")

# Longer word labels (vs. the earlier 3-4 letter abbreviations) need bigger
# nodes and smaller label text to fit without overflowing the circles.
# Title and community legend removed here -- both moved to the LaTeX
# figure caption instead (community names + label--Table 11 mapping).
pdf("occupational_wellbeing_network.pdf", width = 8, height = 8)
qgraph(
  W,
  layout      = "spring",
  labels      = node_labels[node_order],
  groups      = groups_factor,
  color       = community_colors,
  posCol      = "#2166AC",   # positive edges, blue
  negCol      = "#B2182B",   # negative edges, red
  edge.labels = FALSE,
  vsize       = 11,
  label.cex   = 0.8,
  legend = FALSE,
  label.scale = FALSE        # keep uniform text size across nodes rather
  # than shrinking to fit -- prevents "mood
  # mod." rendering smaller than "vigor"
)
dev.off()


cat("Wrote occupational_wellbeing_network.pdf\n")

# =========================================================================
# 10. Reproducibility: pin the exact package/R versions used for this run.
# =========================================================================
sessioninfo_path <- "sessionInfo_bereznowski.txt"
writeLines(capture.output(sessionInfo()), sessioninfo_path)
cat(sprintf("Wrote %s\n", sessioninfo_path))
