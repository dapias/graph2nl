#!/usr/bin/env python3
"""
Score LLM interpretations against known synthetic or procedural ground truth.

Checks use keywords and regular expressions. They are reproducible, but
unusual phrasing may cause false positives or false negatives.

Each score_* function takes the LLM's raw text output and the test's
ground_truth dict, and returns:
    {"passed": bool, "score": float in [0,1], "details": [str, ...]}
"""
import re

CALIBRATION_TERMS = {
    "negligible": ["negligible", "trivial", "essentially no", "near-zero", "near zero", "no meaningful"],
    "weak": ["weak"],
    "moderate": ["moderate"],
    "strong": ["strong"],
}
# Magnitude bands in ascending order.
BAND_ORDER = ["negligible", "weak", "moderate", "strong"]

# In plain language, "very weak" can describe a negligible edge and
# "moderately strong" a moderate edge. Mask these phrases only for the
# corresponding true band; bare band words still count.
INTENSIFIED_ADJACENT_TERMS = {
    ("weak", "negligible"): [r"very\s+weak", r"extremely\s+weak", r"quite\s+weak", r"barely\s+weak"],
    ("strong", "moderate"): [r"moderately\s+strong", r"fairly\s+strong"],
}

# Comparative forms such as "strongest" rank edges; they do not assign
# the formal "strong" band to every edge in the same sentence.
_COMPARATIVE_SAFE_PATTERN = {
    "weak": re.compile(r"weak(?!er\b|est\b)"),
    "strong": re.compile(r"strong(?!er\b|est\b)"),
}

# "less strongly" / "more weakly" compare edges; they do not assign a band
# (same rationale as the -er/-est exclusion above).
_COMPARATIVE_PHRASE = re.compile(r"\b(?:less|more)\s+(?:strong|weak)(?:ly)?\b")

POSITIVE_TERMS = [
    r"positiv\w*",                              # positive, positively (also matches inside "net positive influence")
    r"go(es)? together", r"move together", r"co-?occur\w*", r"increases? with",
    r"in the same direction",
    r"\bhigher\b[^.!?]{0,120}\bhigher\b",       # "a higher X ... higher Y" -- implicit-comparison phrasing,
    r"\blower\b[^.!?]{0,120}\blower\b",         # common in the plain-language layer, which avoids jargon like
]                                                # "positive association" on purpose (see prompts/full_protocol.md)
NEGATIVE_TERMS = [
    r"negativ\w*",                              # negative, negatively (also matches "net negative influence")
    r"invers\w*", r"opposite direction", r"decreases? with",
    r"\bhigher\b[^.!?]{0,120}\blower\b",        # "a higher X ... lower Y"
    r"\blower\b[^.!?]{0,120}\bhigher\b",
]
# Regexes also recognize directional descriptions such as "higher X, lower Y".

CAUSAL_TERMS_DEFAULT = ["causes", "cause of", "leads to", "leading to", "drives", "driving",
                         "results in", "resulting in", "because of", "due to", "brings about"]

RELATION_TERMS = [
    r"associat\w*", r"correlat\w*", r"\blink\w*", r"\brelat\w*", r"\bconnect\w*",
    r"tend(s|ed)? to", r"move together", r"go(es)? together", r"co-?occur\w*", r"\btie[sd]?\b",
]
# Distinguish a relationship claim from an overview that merely names nodes.

NEGATION_CUES = [r"\bnot\b", r"n't\b", r"\bno evidence\b", r"\bdoes not mean\b", r"\bdoesn't mean\b",
                  r"\bcannot\b", r"\bcan't\b", r"\bisn't evidence\b", r"\bis not evidence\b", r"\brather than\b"]
# Regexes recognize negation even around Markdown emphasis ("**not**").

# Allow words between denial cues and relationship terms.
DENIAL_PATTERNS = [
    r'\bno\b[^.]{0,130}\b(association|relationship|link\w*|correlat\w*|connect\w*)\b',
    r'\b(not|n[o\']t)\b[^.]{0,40}\b(associat\w*|relat\w*|link\w*|correlat\w*|connect\w*)\b',
    r'\bunrelated\b',
    # Match "no" or "not" before a later tendency or connection term.
    r'\b(?:no|not)\b[^.]{0,40}\b(tendency|pattern|connection|link\w*)\b',
    # "Isolated" itself denies a connection.
    r'\bisolat\w*\b',
    # "Zero association" and "lack of a link" also deny an edge.
    r'\bzero\b[^.]{0,60}\b(association|relationship|link\w*|correlat\w*|connect\w*)\b',
    r'\black(?:s|ing)?\s+(?:of\s+)?(?:a\s+|an\s+)?(association|relationship|link\w*|correlat\w*|connect\w*)\b',
]
# Word stems cover forms such as "link", "linked", and "correlations".


def _normalize(text):
    """Normalize whitespace while preserving newlines between table rows."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    return re.sub(r'[ \t]+', ' ', text)


_SECTION_HEADING_RE = re.compile(
    r'^##\s+(Plain-language summary|Technical interpretation)\s*$', re.MULTILINE)


def _split_sections(text):
    """Split exact protocol headings, falling back to full text if absent.

    Currently unused by the scorers, which check the entire response.
    """
    matches = list(_SECTION_HEADING_RE.finditer(text))
    if not matches:
        return text, text, text
    sections = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections[m.group(1)] = text[start:end].strip()
    plain = sections.get("Plain-language summary")
    technical = sections.get("Technical interpretation")
    if plain is None and technical is None:
        return text, text, text
    return (plain if plain is not None else text), (technical if technical is not None else text), text


_STRUCTURAL_LINE = re.compile(r'^(#+\s|[-*•]\s|\d+[.)]\s|\|)')


def _join_soft_wraps(text):
    """Join word-wrapped prose while preserving Markdown structure."""
    lines = text.split('\n')
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append('')  # preserve paragraph breaks
            continue
        if out and out[-1]:
            prev = out[-1]
            prev_ends = bool(re.search(r'[.!?:]\s*$', prev)) or prev.rstrip().endswith('|')
            this_structural = bool(_STRUCTURAL_LINE.match(stripped))
            if not prev_ends and not this_structural:
                out[-1] = prev + ' ' + stripped
                continue
        out.append(stripped)
    return '\n'.join(out)


def _sentences(text):
    """Split prose into sentences and Markdown into separate segments."""
    segments = []
    for line in _join_soft_wraps(text).split('\n'):
        line = line.strip()
        if not line:
            continue
        parts = re.split(r'(?<=[.!?])\s+(?=[A-Z0-9(#]|[-*•]\s)', line)
        segments.extend(p for p in parts if p.strip())
    return segments


_TOKEN_RE_CACHE = {}


def _token_re(token):
    """Match a whole node ID, allowing a space before its final character."""
    if token not in _TOKEN_RE_CACHE:
        core = re.escape(token) if len(token) < 2 else re.escape(token[:-1]) + r'\s?' + re.escape(token[-1])
        pattern = r'(?<!\w)' + core + r'(?!\w)'
        _TOKEN_RE_CACHE[token] = re.compile(pattern)
    return _TOKEN_RE_CACHE[token]


def _token_in(token, text):
    """Whether a whole node ID appears in text."""
    return _token_re(token).search(text) is not None


def _mentions(text, token):
    """Positions of whole node IDs, including their optional-space form."""
    return [m.start() for m in _token_re(token).finditer(text)]


def _window(text, pos, size=400):
    return text[max(0, pos - size):pos + size]


_CONTROL_CUES_RE = re.compile(
    r'\b(net of|after (?:accounting|controlling|adjusting) for|controlling for|adjusting for|'
    r'conditional on|holding [^.]{0,25} constant|once [^.]{0,25} (?:is|are) (?:accounted|controlled) for)\b',
    re.IGNORECASE,
)


def _other_nodes_in(s, a, b, all_nodes):
    """Count OTHER known nodes mentioned in sentence `s`, EXCLUDING any
    mentioned only as a controlled-for covariate ('net of X', 'after
    accounting for X', 'controlling for X', ...) rather than as a genuine
    second relationship. Describing a partial correlation correctly and
    completely REQUIRES naming the other variables it's conditioned on --
    treating every such mention as just as contaminating as a real second
    relationship (a recap sentence, a cross-pair comparison) was too broad.
    Found via real model output: the one sentence with the actual
    quantitative characterization of an edge ("...partial r = 0.45
    (**Strong**)... even after accounting for synB, synC, and synD.") was
    excluded from the "clean" set for exactly this reason, so `prefer_clean`
    fell back to two other sentences that never mentioned a band word at
    all, silently producing a false "no band word found" (WRONG) result on
    an edge the model actually characterized correctly."""
    if not all_nodes:
        return 0
    count = 0
    for n in all_nodes:
        if n in (a, b):
            continue
        for m in _token_re(n).finditer(s):
            preceding = s[max(0, m.start() - 60):m.start()]
            if _CONTROL_CUES_RE.search(preceding):
                continue  # named only as a controlled-for covariate -- doesn't count
            count += 1
    return count


# Split article-led list items without splitting coordinated node names.
_LIST_ITEM_BOUNDARY_RE = re.compile(r'(?:,|\band)\s+(?:a|an|the)\b', re.IGNORECASE)

# Generic references to remaining nodes are not pair-specific claims.
_GENERIC_OTHER_ITEMS_RE = re.compile(
    r'\b(?:the\s+)?(?:remaining|other|rest\s+of\s+the)\s+'
    r'(?:items?|variables?|nodes?|pairs?|edges?|connections?|associations?|links?)\b',
    re.IGNORECASE)


def _prefer_clean_window_text(s, a, b, all_nodes):
    """Return a pair-specific segment if its relationship claim is unambiguous.

    For a hub-and-spoke list, split at clause or article-led list boundaries
    and keep only segments naming the target spoke. A generic reference to
    other unnamed items prevents this relaxation. Only calibration and sign
    scoring use it; cross-community checks retain stricter pair matching.
    """
    base_count = _other_nodes_in(s, a, b, all_nodes)
    if base_count == 0:
        return s  # already clean under the standard check; return unchanged

    boundaries = sorted(set(
        [m.start() for m in _LIST_ITEM_BOUNDARY_RE.finditer(s)] +
        [m.start() for m in _CLAUSE_DELIM_RE.finditer(s)]
    ))
    if not boundaries:
        return None  # no enumerated/clause structure detected; not clean

    seg_starts = [0] + boundaries
    seg_ends = boundaries + [len(s)]
    segments = list(zip(seg_starts, seg_ends))

    def seg_of(pos):
        for i, (lo, hi) in enumerate(segments):
            if lo <= pos < hi:
                return i
        return len(segments) - 1

    b_positions = [m.start() for m in _token_re(b).finditer(s)]
    if not b_positions:
        return None
    relevant = {seg_of(p) for p in b_positions}

    for n in all_nodes:
        if n in (a, b):
            continue
        for m in _token_re(n).finditer(s):
            preceding = s[max(0, m.start() - 60):m.start()]
            if _CONTROL_CUES_RE.search(preceding):
                continue
            if seg_of(m.start()) in relevant:
                return None  # a genuinely entangled other-node mention -- still ambiguous

    result_text = " ".join(s[lo:hi] for i, (lo, hi) in enumerate(segments) if i in relevant)

    if _GENERIC_OTHER_ITEMS_RE.search(result_text):
        # The specific segment(s) being returned for b's claim sweep up
        # unnamed "other items" (e.g. "...and weaker or negligible links
        # to the REMAINING ITEMS, all within a single homogeneous
        # community") -- not a safe relaxation target, even though no
        # KNOWN node token triggered contamination. This is checked only
        # against the segment(s) actually being returned, not the whole
        # sentence: a well-formed list's own LEAD-IN legitimately uses
        # similar phrasing ("The REMAINING CONNECTIONS from Node A are
        # weaker: a weak association with B... and a negligible one with
        # C...") without being vague -- that lead-in sits in a DIFFERENT
        # segment than each spoke's own claim and is never part of what
        # gets returned for any individual pair, so it does not trigger
        # this guard.
        return None

    return result_text


def _pair_windows(text, a, b, sentence_radius=1, all_nodes=None, prefer_clean=False):
    """Sentence-based windows around every place both tokens a and b appear
    in the same sentence, or within `sentence_radius` sentences of each
    other -- deliberately narrower than a raw character radius, since a
    fixed character radius pulls in neighbouring sentences about entirely
    different edges, producing false "ambiguous" results whenever a
    DIFFERENT pair happens to be discussed a few sentences away.

    prefer_clean (only meaningful if all_nodes is given): a real model's
    concluding recap sentence often names EVERY node at once ("Overall,
    synF relates positively to synG and negatively to synH...") -- such a
    sentence trivially contains any pair you ask about, so treating it as
    a normal direct hit reintroduces exactly the cross-contamination this
    function exists to prevent (calibration and sign_direction both showed
    an identical score on every one of 5 repeats against real model
    output, which turned out to mean every edge's window included the same
    all-nodes recap sentence). When prefer_clean is set, sentences that
    mention ONLY a and b (no other known node, aside from controlled-for
    covariates -- see _other_nodes_in) are tried first; only if NONE exist
    does it fall back to the normal (possibly noisier) match. Not applied
    by default -- score_community's pairwise-cohesion check WANTS
    multi-node sentences (e.g. "Alpha1, Alpha2, and Alpha3 are all
    interconnected" is exactly the signal that check is looking for), so
    excluding them there would be a regression, not a fix."""
    sents = _sentences(text)

    if prefer_clean and all_nodes:
        clean_texts = []
        for s in sents:
            if not (_token_in(a, s) and _token_in(b, s)):
                continue
            w = _prefer_clean_window_text(s, a, b, all_nodes)
            if w is not None:
                clean_texts.append(w)
        if clean_texts:
            return clean_texts

    hit_idx = [i for i, s in enumerate(sents) if _token_in(a, s) and _token_in(b, s)]
    if hit_idx:
        return [sents[i] for i in hit_idx]
    # fall back to "mentioned within `sentence_radius` sentences of each other"
    a_idx = [i for i, s in enumerate(sents) if _token_in(a, s)]
    b_idx = [i for i, s in enumerate(sents) if _token_in(b, s)]
    windows = []
    for ia in a_idx:
        for ib in b_idx:
            if abs(ia - ib) <= sentence_radius:
                lo, hi = sorted((ia, ib))
                windows.append(" ".join(sents[lo:hi + 1]))
    return windows


def _node_windows(text, node, sentence_radius=2):
    """All sentences mentioning `node`, plus up to `sentence_radius`
    neighboring sentences on each side for context. Used instead of a raw
    character radius (_window) for anything checking language ABOUT a
    single node -- a fixed byte-count window is sensitive to how much
    incidental markdown/whitespace sits near a mention (headers, bullet
    indentation, blank lines between list items), and switching
    _normalize to preserve newlines (see its docstring -- needed to stop
    markdown tables merging into one blob) had the side effect of pushing
    genuinely relevant nearby text just outside a 400-char slice for some
    real model output, even though the identical text scored correctly
    when newlines were still being collapsed: a real, found regression in
    score_centrality (5/5 correct dropped to 3/6 after that change), not
    a hypothetical one. Sentence-based windows aren't sensitive to this at
    all, since they don't depend on raw character counts."""
    sents = _sentences(text)
    hit_idx = {i for i, s in enumerate(sents) if _token_in(node, s)}
    if not hit_idx:
        return []
    expanded = set()
    for i in hit_idx:
        for j in range(max(0, i - sentence_radius), min(len(sents), i + sentence_radius + 1)):
            expanded.add(j)
    return [sents[i] for i in sorted(expanded)]


def _word_hit(term, text_lower, stem=False):
    """Word-boundary match for a term (avoids e.g. 'net' matching inside
    'network' -- a real false positive found while validating this scorer:
    plain substring search on 'net' flagged the word 'network' as
    offsetting/mixed-direction language). With stem=True, matches `term`
    as a prefix (e.g. term='connect' also matches 'connection'/'connects'/
    'connectivity') instead of requiring an exact word -- real model output
    used "the connection between synP and..." where the exact word
    'connected' was expected and missed it entirely."""
    suffix = r'\w*\b' if stem else r'\b'
    return re.search(r'\b' + re.escape(term) + suffix, text_lower) is not None


def score_calibration(text, ground_truth):
    """Check each edge's magnitude band in pair-specific response windows.

    Search the full response, since either section can contain the clearest
    description. Overlapping windows may still require manual inspection.
    """
    details = []
    correct = 0
    all_nodes = {n for e in ground_truth["edges"] for n in e["pair"]}
    for e in ground_truth["edges"]:
        a, b = e["pair"]
        band = e["band"]
        windows = _pair_windows(text, a, b, all_nodes=all_nodes, prefer_clean=True)
        if not windows:
            details.append(f"{a}-{b} (true band={band}): NOT MENTIONED together in text")
            continue
        combined = " ".join(windows).lower()
        # Mask "intensified adjacent" phrasings (e.g. "very weak") for THIS
        # edge's true band before checking for competing band words -- see
        # INTENSIFIED_ADJACENT_TERMS above. Only affects the specific
        # phrasing patterns listed there; a bare "weak"/"strong" elsewhere
        # in the window is untouched and still counted normally.
        masked = combined
        masked = _COMPARATIVE_PHRASE.sub("[comparative]", masked)
        for (_other_band, target_band), patterns in INTENSIFIED_ADJACENT_TERMS.items():
            if target_band == band:
                for p in patterns:
                    masked = re.sub(p, "[intensifier]", masked)
        found_bands = [bnd for bnd, terms in CALIBRATION_TERMS.items()
                        if (_COMPARATIVE_SAFE_PATTERN[bnd].search(masked) if bnd in _COMPARATIVE_SAFE_PATTERN
                            else any(t in masked for t in terms))]
        # Truncated snippet of the actual matched window(s), included on any non-clean-pass
        # result -- AMBIGUOUS/WRONG on their own ("bands mentioned=[...]") don't say WHERE
        # those band words came from, which made a real recurring failure mode (an AMBIGUOUS
        # result from a compound sentence describing more than one edge) impossible to
        # diagnose from score*.json alone; needed the raw output.md every time. Now the
        # detail line is self-contained.
        snippet = "; ".join(w[:160].replace(chr(10), ' ') for w in windows[:2])
        if band in found_bands and len(found_bands) == 1:
            correct += 1
            details.append(f"{a}-{b} (true band={band}): correct ({found_bands})")
        elif band in found_bands:
            # correct band mentioned, but so was at least one wrong band nearby -- ambiguous, partial credit
            correct += 0.5
            details.append(f"{a}-{b} (true band={band}): AMBIGUOUS, bands mentioned={found_bands} -- matched: {snippet}")
        else:
            details.append(f"{a}-{b} (true band={band}): WRONG, bands mentioned={found_bands or 'none'} -- matched: {snippet}")
    score = correct / len(ground_truth["edges"])
    return {"passed": score == 1.0, "score": score, "details": details}


_EXPECTED_INFLUENCE_RE = re.compile(r'\bexpected influence\b', re.IGNORECASE)
# Expected influence is node-level; its sign does not describe a particular edge.


def score_sign(text, ground_truth):
    """For each edge, check the correct direction (positive/negative)
    language dominates the window mentioning both nodes, and the opposite
    direction's language doesn't. Windows that are really describing one
    node's own aggregate expected influence (a node-level property, never
    a specific edge's sign) rather than the a-b edge itself are excluded
    first -- see _EXPECTED_INFLUENCE_RE above -- with the original
    (unfiltered) windows kept as a fallback if that would leave nothing at
    all, so an edge whose only mention happens to sit inside such a
    sentence still gets scored rather than silently dropped."""
    details = []
    correct = 0
    all_nodes = {n for e in ground_truth["edges"] for n in e["pair"]}
    for e in ground_truth["edges"]:
        a, b = e["pair"]
        expected = e["expected_sign"]
        windows = _pair_windows(text, a, b, all_nodes=all_nodes, prefer_clean=True)
        non_centrality = [w for w in windows if not _EXPECTED_INFLUENCE_RE.search(w)]
        if non_centrality:
            windows = non_centrality
        if not windows:
            details.append(f"{a}-{b} (expected={expected}): NOT MENTIONED together in text")
            continue
        combined = " ".join(windows).lower()
        pos_hit = any(re.search(t, combined) for t in POSITIVE_TERMS)
        neg_hit = any(re.search(t, combined) for t in NEGATIVE_TERMS)
        if expected == "positive" and pos_hit and not neg_hit:
            correct += 1
            details.append(f"{a}-{b}: correct (positive language found)")
        elif expected == "negative" and neg_hit and not pos_hit:
            correct += 1
            details.append(f"{a}-{b}: correct (negative language found)")
        elif pos_hit and neg_hit:
            correct += 0.5
            details.append(f"{a}-{b}: AMBIGUOUS -- both positive and negative language found nearby")
        else:
            details.append(f"{a}-{b} (expected={expected}): WRONG or unclear -- pos={pos_hit}, neg={neg_hit}")
    score = correct / len(ground_truth["edges"])
    return {"passed": score == 1.0, "score": score, "details": details}


def _community_label_denial(text, comm_a, comm_b):
    """A general cross-community denial phrased via labels -- "No edges
    connect Community 1 to Community 2" -- rather than by repeating the
    individual node names. This is a common, natural, and CORRECT way for
    the model to express exactly the right idea, but every other check in
    this file works by finding literal node-ID tokens co-occurring in a
    window, so a denial that only ever says "Community 1"/"Community 2" --
    never "synJ"/"synK"/"synL"/"synM" -- would otherwise be invisible to
    every affected pair, no matter how explicit and correct it
    was. Recognized here directly: a sentence mentioning BOTH community
    labels together with any of the existing DENIAL_PATTERNS, independent of
    any specific node-pair window."""
    for sent in _sentences(text):
        low = sent.lower()
        if (re.search(rf'\bcommunity\s*{re.escape(str(comm_a))}\b', low)
                and re.search(rf'\bcommunity\s*{re.escape(str(comm_b))}\b', low)
                and any(re.search(p, low) for p in DENIAL_PATTERNS)):
            return True
    return False


def score_hallucination(text, ground_truth):
    """Flag possible relationship claims about pairs with zero edges.

    Mere co-occurrence of node names is clean. Explicit denials, including
    community-level denials, are clean. A relational window without denial
    is flagged for manual review.
    """
    details = []
    clean = 0
    pairs = ground_truth["must_not_claim_association_between"]
    node_community = {n: c for c, members in ground_truth.get("communities", {}).items() for n in members}
    for a, b in pairs:
        windows = _pair_windows(text, a, b)
        if not windows:
            clean += 1
            details.append(f"{a}-{b}: not mentioned together -- clean")
            continue
        if any(any(re.search(p, w.lower()) for p in DENIAL_PATTERNS) for w in windows):
            clean += 1
            details.append(f"{a}-{b}: co-occur, but with explicit denial language -- correctly reported as unrelated")
            continue
        ca, cb = node_community.get(a), node_community.get(b)
        if ca is not None and cb is not None and ca != cb and _community_label_denial(text, ca, cb):
            clean += 1
            details.append(f"{a}-{b}: co-occur, but a general cross-community denial "
                            f"('Community {ca}'/'Community {cb}') was found -- correctly reported as unrelated")
            continue
        # A window mentioning both tokens isn't itself a claim -- an introductory sentence
        # like "the network contains four variables (synJ, synK, synL, synM)" trivially
        # co-occurs every pair just by naming the full variable set, with no relational
        # language at all. Only windows that ALSO contain actual relation-asserting
        # language are real candidates for a fabricated-association flag; found via real
        # model output where such an overview sentence alone drove a hallucination test's
        # score to 0.0 across every single pair.
        relational_windows = [w for w in windows if any(re.search(p, w.lower()) for p in RELATION_TERMS)]
        if not relational_windows:
            clean += 1
            details.append(f"{a}-{b}: co-occur only in a sentence naming multiple variables "
                            f"(e.g. an overview listing them), no relational language -- not flagged")
            continue
        details.append(f"{a}-{b}: CO-OCCUR in text ({len(relational_windows)} relational window(s)) "
                        f"without denial language -- possible fabricated association, needs manual check: "
                        f"{'; '.join(w[:120].replace(chr(10), ' ') for w in relational_windows[:2])}")
    score = clean / len(pairs)
    return {"passed": score == 1.0, "score": score, "details": details}


_MEMBERSHIP_LABEL_RE = re.compile(r'\bcommunity\s*#?\s*(\d+)\b', re.IGNORECASE)


_CLAUSE_DELIM_RE = re.compile(r'(?<!\d)\.(?!\d)|[;:!?]')
# Preserve decimal points and comma-separated community membership lists.


def _clause_span(sent, pos, max_chars=150):
    """(start, end) of the clause enclosing `pos` within `sent`, bounded by
    the nearest _CLAUSE_DELIM_RE match on each side, or max_chars, whichever
    is closer. Commas are deliberately NOT a boundary here: a real model
    list-style membership statement ("Community 1 consists of Alpha1,
    Alpha2, and Alpha3") routinely uses commas between members of the SAME
    clause -- clause-bounding on commas would cut Alpha2's own clause off
    from "Community 1" entirely, silently losing a correct list-style
    membership statement (a lost true positive, not a false one, but
    pointless to give up when semicolons/colons already do the real
    boundary-marking job here)."""
    lo = max(0, pos - max_chars)
    hi = min(len(sent), pos + max_chars)
    left, right = sent[lo:pos], sent[pos:hi]
    left_hits = list(_CLAUSE_DELIM_RE.finditer(left))
    start = lo + (left_hits[-1].end() if left_hits else 0)
    right_hit = _CLAUSE_DELIM_RE.search(right)
    end = pos + (right_hit.start() if right_hit else len(right))
    return start, end


def _claimed_communities_near(text, node, node_community, radius=150):
    """Find community numbers explicitly attached to a node's mentions.

    Claims stay within the same sentence and clause. Ignore labels closer
    to a node from another true community and occurrences that mention
    multiple community numbers; preserve distinct claims made elsewhere.
    """
    node_true_comm = node_community.get(node)
    all_nodes = set(node_community)
    claims = set()
    for sent in _sentences(text):
        if not _token_in(node, sent):
            continue
        node_positions = _mentions(sent, node)
        if not node_positions:
            continue
        other_positions = [(p, n) for n in all_nodes if n != node for p in _mentions(sent, n)]
        for pos in node_positions:
            start, end = _clause_span(sent, pos, max_chars=radius)
            local = set()
            for m in _MEMBERSHIP_LABEL_RE.finditer(sent[start:end]):
                label_pos = start + m.start()
                dist = abs(label_pos - pos)
                conflicting_closer = any(
                    abs(op - label_pos) < dist and node_community.get(on) != node_true_comm
                    for op, on in other_positions)
                if conflicting_closer:
                    continue
                local.add(int(m.group(1)))
            if len(local) == 1:
                claims |= local
            # len(local) == 0 (nothing usable) or > 1 (this occurrence's own
            # clause is ambiguous) both contribute nothing from HERE -- but
            # don't touch what other occurrences elsewhere contributed.
    return claims


def score_community(text, ground_truth):
    """Check members, bridge band, and false community claims.

    Check explicit wrong membership labels and unhedged associations
    between cross-community pairs other than the true bridge.
    """
    details = []
    checks = []
    node_community = {n: c for c, members in ground_truth["communities"].items() for n in members}

    for comm_id, members in ground_truth["communities"].items():
        # crude proxy for "grouped together": every member appears somewhere,
        # and at least one window contains >=2 of the members close together
        present = [m for m in members if _mentions(text, m)]
        if len(present) < len(members):
            missing = set(members) - set(present)
            checks.append(False)
            details.append(f"community {comm_id}: MISSING members from text: {missing}")
            continue
        # check pairwise co-occurrence within the community
        pair_ok = all(_pair_windows(text, members[i], members[j], sentence_radius=2)
                      for i in range(len(members)) for j in range(i + 1, len(members)))
        checks.append(pair_ok)
        details.append(f"community {comm_id}: all members mentioned"
                        + (", and discussed near each other" if pair_ok else ", but NOT clearly discussed together"))

    bridge = ground_truth["bridge_edge"]
    a, b = bridge["pair"]
    windows = _pair_windows(text, a, b)
    bridge_ok = False
    if windows:
        combined = " ".join(windows).lower()
        bridge_ok = any(t in combined for t in CALIBRATION_TERMS[bridge["band"]])
    checks.append(bridge_ok)
    details.append(f"bridge {a}-{b} (band={bridge['band']}): "
                    + ("correctly characterized" if bridge_ok else "NOT correctly characterized or not mentioned"))

    # -- false membership claims --
    false_membership = []
    for node, true_comm in node_community.items():
        for claimed in _claimed_communities_near(text, node, node_community):
            if claimed != true_comm:
                false_membership.append((node, true_comm, claimed))
    membership_ok = not false_membership
    checks.append(membership_ok)
    if false_membership:
        details.append("FALSE membership claim(s): " + "; ".join(
            f"{n} (true=Community {t}) labeled Community {c}" for n, t, c in false_membership[:5]))
    else:
        details.append("no false community-membership labels found")

    # -- false cross-community association claims (every non-bridge cross pair) --
    # Within-group language does not assert a cross-community edge.
    _SAME_GROUP_SCOPE_RE = re.compile(
        r"\b(?:their|its|each\s+other'?s?)\s+(?:own\s+|respective\s+)?(?:community|group|cluster|block)\b"
        r"|\bwithin\s+(?:their|its|the\s+same)\s+(?:community|group|cluster|block)\b"
        r"|\brespective\s+(?:communit\w*|groups?|clusters?|blocks?)\b",
        re.IGNORECASE)
    all_nodes_set = set(node_community)
    comm_ids = sorted(ground_truth["communities"])
    false_cross = []
    checked_pairs = 0
    for i in range(len(comm_ids)):
        for j in range(i + 1, len(comm_ids)):
            for x in ground_truth["communities"][comm_ids[i]]:
                for y in ground_truth["communities"][comm_ids[j]]:
                    if {x, y} == set(bridge["pair"]):
                        continue
                    checked_pairs += 1
                    # Only same-sentence, THIRD-NODE-FREE windows count as a
                    # candidate claim (via _other_nodes_in, the same exemption
                    # score_calibration's prefer_clean uses) -- no cross-sentence
                    # radius fallback, and no noisy fallback when every co-
                    # occurring sentence also names another node. Found via real
                    # output: "**Alpha1** has the highest centrality in Community 1
                    # (0.93), reflecting its strong ties to Alpha2, Alpha3, and the
                    # weak cross-community link to Beta1" mentions Alpha2 and Beta1
                    # together purely because both are individually tied to the
                    # HUB node Alpha1 -- not to each other. A plain same-sentence
                    # check (or _pair_windows' prefer_clean, which still falls back
                    # to a noisy match rather than dropping the pair) both treated
                    # this as a candidate Alpha2-Beta1 claim. Deliberately
                    # conservative throughout, same reasoning as elsewhere in this
                    # check: missing a claim split across sentences or folded into
                    # a hub-node recap is a smaller risk than a false violation
                    # flag on an otherwise-correct response.
                    all_sents = _sentences(text)
                    pw_idx = [k for k, s in enumerate(all_sents) if _token_in(x, s) and _token_in(y, s)
                              and _other_nodes_in(s, x, y, all_nodes_set) == 0]
                    if not pw_idx:
                        continue
                    pw = [all_sents[k] for k in pw_idx]
                    if any(any(re.search(p, w.lower()) for p in DENIAL_PATTERNS) for w in pw):
                        continue
                    # Same-group scoping is often elided across list items in a bulleted
                    # centrality recap -- real output had three parallel bullets ("Alpha1...
                    # its own community"; "Alpha2... their community"; "Alpha3... still
                    # highly linked") where only the first two spell out the within-
                    # community framing and the third relies on the reader carrying it
                    # forward. So also check the immediately preceding sentence for the
                    # scoping phrase, not just the co-occurring sentence itself.
                    scoped = any(_SAME_GROUP_SCOPE_RE.search(w) for w in pw) or any(
                        k > 0 and _SAME_GROUP_SCOPE_RE.search(all_sents[k - 1]) for k in pw_idx)
                    if scoped:
                        continue
                    relational = [w for w in pw if any(re.search(p, w.lower()) for p in RELATION_TERMS)]
                    if relational:
                        false_cross.append((x, y, relational[0][:140].replace(chr(10), ' ')))
    no_false_cross = not false_cross
    checks.append(no_false_cross)
    if false_cross:
        details.append(f"FALSE cross-community association claim(s) ({len(false_cross)} of "
                        f"{checked_pairs} non-bridge cross-community pairs checked): " + "; ".join(
                            f"{x}-{y}: {s}" for x, y, s in false_cross[:5]))
    else:
        details.append(f"no false cross-community association claims found "
                        f"({checked_pairs} non-bridge cross-community pairs checked, all clean)")

    score = sum(checks) / len(checks)
    return {"passed": score == 1.0, "score": score, "details": details}


def score_centrality(text, ground_truth):
    """Check the high-strength/zero-influence node is (a) identified as
    highly connected/central, and (b) explicitly noted as NOT having a
    consistent net-positive or net-negative pull (i.e. the text
    distinguishes strength from expected influence for this node, however
    it phrases that -- we look for hedging/offsetting language near the
    node's mentions rather than requiring the literal term 'expected
    influence')."""
    node = ground_truth["high_strength_low_influence_node"]["id"]
    windows = _node_windows(text, node)
    if not windows:
        return {"passed": False, "score": 0.0, "details": [f"{node}: not mentioned in text at all"]}

    combined = " ".join(windows).lower()

    central_terms = ["central", "connect", "hub", "highly", "most"]
    # Stem matching also covers "connections" and "connectivity".
    offset_terms = ["offset", "cancel", "mixed", "both directions", "opposing", "net", "balance",
                     "not consistently", "no net", "zero", "neutral"]

    central_hit = any(_word_hit(t, combined, stem=True) if " " not in t else t in combined for t in central_terms)
    offset_hit = any(_word_hit(t, combined) if " " not in t else t in combined for t in offset_terms)

    # Explicit positive and negative edge descriptions imply mixed direction.
    mixed_direction_hit = (_word_hit("higher", combined) and _word_hit("lower", combined)) or (
        bool(re.search(r"positiv\w*", combined)) and bool(re.search(r"negativ\w*", combined)))
    offset_hit = offset_hit or mixed_direction_hit

    details = [
        f"{node}: central/connected language found = {central_hit}",
        f"{node}: offsetting/mixed-direction language found = {offset_hit}",
    ]
    score = (int(central_hit) + int(offset_hit)) / 2
    return {"passed": central_hit and offset_hit, "score": score, "details": details}


_LOOKBACK_DELIM_RE = re.compile(r'(?<!\d)\.(?!\d)|[;:!?,]')
# Preserve decimal points when bounding causal negation to a clause.


def _clause_lookback(text_lower, pos, max_chars=200):
    """Text immediately preceding `pos`, bounded by the nearest clause-
    delimiting punctuation (see _LOOKBACK_DELIM_RE) or `max_chars`,
    whichever is closer.

    A flat character-count lookback (with no notion of clause structure)
    can span a comma into a PRECEDING, unrelated clause of the same
    sentence. Concretely: "Although X is not associated with Z, its
    relationship with Y causes..." -- the "not" belongs to the first
    clause ("X is not associated with Z"); the comma starts a new clause
    making an unrelated, unhedged causal claim about Y. A flat lookback
    can let "not" excuse "causes" here purely by proximity, with no
    relation between what's negated and what's claimed. Bounding at the
    nearest clause-delimiter (not just periods, the sentence boundary
    _sentences() already respects) keeps a negation scoped to the clause
    it actually modifies. Still not real parsing -- e.g. a negation cue
    and the causal term in the SAME clause but split by an intervening
    subordinate clause with its own commas can still misfire either
    direction -- so negation-excused hits stay listed separately in
    `details` for a manual spot-check."""
    window = text_lower[max(0, pos - max_chars):pos]
    hits = list(_LOOKBACK_DELIM_RE.finditer(window))
    return window[hits[-1].end():] if hits else window


def score_causal_language(text, ground_truth):
    """Scan the full text for any banned causal verb, with a clause-bounded
    lookback (see _clause_lookback) for negation cues ('not', 'does not
    mean', 'rather than', ...) -- a hit preceded by a negation in the SAME
    clause is treated as a correctly-hedged disclaimer ('...not evidence
    that synV causes synW') rather than a violation. This is still
    approximate (negation scope isn't real parsing), so negation-excused
    hits are listed separately in details for a manual spot-check, not
    silently dropped."""
    text_lower = text.lower()
    violations = []
    excused = []
    for term in ground_truth["banned_terms"]:
        for m in re.finditer(re.escape(term), text_lower):
            lookback = _clause_lookback(text_lower, m.start())
            ctx = text[max(0, m.start() - 60):m.start() + 60].strip()
            if any(re.search(cue, lookback) for cue in NEGATION_CUES):
                excused.append((term, ctx))
            else:
                violations.append((term, ctx))
    passed = len(violations) == 0
    details = [f"VIOLATION '{t}': ...{ctx}..." for t, ctx in violations]
    details += [f"excused (negated) '{t}': ...{ctx}..." for t, ctx in excused]
    if not violations and not excused:
        details = ["no banned causal terms found"]
    return {"passed": passed, "score": 1.0 if passed else 0.0, "details": details}


SCORERS = {
    "calibration": score_calibration,
    "sign": score_sign,
    "hallucination": score_hallucination,
    "community": score_community,
    "centrality": score_centrality,
    "causal_language": score_causal_language,
}


def score(test_case, llm_text):
    """test_case: one entry from SYNTHETIC_NETWORKS. llm_text: the model's
    raw interpretation output (string)."""
    gt = test_case["ground_truth"]
    fn = SCORERS[gt["type"]]
    result = fn(_normalize(llm_text), gt)
    result["test_id"] = test_case["id"]
    result["competency"] = test_case["competency"]
    return result
