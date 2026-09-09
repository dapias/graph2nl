#!/usr/bin/env python3
"""
test_scorer_examples.py -- sanity-check the scorer itself.

For each synthetic test, hand-writes one deliberately CORRECT interpretation
(should score 1.0 / pass) and one deliberately FLAWED interpretation (should
score < 1.0 / fail) targeting exactly the competency being tested, and checks
the scorer tells them apart. This is the validator-of-the-validator step --
if this doesn't pass, the scorer isn't trustworthy enough to use on real LLM
output yet.

Usage: python3 -m graph2nl_core.validation.test_scorer_examples
"""
from graph2nl_core.validation.synthetic_networks import get_network
from graph2nl_core.validation.scorer import score

EXAMPLES = {
    "calibration": {
        "good": """
The network shows four associations with item synA. The link between synA
and synB is negligible (0.02), essentially no relationship. The link between
synA and synC is weak (0.10). The link between synA and synD is moderate
(0.22). The link between synA and synE is strong (0.45), clearly the most
substantial association in this network.
""",
        "bad": """
The network shows four associations with item synA. The link between synA
and synB is weak. The link between synA and synC is also fairly weak. The
link between synA and synD is strong. The link between synA and synE is only
moderate, a middling effect.
""",
    },
    "sign_direction": {
        "good": """
synF and synG are positively associated -- higher scores on synF go together
with higher scores on synG. In contrast, synF and synH show a negative
association: higher synF is linked to lower synH, an inverse relationship.
""",
        "bad": """
synF and synG show an inverse relationship, with one rising as the other
falls. synF and synH, meanwhile, are positively associated and tend to move
together.
""",
    },
    "hallucination": {
        "good": """
The only notable association in this network is between synJ and synK
(partial r = 0.35), a moderately strong link. No other pairs among synJ,
synK, synL, and synM showed any notable association in this data.
""",
        "bad": """
synJ and synK are linked (0.35). Interestingly, synL and synM also appear
closely related, likely reflecting a shared underlying tendency, and synJ
shows a weak connection to synM as well.
""",
    },
    "community_grounding": {
        "good": """
Two clear clusters emerge. The first group -- Alpha1, Alpha2, and Alpha3 --
are all strongly interconnected with each other. The second group -- Beta1,
Beta2, and Beta3 -- likewise form a tightly connected set among themselves.
A single weak bridge (0.06) connects Alpha1 to Beta1, linking the two
clusters only faintly.
""",
        "bad": """
Two clusters emerge. The first group -- Alpha1, Beta2, and Alpha3 -- are
strongly interconnected. The second group -- Beta1, Alpha2, and Beta3 --
also form a tight set. A strong bridge connects Alpha1 to Beta1.
""",
    },
    "centrality_nuance": {
        "good": """
synP is the most highly connected node in the network, with more total
connection strength than any other item. However, its connections point in
both directions -- two positive and two negative, of equal size -- so they
largely cancel out, leaving synP with no net or consistent influence despite
being the most central node.
""",
        "bad": """
synP is clearly the most important and influential item in this network,
strongly and positively driving all of the other variables it connects to.
""",
    },
    "causal_avoidance": {
        "good": """
synV and synW show a strong association (partial r = 0.70). Workers who
report more of synV also tend to report more of synW, net of everything
else in the network. This is a cross-sectional association, not evidence
that one causes the other.
""",
        "bad": """
synV clearly drives synW -- the strong link suggests that increases in synV
directly cause corresponding increases in synW, and this effect leads to
substantial downstream consequences.
""",
    },
}


def main():
    total = 0
    correct_direction = 0
    for test_id, examples in EXAMPLES.items():
        test_case = get_network(test_id)
        for label, text in examples.items():
            result = score(test_case, text)
            total += 1
            expect_pass = (label == "good")
            got_pass = result["passed"]
            ok = "OK" if got_pass == expect_pass else "**MISMATCH**"
            if got_pass == expect_pass:
                correct_direction += 1
            print(f"[{ok}] {test_id:22s} {label:5s} -> passed={got_pass} score={result['score']:.2f}")
            for d in result["details"]:
                print(f"         - {d}")
        print()

    print(f"\n{correct_direction}/{total} examples scored in the expected direction.")
    if correct_direction < total:
        print("Scorer has mismatches -- review the details above before trusting it on real output.")
    else:
        print("Scorer correctly discriminated every hand-written good/bad pair.")


if __name__ == "__main__":
    main()
