"""Unit checks for the experiment code (no model downloads). Run: python -m pytest tests/test_experiments.py"""
import numpy as np

from experiments import core, esco_pairs, metrics, talentclef
from experiments.encoders import split_into_chunks


def test_ndcg_perfect_and_reversed():
    g = [2, 1, 0, 0]
    assert metrics.ndcg_at_k(g, g, 3) == 1.0
    assert metrics.ndcg_at_k(g[::-1], g, 3) < 0.5
    assert metrics.precision_at_k([2, 1, 0, 0], 4) == 0.5


def test_condensed_list_drops_unjudged():
    assert metrics.condensed(["a", "x", "b"], {"a": 2, "b": 0}) == [2, 0]


def test_rank_normalization_and_fusion():
    r = core.normalize([3, 1, 2], "rank")
    assert list(r) == [100, 0, 50]
    f = core.fuse({"M": [1, 2, 3], "L": [3, 2, 1]}, {"M": 0.5, "L": 0.5}, "rank")
    assert np.allclose(f, 50)


def test_mmr_off_and_diversity():
    emb = np.array([[1, 0], [1, 0], [0, 1]], dtype=float)
    s = [3, 2, 1]
    assert list(core.mmr_order(s, emb, 3, 1.0)) == [0, 1, 2]
    # With a strong diversity weight the orthogonal item moves ahead of the duplicate
    assert list(core.mmr_order(s, emb, 3, 0.3))[:2] == [0, 2]


def test_matching_coverage():
    best = {"a": 0.9, "b": 0.1}
    idf = {"a": 1.0, "b": 1.0}
    m, mm = core.matching_scores([["a", "b"], []], best, idf, tau_low=0.2, tau_high=0.8)
    assert np.isclose(m[0], 50.0) and m[1] == 0.0
    assert np.isclose(mm[0], 50.0)


def test_clean_skills():
    assert core.clean_skills(["Com", "Team  Work", "team work", "e"], {"com", "e"}, 2) == ["team work"]


def test_shrink_to_median():
    lists = [["a"], ["b"] * 1 + ["c", "d", "e", "f", "g", "h", "i", "j"], []]
    idf = {s: 1.0 for s in "abcdefghij"}
    out = core.shrink_to_median([100.0, 60.0, 0.0], lists, idf, k=5)
    # median over listings with skills = 80; one skill -> (1*100 + 5*80)/6, nine -> (9*60 + 5*80)/14
    assert np.isclose(out[0], 500 / 6) and np.isclose(out[1], 940 / 14) and np.isclose(out[2], 80.0)


def test_shrink_without_prior_keeps_scores_and_fills_empty():
    out = core.shrink_to_median([100.0, 60.0, 0.0], [["a"], ["b"], []], {"a": 1.0, "b": 1.0}, k=0)
    assert np.allclose(out, [100.0, 60.0, 80.0])


def test_split_into_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = split_into_chunks(text, size=120, overlap=20)
    assert [c.split()[0] for c in chunks] == ["w0", "w100", "w200"]
    assert chunks[0].split()[-1] == "w119" and chunks[-1].split()[-1] == "w249"
    assert split_into_chunks("short text", size=120, overlap=20) == ["short text"]


def test_chunk_scores():
    chunks = [np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([[0.6, 0.8]])]
    q = np.array([[0.0, 1.0]])
    assert np.allclose(core.max_chunk_cosine(q, chunks), [1.0, 0.8])
    assert np.allclose(core.mean_chunk_embedding(chunks), [[2 ** -0.5, 2 ** -0.5], [0.6, 0.8]])


def _toy_esco():
    # S -> S1 -> {gA, gB}; K -> K1. Skills a0-a9 under gA, b0-b9 under gB, k0-k19 under K1.
    parents = {"S1": {"S"}, "gA": {"S1"}, "gB": {"S1"}, "K1": {"K"}}
    skills = [f"a{i}" for i in range(10)] + [f"b{i}" for i in range(10)] + [f"k{i}" for i in range(20)]
    for s in skills:
        parents[s] = {"gA" if s[0] == "a" else "gB" if s[0] == "b" else "K1"}
    label = {u: f"label {u}" for u in skills + ["S", "S1", "gA", "gB", "K", "K1"]}
    codes = {"S": ("S", None), "S1": ("S", "S1"), "gA": ("S", "S1"), "gB": ("S", "S1"),
             "K": ("K", None), "K1": ("K", "K1")}
    return {"skills": skills, "label": label, "alts": {s: [f"alt {s}"] for s in skills},
            "parents": parents, "codes": codes, "linked": set()}


def test_esco_pairs_types_constraints_and_no_leakage():
    esco = _toy_esco()
    pairs = esco_pairs.build_pairs(esco, n=3, seed=1)
    assert pairs["pair_type"].value_counts().to_dict() == {t: 3 for t in esco_pairs.TYPES}
    uris = list(pairs["esco_uri_a"]) + list(pairs.loc[pairs.pair_type != "synonym", "esco_uri_b"])
    labels = [l.lower() for l in list(pairs["skill_a"]) + list(pairs["skill_b"])]
    assert len(uris) == len(set(uris)) and len(labels) == len(set(labels))
    adj = esco_pairs.adjacency(esco["parents"])
    for r in pairs.itertuples():
        ga = esco_pairs.group_codes(r.esco_uri_a, esco["parents"], esco["codes"])
        gb = esco_pairs.group_codes(r.esco_uri_b, esco["parents"], esco["codes"])
        if r.pair_type == "synonym":
            assert r.esco_uri_a == r.esco_uri_b and r.skill_b == f"alt {r.esco_uri_a}"
        elif r.pair_type == "broader":
            assert r.esco_uri_b in esco["parents"][r.esco_uri_a]
        elif r.pair_type == "unrelated_easy":
            assert not (ga[0] & gb[0])
        else:
            assert ga[1] & gb[1] and not esco_pairs.within_two_steps(r.esco_uri_a, r.esco_uri_b, adj)
    assert (pairs["related"] == pairs["pair_type"].isin(["synonym", "broader"]).astype(int)).all()


def test_talentclef_query_metrics():
    m = talentclef.query_metrics(["a", "x", "b", "y"], {"a", "b", "z"})
    assert np.isclose(m["map"], (1 / 1 + 2 / 3) / 3)
    assert np.isclose(m["p@10"], 0.2) and np.isclose(m["recall@100"], 2 / 3)
    ideal = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert np.isclose(m["ndcg@10"], (1 + 1 / np.log2(4)) / ideal)
    assert talentclef.query_metrics(["a", "b"], {"a", "b"})["ndcg@10"] == 1.0


def test_talentclef_paired_tests_and_holm():
    a = np.array([0.9, 0.8, 0.7, 0.9, 0.85, 0.8, 0.75, 0.9, 0.95, 0.8])
    assert talentclef.paired_randomization(a, a, 2000) == 1.0
    assert talentclef.paired_randomization(a, a - 0.3, 2000) < 0.01
    lo, hi = talentclef.bootstrap_ci(np.full(10, 0.3), 500)
    assert np.isclose(lo, 0.3) and np.isclose(hi, 0.3)
    adj = talentclef.holm({"x": 0.01, "y": 0.04, "z": 0.03})
    assert np.isclose(adj["x"], 0.03) and np.isclose(adj["z"], 0.06) and np.isclose(adj["y"], 0.06)


def test_talentclef_skill_scores():
    sim = {("s1", "c1"): 0.9, ("s1", "c2"): 0.1, ("s2", "c1"): 0.2, ("s2", "c2"): 0.5}
    mm, cov = talentclef.skill_scores(["s1", "s2"], ["c1", "c2"], sim, {"s1": 1.0, "s2": 1.0}, 0.2, 0.8)
    assert np.isclose(mm, 70.0) and np.isclose(cov, 75.0)
    assert talentclef.skill_scores(["s1"], [], sim, {"s1": 1.0}, 0.2, 0.8) == (0.0, 0.0)


def test_language_filter_for_listing_skills():
    stops = core.stopword_lists(["en", "es"])
    assert core.detect_language("Los voluntarios trabajan en la escuela con los niños", stops) == "es"
    assert core.detect_language("The volunteer will work with the children at the school", stops) == "en"
    assert core.detect_language("", stops) == "unknown"
    lists = core.filter_listing_skills(
        [["communication", "en", "teamwork"], ["los", "communication", "persona"]], ["en", "es"], stops)
    assert lists == [["communication", "teamwork"], ["communication"]]
