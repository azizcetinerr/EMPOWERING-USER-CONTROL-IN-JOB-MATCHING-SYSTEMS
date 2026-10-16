"""Unit checks for the experiment code (no model downloads). Run: python -m pytest tests/test_experiments.py"""
import numpy as np

from experiments import core, metrics
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


def test_language_filter_for_listing_skills():
    stops = core.stopword_lists(["en", "es"])
    assert core.detect_language("Los voluntarios trabajan en la escuela con los niños", stops) == "es"
    assert core.detect_language("The volunteer will work with the children at the school", stops) == "en"
    assert core.detect_language("", stops) == "unknown"
    lists = core.filter_listing_skills(
        [["communication", "en", "teamwork"], ["los", "communication", "persona"]], ["en", "es"], stops)
    assert lists == [["communication", "teamwork"], ["communication"]]
