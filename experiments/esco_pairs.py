"""ESCO skill pairs for the encoder check (v3).

  python -m experiments.esco_pairs            # writes experiments/esco_pairs.csv (+ .meta.json)

Pair types (each ~n pairs, fixed seed):
  synonym         related   preferredLabel of a skill and one of its altLabels (same concept)
  broader         related   a skill and its direct broader concept (broaderRelationsSkillPillar)
  unrelated_easy  unrelated two skills whose top-level (Level 0) groups do not overlap
  unrelated_hard  unrelated two skills in the same Level 1 group, >= 3 steps apart in the
                            hierarchy and not linked in skillSkillRelations
Every concept URI and every label is used at most once in the whole file (no leakage).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ESCO_DIR = ROOT / "data" / "external" / "esco"
TYPES = ["synonym", "broader", "unrelated_easy", "unrelated_hard"]


def load_esco(d: Path = ESCO_DIR) -> dict:
    s = pd.read_csv(d / "skills_en.csv")
    s = s[s["status"] == "released"]
    b = pd.read_csv(d / "broaderRelationsSkillPillar_en.csv")
    h = pd.read_csv(d / "skillsHierarchy_en.csv")
    rel = pd.read_csv(d / "skillSkillRelations_en.csv")

    label = dict(zip(s["conceptUri"], s["preferredLabel"].str.strip()))
    label.update({u: l.strip() for u, l in zip(b["broaderUri"], b["broaderLabel"]) if u not in label})
    label.update({u: l.strip() for u, l in zip(b["conceptUri"], b["conceptLabel"]) if u not in label})
    alts = {u: [a.strip() for a in str(x).split("\n") if a.strip()] if isinstance(x, str) else []
            for u, x in zip(s["conceptUri"], s["altLabels"])}
    parents = defaultdict(set)
    for c, p in zip(b["conceptUri"], b["broaderUri"]):
        parents[c].add(p)
    codes = {}
    for _, r in h.iterrows():
        for lv in range(4):
            u = r[f"Level {lv} URI"]
            if isinstance(u, str):
                codes[u] = (r["Level 0 code"], r["Level 1 code"] if lv >= 1 else None)
    linked = {frozenset(p) for p in zip(rel["originalSkillUri"], rel["relatedSkillUri"])}
    return {"skills": sorted(s["conceptUri"]), "label": label, "alts": alts,
            "parents": dict(parents), "codes": codes, "linked": linked}


def adjacency(parents: dict) -> dict:
    adj = defaultdict(set)
    for c, ps in parents.items():
        for p in ps:
            adj[c].add(p)
            adj[p].add(c)
    return adj


def within_two_steps(a, b, adj) -> bool:
    return a == b or b in adj[a] or bool(adj[a] & adj[b])


def group_codes(uri, parents, codes) -> tuple[set, set]:
    """Level 0 and Level 1 codes of every hierarchy group above a concept."""
    l0, l1, seen, stack = set(), set(), set(), [uri]
    while stack:
        u = stack.pop()
        if u in seen:
            continue
        seen.add(u)
        if u in codes:
            c0, c1 = codes[u]
            l0.add(c0)
            if c1:
                l1.add(c1)
        stack.extend(parents.get(u, ()))
    return l0, l1


def build_pairs(esco: dict, n: int = 1000, seed: int = 2026, max_tries: int = 2_000_000) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label, alts, parents, linked = esco["label"], esco["alts"], esco["parents"], esco["linked"]
    adj = adjacency(parents)
    used_uri, used_label, rows = set(), set(), []

    def free(u, *labels):
        return u not in used_uri and all(l.lower() not in used_label for l in labels)

    def take(kind, ua, la, ub, lb, related):
        used_uri.update((ua, ub))
        used_label.update((la.lower(), lb.lower()))
        rows.append({"skill_a": la, "skill_b": lb, "related": related, "pair_type": kind,
                     "esco_uri_a": ua, "esco_uri_b": ub})

    skills = list(esco["skills"])
    for u in rng.permutation(skills):
        if sum(r["pair_type"] == "synonym" for r in rows) >= n:
            break
        la = label[u]
        cands = [a for a in alts.get(u, []) if a.lower() != la.lower() and a.lower() not in used_label]
        if free(u, la) and cands:
            take("synonym", u, la, u, cands[rng.integers(len(cands))], 1)

    for u in rng.permutation(skills):
        if sum(r["pair_type"] == "broader" for r in rows) >= n:
            break
        ps = [p for p in sorted(parents.get(u, ())) if p in label and free(p, label[p])]
        if free(u, label[u]) and ps:
            p = ps[rng.integers(len(ps))]
            if label[p].lower() != label[u].lower():
                take("broader", u, label[u], p, label[p], 1)

    gc = {u: group_codes(u, parents, esco["codes"]) for u in skills}
    by_l1 = defaultdict(list)
    for u, (_, l1) in gc.items():
        for c in l1:
            by_l1[c].append(u)
    l1_keys = sorted(by_l1)
    l1_w = np.array([len(by_l1[k]) for k in l1_keys], float)
    l1_w /= l1_w.sum()

    def negative_ok(a, b):
        return (a != b and free(a, label[a]) and free(b, label[b]) and label[a].lower() != label[b].lower()
                and frozenset((a, b)) not in linked and not within_two_steps(a, b, adj))

    tries = 0
    while sum(r["pair_type"] == "unrelated_hard" for r in rows) < n and tries < max_tries:
        tries += 1
        grp = by_l1[l1_keys[rng.choice(len(l1_keys), p=l1_w)]]
        a, b = grp[rng.integers(len(grp))], grp[rng.integers(len(grp))]
        if negative_ok(a, b):
            take("unrelated_hard", a, label[a], b, label[b], 0)

    tries = 0
    while sum(r["pair_type"] == "unrelated_easy" for r in rows) < n and tries < max_tries:
        tries += 1
        a, b = skills[rng.integers(len(skills))], skills[rng.integers(len(skills))]
        if gc[a][0] and gc[b][0] and not (gc[a][0] & gc[b][0]) and negative_ok(a, b):
            take("unrelated_easy", a, label[a], b, label[b], 0)

    return pd.DataFrame(rows, columns=["skill_a", "skill_b", "related", "pair_type", "esco_uri_a", "esco_uri_b"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--esco-version", default="v1.2.1 (package \"ESCO dataset - v1.2.1 - classification - en - csv.zip\", md5 59fd2791cd60dc96405a5d8c0c35436f; CSV files dated 2025-12-16; latest modifiedDate field 2025-12-15)")
    ap.add_argument("--out", default=str(ROOT / "experiments" / "esco_pairs.csv"))
    a = ap.parse_args(argv)
    pairs = build_pairs(load_esco(), a.n, a.seed)
    pairs.to_csv(a.out, index=False)
    meta = {"esco_version": a.esco_version, "esco_files": ["skills_en.csv", "broaderRelationsSkillPillar_en.csv",
                                                           "skillsHierarchy_en.csv", "skillSkillRelations_en.csv"],
            "seed": a.seed, "n_per_type": a.n, "counts": pairs["pair_type"].value_counts().to_dict(),
            "definitions": __doc__.split("Pair types")[1].strip()}
    Path(a.out).with_suffix(".meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(pairs["pair_type"].value_counts().to_string())


if __name__ == "__main__":
    main()
