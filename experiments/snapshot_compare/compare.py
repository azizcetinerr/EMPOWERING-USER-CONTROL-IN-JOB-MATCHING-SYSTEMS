"""Compare the December 2025 snapshot (939 distinct listings scored by the corrected ranker) with later open sets:
outputs/api_cache.json (portal API response cached on 4 Feb 2026) and open_2026-10-18.json (fetch_open.py).
Writes compare.json next to this script."""
import json, re, pathlib, collections
import pandas as pd
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
d = pd.read_csv(ROOT / "outputs/results/IP5_S_recomputed.csv", encoding="utf-8-sig")
dec = {int(re.search(r"/(\d+)_", l).group(1)) for l in d["link"]}
feb_c = json.loads((ROOT / "outputs/api_cache.json").read_text())
feb = {int(h["_source"]["opid"]) for h in feb_c["api_data"]["hits"]["hits"]}
octj = json.loads((HERE / "open_2026-10-18.json").read_text())
items = {int(s["opid"]): s for s in octj["items"]}; octo = set(items)
rk = json.loads((ROOT / "experiments/out/rankings.json").read_text())["rankings"]
res = {
  "dec_snapshot_listings": len(dec),
  "feb": {"date": feb_c["timestamp"][:10], "open": len(feb), "of_dec_still_open": len(dec & feb)},
  "oct": {"date": octj["date"], "open": len(octo), "api_total": octj["api_total"],
          "of_dec_still_open": len(dec & octo), "new_vs_dec": len(octo - dec),
          "no_deadline_among_open": sum(bool(items[o].get("has_no_deadline")) for o in octo),
          "no_deadline_among_dec_still_open": sum(bool(items[o].get("has_no_deadline")) for o in dec & octo)},
  "top_still_open_oct": {n: {"top10": sum(int(x) in octo for x in rk[n][:10]), "top20": sum(int(x) in octo for x in rk[n][:20])}
                         for n in ["Fused (v3)", "original_A (reported run)", "BM25 (profile)", "Dense (profile)"]},
}
(HERE / "compare.json").write_text(json.dumps(res, indent=2))
print(json.dumps(res, indent=2))
