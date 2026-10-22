"""Current status of each December 2025 snapshot listing (by opid) from the Youth Portal search API.
Appends to status_<date>.jsonl; re-running skips opids already fetched. Usage: python fetch_status.py [max_requests]"""
import json, re, sys, time, datetime, urllib.request, pathlib
import pandas as pd
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
today = datetime.date.today().isoformat()
out = HERE / f"status_{today}.jsonl"
d = pd.read_csv(ROOT / "outputs/results/IP5_S_recomputed.csv", encoding="utf-8-sig")
dec = sorted({int(re.search(r"/(\d+)_", l).group(1)) for l in d["link"]})
done = set()
if out.exists():
    done = {json.loads(l)["opid"] for l in out.read_text().splitlines() if l.strip()}
F = ["opid","status","date_application_end","has_no_deadline","date_start","date_end","created","changed"]
base = "https://youth.europa.eu/api/rest/eyp/v1/search_en?type=Opportunity&size=2&filters%5Bopid%5D={}&" + "&".join(f"fields%5B{i}%5D={f}" for i,f in enumerate(F))
from concurrent.futures import ThreadPoolExecutor
import threading
limit = int(sys.argv[1]) if len(sys.argv) > 1 else 10**9
todo = [o for o in dec if o not in done][:limit]
lock = threading.Lock(); n = 0
def one(o):
    req = urllib.request.Request(base.format(o), headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
    try:
        hits = json.load(urllib.request.urlopen(req, timeout=30))["hits"]["hits"]
    except Exception:
        return None
    rec = hits[0]["_source"] if hits else {"opid": o, "status": "not_found"}
    rec["opid"] = o; return rec
with out.open("a") as fh, ThreadPoolExecutor(6) as ex:
    for rec in ex.map(one, todo):
        if rec: fh.write(json.dumps(rec) + "\n"); fh.flush(); n += 1
print("fetched", n, "total done", len(done) + n, "of", len(dec))
