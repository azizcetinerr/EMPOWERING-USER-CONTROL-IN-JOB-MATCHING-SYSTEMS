"""Fetch all open opportunities from the European Youth Portal search API with the same filters
as src/data_collection/youth_portal_scraper.py (Dec 2025 snapshot), paginated (the API returns at most 1000 per call).
Writes open_<date>.json (list of _source dicts) next to this script."""
import json, sys, time, datetime, urllib.request, urllib.parse, pathlib
HERE = pathlib.Path(__file__).resolve().parent
today = datetime.date.today().isoformat()
FIELDS = ["opid","title","town","country","has_no_deadline","duration","date_start","date_end",
          "date_application_end","is_esc_related","created","changed","status"]
def url(frm, size=1000):
    q = [("type","Opportunity"),("size",str(size)),("from",str(frm)),("filters[status]","open"),
         ("filters[date_end][operator]",">="),("filters[date_end][value]",today),("filters[date_end][type]","must")]
    for i,fid in enumerate([5,4,3,2,1,8,6,7]): q.append((f"filters[funding_programme][id][{i}]",str(fid)))
    q += [("filters[date_application_end][operator]",">="),("filters[date_application_end][value]",today),
          ("filters[date_application_end][type]","must"),("filters[date_application_end][group]","deadline"),
          ("filters[has_no_deadline][value]","true"),("filters[has_no_deadline][type]","must"),
          ("filters[has_no_deadline][group]","deadline")]
    q += [(f"fields[{i}]",f) for i,f in enumerate(FIELDS)]
    return "https://youth.europa.eu/api/rest/eyp/v1/search_en?"+urllib.parse.urlencode(q, safe="[]>=")
def get(u):
    req = urllib.request.Request(u, headers={"User-Agent":"Mozilla/5.0","Accept":"application/json"})
    return json.load(urllib.request.urlopen(req, timeout=60))
first = get(url(0)); total = first["hits"]["total"]["value"]; hits = first["hits"]["hits"]
while len(hits) < total:
    time.sleep(1); page = get(url(len(hits)))["hits"]["hits"]
    if not page: break
    hits += page
src = [h["_source"] for h in hits]
out = HERE / f"open_{today}.json"
out.write_text(json.dumps({"date": today, "api_total": total, "n": len(src), "items": src}, ensure_ascii=False))
print(today, "api_total", total, "fetched", len(src), "distinct opid", len({s.get('opid') for s in src}))
