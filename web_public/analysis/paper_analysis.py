#!/usr/bin/env python3
"""
Produces metrics, a table and a paragraph for the paper from the user data collected on the site.

Data source (choose one):
  --repo  USER/fair-pass-user-data        (private HF dataset repository; requires hf auth login)
  --folder path/                          (downloaded .jsonl files)
  --file fairpass_events.jsonl            (file downloaded with /admin/download)

Example:
  python web_public/analysis/paper_analysis.py --repo azizcetiner/fair-pass-user-data
  python web_public/analysis/paper_analysis.py --file ~/Downloads/fairpass_events.jsonl --start 2026-10-20

Outputs (web_public/analysis/output/):
  summary.json                   all metrics
  events.csv                     raw events (table)
  user_study_table.tex           IEEEtran-compatible table
  paper_paragraph_en.txt         English paragraph draft with the numbers filled in
"""

import argparse
import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import metrics  # noqa: E402
from event_log import read_jsonl  # noqa: E402


def load_events(args):
    if args.repo:
        from huggingface_hub import snapshot_download
        path = snapshot_download(repo_id=args.repo, repo_type="dataset",
                                local_dir=str(HERE / "output" / "raw_data"), allow_patterns=["data/*.jsonl"])
        files = sorted(Path(path).rglob("*.jsonl"))
    elif args.folder:
        files = sorted(Path(args.folder).expanduser().rglob("*.jsonl"))
    else:
        files = [Path(d).expanduser() for d in args.file]
    events = read_jsonl(files)
    # Deduplicate if the same event came from two sources
    seen, unique = set(), []
    for o in events:
        key = json.dumps(o, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            unique.append(o)
    if args.start:
        unique = [o for o in unique if o.get("time", "") >= args.start]
    if args.end:
        unique = [o for o in unique if o.get("time", "") <= args.end + "T23:59:59"]
    print(f"📥 Read {len(unique)} events from {len(files)} files")
    return unique


def percent(o):
    return "--" if o["rate"] is None else f"{o['success']}/{o['n']} ({100 * o['rate']:.1f}\\%)"


def ci(o):
    return "--" if o["rate"] is None else f"[{100 * o['ci95'][0]:.1f}, {100 * o['ci95'][1]:.1f}]"


def latex_table(summary):
    a, an, pg, z = summary["analysis"], summary["survey"], summary["project_feedback"], summary["visit"]
    s = a["duration_sec"]
    duration_text = "--" if s["median"] is None else f"{s['median']:.0f} (P90: {s['p90']:.0f})"
    lines = [
        ("Unique visitors", str(z["unique_visitors"]), "--"),
        ("Analyses started", str(a["started_count"]), "--"),
        ("Analyses completed without error", percent(a["error_free_completion"]), ci(a["error_free_completion"])),
        ("Median analysis time, s", duration_text, "--"),
        ("Survey responses", str(an["responding_sessions"]), "--"),
        ("Users who liked the system", percent(an["liking"]), ci(an["liking"])),
        ("Users reporting no problems", percent(an["problem_free_use"]), ci(an["problem_free_use"])),
        ("Recommendations rated relevant", percent(pg["positive_rate"]), ci(pg["positive_rate"])),
        ("Sessions with $\\geq$1 relevant item in top-10", percent(pg["top10_at_least_one_suitable"]),
         ci(pg["top10_at_least_one_suitable"])),
    ]
    body = "\n".join(f"{m} & {v} & {c} \\\\" for m, v, c in lines)
    first, last = (summary["period"]["first"] or "")[:10], (summary["period"]["last"] or "")[:10]
    return f"""% Generated automatically: web_public/analysis/paper_analysis.py  (data period: {first} -- {last})
\\begin{{table}}[!t]
\\caption{{Field Deployment of FAIR-PASS: Usage and User Feedback}}
\\label{{tab:field_study}}
\\centering
\\begin{{tabular}}{{lcc}}
\\hline
\\textbf{{Metric}} & \\textbf{{Value}} & \\textbf{{95\\% CI (\\%)}} \\\\
\\hline
{body}
\\hline
\\end{{tabular}}
\\\\[2pt]
\\footnotesize{{CIs are Wilson score intervals. Data collected {first} to {last}.}}
\\end{{table}}
"""


def paragraph(summary):
    a, an, pg, z = summary["analysis"], summary["survey"], summary["project_feedback"], summary["visit"]

    def p(o):
        return "n/a" if o["rate"] is None else f"{100 * o['rate']:.1f}% ({o['success']}/{o['n']}; 95% CI {100 * o['ci95'][0]:.1f}-{100 * o['ci95'][1]:.1f}%)"

    first, last = (summary["period"]["first"] or "")[:10], (summary["period"]["last"] or "")[:10]
    return (
        f"FAIR-PASS was deployed as a public web application between {first} and {last}. "
        f"During this period, {z['unique_visitors']} unique visitors accessed the system and "
        f"{a['started_count']} matching analyses were started, of which {p(a['error_free_completion'])} completed "
        f"without error (median processing time {a['duration_sec']['median']} s). "
        f"In the post-session survey ({an['responding_sessions']} responses), {p(an['liking'])} of users "
        f"reported that they liked the system and {p(an['problem_free_use'])} reported using it without "
        f"any problems. Users rated {pg['total_votes']} individual recommendations, {p(pg['positive_rate'])} of "
        f"which were marked as suitable; in {p(pg['top10_at_least_one_suitable'])} of the rating sessions "
        f"at least one top-10 recommendation was marked as suitable.\n\n"
        f"[TODO: Survey participation was voluntary and anonymous; responses may be subject to self-selection "
        f"bias. Add ethics/consent statement as required by the venue.]\n"
    )


def write_csv(events, path):
    fields = sorted({k for o in events for k in o})
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for o in sorted(events, key=lambda x: x.get("time", "")):
            w.writerow({k: (json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v)
                        for k, v in o.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo")
    source.add_argument("--folder")
    source.add_argument("--file", nargs="+")
    ap.add_argument("--start", help="YYYY-MM-DD (from this date on)")
    ap.add_argument("--end", help="YYYY-MM-DD (this date included)")
    ap.add_argument("--include-test", action="store_true", help="Also count the trials made with ?test=1")
    ap.add_argument("--output", default=str(HERE / "output"))
    args = ap.parse_args()

    events = load_events(args)
    summary = metrics.summarize(events, include_test_sessions=args.include_test)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(events, output / "events.csv")
    (output / "user_study_table.tex").write_text(latex_table(summary), encoding="utf-8")
    (output / "paper_paragraph_en.txt").write_text(paragraph(summary), encoding="utf-8")

    a, an, pg = summary["analysis"], summary["survey"], summary["project_feedback"]
    print("\n================ SUMMARY ================")
    print(f"Period                : {summary['period']['first']}  →  {summary['period']['last']}")
    print(f"Unique visitors       : {summary['visit']['unique_visitors']}")
    print(f"Analyses (start/done) : {a['started_count']} / {a['completed']}  (error: {a['failed']}, incomplete: {a['incomplete']})")
    print(f"Error-free completion : {percent(a['error_free_completion'])}".replace("\\%", "%"))
    print(f"Median duration (s)   : {a['duration_sec']['median']}")
    print(f"Liked                 : {percent(an['liking'])}".replace("\\%", "%"))
    print(f"Used without problems : {percent(an['problem_free_use'])}".replace("\\%", "%"))
    print(f"Problem types         : {an['problem_types']}")
    print(f"Suitable recommendations : {percent(pg['positive_rate'])}".replace("\\%", "%"))
    print(f"\n📄 Outputs: {output}")


if __name__ == "__main__":
    main()
