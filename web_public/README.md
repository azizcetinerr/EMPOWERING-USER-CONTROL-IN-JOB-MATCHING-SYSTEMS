# web_public — FAIR-PASS public release + user data collection

A copy of the Flask interface in the `web/` folder that is ready to be published on the internet. `web/` was not changed.
In addition, the use of the site and user feedback are recorded **anonymously**; this data is turned into a table
and a paragraph for the paper with `analysis/paper_analysis.py`.

## What is recorded?

| Event | When | Use in the paper |
|---|---|---|
| `visit` | When the home page is opened | Number of unique visitors |
| `analysis_started` / `analysis_completed` / `analysis_error` | Automatically | **Error-free completion rate**, analysis duration (median/P90), stage of the error |
| `project_feedback` | When the user gives 👍/👎 to a project | Share of recommendations rated suitable, breakdown by rank, "≥1 suitable project in the top 10" |
| `survey` | Short form at the bottom of the results page (and the error screen) | **Liking rate**, **problem-free use rate**, problem types, comments |

Not recorded: the CV link/content, name, e-mail, IP. The visitor ID is a random code in a browser cookie.
E-mail addresses and links in comments are masked automatically. The user accepts an information notice before the analysis.

**Your own trials:** open the site once with `.../?test=1`; everything from that browser is marked as `test`
and excluded from the stats (`?test=0` turns this off).

## Running locally

```bash
cd /path/to/EMPOWERING-USER-CONTROL-IN-JOB-MATCHING-SYSTEMS
python web_public/app.py          # http://localhost:5001  (data is written to outputs/user_data/)
```

## Publishing (Hugging Face Spaces — free, 16 GB RAM)

The free tiers of Render/Railway (512 MB) cannot hold torch + 3 transformer models + spaCy `lg`;
the free CPU hardware of HF Spaces can. Because the Space disk is not persistent, the collected data
is backed up automatically to a **private HF Dataset repository** every 5 minutes.

1. Create an account at https://huggingface.co and create a token with **Write** permission under *Settings → Access Tokens*.
2. In the terminal (in the project root directory, with your ML environment active):
   ```bash
   pip install -U huggingface_hub
   hf auth login
   python web_public/deploy/prepare_space.py --user HF_USERNAME            # check first
   python web_public/deploy/prepare_space.py --user HF_USERNAME --upload    # then upload
   ```
3. The Docker image is built on the Space page (~15–25 min the first time). The site address, the admin key and the stats link
   are written to `outputs/space_info.txt`.
4. After startup the models are loaded into memory in the background (~2–5 min); analyses arriving meanwhile wait with "The server is getting ready".

> Safer setup: create a *fine-grained* token on HF with write access only to the `…-user-data` repository and
> pass it with `--data-token`. Otherwise your login token is added to the Space as a secret.

### Admin endpoints (`ADMIN_KEY` required)

- `/admin/stats?key=…` — current summary metrics (JSON). `&test=1` includes the test sessions.
- `/admin/download?key=…` — raw events (JSONL).
- `/cache-status` — queue, number of projects in memory, etc.

## Analysis for the paper

```bash
python web_public/analysis/paper_analysis.py --repo HF_USERNAME/fair-pass-user-data \
       --start 2026-10-20 --end 2026-11-20
```

`summary.json`, `events.csv`, the IEEEtran-compatible `user_study_table.tex` (can be added with `\input{}`)
and an English paragraph draft with the numbers filled in are written to `web_public/analysis/output/`. Rates are given
with 95% Wilson confidence intervals; only the last survey response of each session and the last vote for each project are counted.

## Technical differences from `web/app.py`

- Analyses are queued (one analysis at a time) → no memory/CPU overflow on 2 free vCPUs; the user sees their place in the queue.
- Because the S score depends only on the project text, it is cached across projects (`outputs/s_score_cache.pkl`);
  it is computed only for new projects.
- Project data is read from `data/processed/` (web/app.py looked in the root directory and could not find it); new projects
  added by the pipeline are kept in memory, so later users do not download them again.
- The Youth Portal API is checked at most once every 6 hours.
- The ~50 MB checkpoint file written for each analysis is deleted when the analysis ends.
- In Docker, Selenium uses the system chromium/chromedriver (`docker_compat.py`; `src/` unchanged).
- Only `europa.eu/europass` links are accepted; cache clearing is available only to the admin.

## Files

```
web_public/
├── app.py               # Flask application (derived from web/app.py)
├── event_log.py         # JSONL event log + HF Dataset backup
├── metrics.py           # Metric computation from the events (Wilson CI)
├── docker_compat.py     # Adapting Selenium to the container
├── requirements.txt     # Server dependencies
├── templates/           # index.html (consent box), results.html (survey)
├── static/survey.js     # "Did you like it? / Did you have any problems?" form
├── deploy/              # Dockerfile, model download, prepare_space.py
└── analysis/            # paper_analysis.py
```
