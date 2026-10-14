#!/usr/bin/env python3
"""
Prepares the FAIR-PASS web application for publishing on Hugging Face Spaces (and uploads it if you want).

Run from the project root directory:

    # 1) Only prepare and check the folder
    python web_public/deploy/prepare_space.py --user HF_USERNAME

    # 2) Prepare + create the Space and the private data repository + upload
    hf auth login                    # (once; a token with "write" permission)
    python web_public/deploy/prepare_space.py --user HF_USERNAME --upload

The upload does the following:
  - creates the Docker Space <user>/<space> (public site)
  - creates the PRIVATE dataset repository <user>/<space>-user-data (collected data)
  - adds the secrets HF_TOKEN, ADMIN_KEY, FLASK_SECRET_KEY and DATA_REPO to the Space
  - uploads the prepared folder to the Space (large files go through LFS automatically)
"""

import argparse
import os
import secrets
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEPLOY = Path(__file__).resolve().parent
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", "output", "*.log")


def copy_file(source: Path, target: Path, required=True):
    if not source.exists():
        if required:
            sys.exit(f"❌ Not found: {source}")
        print(f"   (skipped, missing) {source.relative_to(PROJECT_ROOT)}")
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        shutil.copytree(source, target, ignore=IGNORE, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)
    print(f"   + {target.relative_to(TARGET)}")
    return True


def readme(space_name):
    return f"""---
title: FAIR-PASS
emoji: 🎯
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
short_description: Europass-based fair volunteer-project matching
---

# FAIR-PASS

A diversity-aware (MMR) recommender that ranks European Solidarity Corps volunteering projects for your
Europass CV with **matching (M)**, **sincerity (S)** and **reliability (R)** scores.
Prototype of a TUBITAK 2209-A research project.

Usage data is collected anonymously (visits, analysis duration/success, 👍/👎 votes and a short survey).
The CV link and personal data are not stored.
"""


def prepare(args):
    global TARGET
    TARGET = Path(args.output).resolve()
    if TARGET.exists():
        shutil.rmtree(TARGET)
    TARGET.mkdir(parents=True)
    print(f"📦 Preparing the Space folder: {TARGET}")

    copy_file(PROJECT_ROOT / "src", TARGET / "src")
    copy_file(PROJECT_ROOT / "config", TARGET / "config")
    copy_file(PROJECT_ROOT / "web_public", TARGET / "web_public")

    ext = PROJECT_ROOT / "data" / "external"
    for name in ["skill_db_relax_20.json", "token_dist.json", "last_scrape_metadata.json"]:
        copy_file(ext / name, TARGET / "data" / "external" / name, required=False)
    # skillNer looks for these two files in the working directory (downloads them from GitHub if missing)
    for name in ["skill_db_relax_20.json", "token_dist.json"]:
        source = PROJECT_ROOT / name if (PROJECT_ROOT / name).exists() else ext / name
        copy_file(source, TARGET / name, required=(name == "skill_db_relax_20.json"))

    processed = PROJECT_ROOT / "data" / "processed"
    vectorized_file = processed / "IP1_output_listings_vectorized.pkl"
    skills_file = processed / "IP1_output_listings_with_skills.pkl"
    if not copy_file(vectorized_file, TARGET / "data" / "processed" / vectorized_file.name, required=False):
        copy_file(skills_file, TARGET / "data" / "processed" / skills_file.name)
    copy_file(processed / "IP3_output_listings_S_scored.pkl",
            TARGET / "data" / "processed" / "IP3_output_listings_S_scored.pkl", required=False)

    shutil.copy2(DEPLOY / "Dockerfile", TARGET / "Dockerfile")
    (TARGET / ".dockerignore").write_text("**/__pycache__\n**/*.pyc\n**/.DS_Store\noutputs/\nlogs/\n")
    (TARGET / "README.md").write_text(readme(args.space), encoding="utf-8")

    # Quickly validate the project pickle
    try:
        import pandas as pd
        pkl = next((TARGET / "data" / "processed").glob("IP1_*.pkl"))
        df = pd.read_pickle(pkl)
        print(f"✅ {pkl.name}: {len(df)} projects, columns: "
              f"{'skills ✓' if 'extracted_skills' in df.columns else 'skills ✗'} "
              f"{'vectors ✓' if 'skill_vectors' in df.columns else 'vectors ✗ (computed at first start, slow)'}")
    except Exception as e:
        print(f"⚠️  Could not read the project pickle ({e}). It may not be downloaded from iCloud; open the file in Finder and try again.")

    size = sum(f.stat().st_size for f in TARGET.rglob("*") if f.is_file()) / 1e6
    print(f"📏 Total size: {size:.1f} MB")


def upload(args):
    from huggingface_hub import HfApi, get_token

    token = args.token or os.environ.get("HF_TOKEN") or get_token()
    if not token:
        sys.exit("❌ Hugging Face token not found. Run `hf auth login` first.")
    api = HfApi(token=token)
    user = args.user or api.whoami()["name"]
    space_id = f"{user}/{args.space}"
    data_id = f"{user}/{args.space}-user-data"

    print(f"☁️  Private data repository: {data_id}")
    api.create_repo(data_id, repo_type="dataset", private=True, exist_ok=True)
    print(f"☁️  Space: {space_id}")
    api.create_repo(space_id, repo_type="space", space_sdk="docker", exist_ok=True)

    data_token = args.data_token or os.environ.get("FAIRPASS_DATA_TOKEN")
    if not data_token:
        print("⚠️  --data-token not given; your login token will be added to the Space as a secret.\n"
              "   Safer: create a 'fine-grained' token with write access only to the data repository\n"
              "   and pass it with --data-token.")
        data_token = token

    info_file = PROJECT_ROOT / "outputs" / "space_info.txt"
    admin = args.admin_key
    if not admin and info_file.exists():
        for line in info_file.read_text().splitlines():
            if line.startswith("ADMIN_KEY="):
                admin = line.split("=", 1)[1]
    admin = admin or secrets.token_urlsafe(18)

    api.add_space_secret(space_id, "HF_TOKEN", data_token)
    api.add_space_secret(space_id, "ADMIN_KEY", admin)
    api.add_space_secret(space_id, "FLASK_SECRET_KEY", secrets.token_hex(32))
    api.add_space_variable(space_id, "DATA_REPO", data_id)

    print("⬆️  Uploading files (may take a few minutes the first time)...")
    api.upload_folder(folder_path=str(TARGET), repo_id=space_id, repo_type="space",
                      commit_message="FAIR-PASS web release")

    subdomain = f"{user}-{args.space}".lower().replace("_", "-").replace(".", "-")
    site = f"https://{subdomain}.hf.space"
    info_file.parent.mkdir(parents=True, exist_ok=True)
    info_file.write_text(
        f"SITE={site}\nSPACE=https://huggingface.co/spaces/{space_id}\nDATA_REPO={data_id}\n"
        f"ADMIN_KEY={admin}\n"
        f"STATS={site}/admin/stats?key={admin}\n")
    print("\n🎉 Uploaded! The Docker image is being built on the Space page (first build ~15-25 min).")
    print(f"   Site:            {site}")
    print(f"   Space settings:  https://huggingface.co/spaces/{space_id}")
    print(f"   Your own tests:  {site}/?test=1   (excluded from the stats)")
    print(f"   Stats:           {site}/admin/stats?key={admin}")
    print(f"   This information was saved to: {info_file}  (outputs/ is not tracked by git)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--user", help="Your Hugging Face user name (for the upload; taken from the token if empty)")
    p.add_argument("--space", default="fair-pass", help="Space name (default: fair-pass)")
    p.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "space_build"), help="Prepared folder")
    p.add_argument("--upload", action="store_true", help="Upload to Hugging Face after preparing")
    p.add_argument("--token", help="HF token (default: the one saved by hf auth login)")
    p.add_argument("--data-token", help="Token the Space uses to write to the data repository (recommended: fine-grained)")
    p.add_argument("--admin-key", help="Key of the stats page (generated if empty)")
    a = p.parse_args()
    prepare(a)
    if a.upload:
        upload(a)
    else:
        print("\nReady. Run the same command with --upload to upload.")
