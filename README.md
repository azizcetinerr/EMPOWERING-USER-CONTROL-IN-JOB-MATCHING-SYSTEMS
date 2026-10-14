# EMPOWERING USER CONTROL IN JOB MATCHING SYSTEMS

This project is an AI-supported system that performs fair, transparent and skill-based matching between volunteers and projects.

## 🚀 Features

- **Data collection**: Youth Portal and Europass CV integration.
- **Skill extraction**: skill analysis of texts with NLP (SkillNER, spaCy).
- **Scoring system**:
  - **M-Score (Matching)**: skill- and vector-based fit.
  - **S-Score (Sincerity)**: sincerity and motivation analysis.
  - **R-Score (Reliability)**: organization and project reliability analysis.
- **Fair ranking**: diversity-aware ranking (MMR) and multi-objective optimization.

## 📁 Project Structure

```
EMPOWERING-USER-CONTROL-IN-JOB-MATCHING-SYSTEMS/
├── config/                 # System settings (config.yaml)
├── src/                    # Source code
│   ├── data_collection/    # Data collection modules
│   ├── skill_extraction/   # Skill analysis modules
│   ├── scoring/            # Scoring algorithms
│   └── pipeline.py         # Main entry point
├── data/                   # Data files (raw and processed)
├── outputs/                # Outputs and reports
├── notebooks/              # Analysis notebooks
├── PAPER/                  # Paper reproducibility scripts
├── web/                    # (Optional) web interface
├── web_public/             # Public release of the web interface with user data collection
├── skill_db_relax_20.json  # SkillNER skill database used for the reported run
└── token_dist.json         # SkillNER token distribution used for the reported run
```

SkillNER reads `skill_db_relax_20.json` and `token_dist.json` from the working directory and downloads them
if they are missing, so run the scripts from the repository root to use the versions stored here.

## 🛠 Installation

See [docs/INSTALLATION.md](docs/INSTALLATION.md) for the installation steps.

Quick installation:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

## 💻 Usage

To run the system:

```bash
python src/pipeline.py
```

For detailed usage instructions, see [docs/INSTALLATION.md](docs/INSTALLATION.md) and [docs/RUNNING_INSTRUCTIONS.md](docs/RUNNING_INSTRUCTIONS.md).

## 📊 Outputs

After a run, the system writes the following to the `outputs/` folder:
- Ranked project lists
- Scoring details
- Performance and fairness charts

## 📄 Paper data and verification script

This section lists the files cited in the Data and Code Availability Statement of the FAIR-PASS paper (fairpass_v2).

| File | Content |
|---|---|
| `outputs/results/IP5_S_recomputed.csv` | Scored listing file for the 952 listings (raw and standardized M/S/R scores, `Final_Score`, `Fair_Score`, `Human_Score`, `Context_Score`) |
| `outputs/results/IP5_final_fair_ranking.csv` | Score file of the earlier version; input to `recompute_sincerity.py` |
| `PAPER/recompute_sincerity.py` | Recomputes the sincerity score with the classes in `src/scoring` and writes `IP5_S_recomputed.csv` (downloads models) |
| `PAPER/verify_numbers.py` | Recomputes the derived values reported in the Results section of the paper |
| `config/config.yaml` | Contains the motivation letter used in the reported run |

Column glossary for the score files:

| Column | Meaning |
|---|---|
| `link` | URL of the listing on the European Youth Portal |
| `full_description` | Full listing description |
| `extracted_skills` | Skills extracted from the description |
| `M_Score_Raw`, `S_Score_Raw`, `R_Score_Raw` | Raw matching, sincerity and reliability scores |
| `M_Score`, `S_Score`, `R_Score` | Standardized scores (z-score mapped to 0-100) |
| `M_Score_Normalized`, `S_Score_Normalized`, `R_Score_Normalized` | Standardized scores as written by the earlier pipeline run (`recompute_sincerity.py` does not update `S_Score_Normalized`) |
| `Final_Score` | Weighted sum of the standardized scores (F) |
| `Fair_Score` | Final score after the keyword bonus (adjusted score A) |
| `Human_Score`, `Context_Score` | Components of the sincerity score (S_human, S_context) |
| `S_Score_Raw_old` | Raw sincerity score of the earlier version |

Run from the repository root:

```bash
pip install pandas scipy
python PAPER/verify_numbers.py outputs/results/IP5_S_recomputed.csv
```

To regenerate `IP5_S_recomputed.csv`:

```bash
pip install -r requirements.txt
python PAPER/recompute_sincerity.py
```

The curriculum vitae is not shared because it contains personal data.

## 📜 License

The code is released under the [MIT License](LICENSE).
