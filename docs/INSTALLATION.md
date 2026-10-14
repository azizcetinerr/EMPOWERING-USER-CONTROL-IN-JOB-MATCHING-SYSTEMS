# Installation and Usage Guide

## 📁 Folder Structure

```
EMPOWERING-USER-CONTROL-IN-JOB-MATCHING-SYSTEMS/
├── config/                    # Configuration files
│   └── config.yaml           # All system settings
│
├── src/                       # Source code
│   ├── data_collection/       # Data collection modules
│   │   ├── youth_portal_scraper.py      # Fetches the project links
│   │   ├── project_detail_scraper.py    # Fetches the project details
│   │   └── cv_parser.py                 # Parses the CV
│   │
│   ├── skill_extraction/      # Skill extraction modules
│   │   ├── skill_extractor.py           # Extracts skills with SkillNER
│   │   └── vectorizer.py               # Vectorization
│   │
│   ├── scoring/               # Scoring modules
│   │   ├── matching_score.py            # Computes the M-Score
│   │   ├── sincerity_score.py           # Computes the S-Score
│   │   ├── reliability_score.py         # Computes the R-Score
│   │   └── final_ranking.py            # Final ranking
│   │
│   ├── utils/                # Utility functions
│   │   ├── config_loader.py             # Config loader
│   │   ├── logger.py                    # Logging system
│   │   └── data_cleaner.py              # Data cleaning
│   │
│   └── pipeline.py           # Main pipeline (runs all steps)
│
├── outputs/                   # Output files
│   ├── raw/                   # Raw data
│   ├── processed/             # Processed data
│   └── results/               # Results (CSV, charts)
│
├── logs/                      # Log files
│   └── project.log           # System logs
│
├── requirements.txt           # Python dependencies
├── README.md                  # General documentation
├── DEVELOPMENT_NOTES.md      # Improvement notes
└── INSTALLATION.md                 # This file
```

## 🚀 Quick Start

### 1. Install the dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Edit the configuration

Open `config/config.yaml` and check:
- The CV URL (`data_sources.europass_cv.url`)
- The motivation letter (`data_sources.motivation_letter.text`)
- Other settings

### 3. Run the pipeline

```bash
python src/pipeline.py
```

or in Python:

```python
from src.pipeline import ProjectPipeline

pipeline = ProjectPipeline()
df_final = pipeline.run_full_pipeline()
```

## 📝 Step-by-Step Usage

### Data collection only (IP1)

```python
from src.pipeline import ProjectPipeline

pipeline = ProjectPipeline()
df_projects, candidate_vectors, cv_text, candidate_skills = pipeline.run_ip1_data_collection()
```

### Scoring only (IP2-IP5)

```python
# Run IP1 first, then:
df_projects = pipeline.run_ip2_matching_score(df_projects, candidate_vectors)
df_projects = pipeline.run_ip3_sincerity_score(df_projects)
r_score = pipeline.run_ip4_reliability_score(candidate_skills, cv_text)
df_final = pipeline.run_ip5_final_ranking(df_projects, r_score)
```

## 🔧 Module-Level Usage

Each module can also be used on its own:

```python
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.data_collection.youth_portal_scraper import YouthPortalScraper

# Config and logger
config = load_config()
logger = setup_logger()

# Fetch links only
scraper = YouthPortalScraper(config, logger)
links = scraper.get_project_links()
```

## 📊 Output Files

The system produces the following files:

- `outputs/processed/IP1_output_*.pkl`: Processed data
- `outputs/results/IP5_final_fair_ranking.csv`: Final ranking
- `logs/project.log`: System logs

## ⚠️ Important Notes

1. **First run**: all models will be downloaded (a few GB)
2. **Internet connection**: required for web scraping
3. **Chrome/Chromium**: required for Selenium
4. **Processing time**: about 30-40 minutes for 952 projects

## 🐛 Troubleshooting

### Model not found error
```bash
python -m spacy download en_core_web_lg
```

### Import error
```bash
# Make sure you run it from the project root directory
python src/pipeline.py
```

### Changing the log level
In `config/config.yaml`:
```yaml
logging:
  level: "DEBUG"  # For more detailed logs
```

## 📚 More Information

- [README.md](../README.md) - General project documentation
- [DEVELOPMENT_NOTES.md](DEVELOPMENT_NOTES.md) - Improvement details
- [config/config.yaml](../config/config.yaml) - All settings
