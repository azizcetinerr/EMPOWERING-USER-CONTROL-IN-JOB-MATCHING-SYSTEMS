# Development Notes and Improvements Made

## ✅ Completed Improvements

### 1. Move to a modular structure ✅
- **Before**: all code in Jupyter notebooks, in single cells
- **Now**: modular Python package structure
  - `src/data_collection/`: data collection modules
  - `src/skill_extraction/`: skill extraction modules
  - `src/scoring/`: scoring modules
  - `src/utils/`: utility functions

### 2. Configuration file ✅
- **Before**: hard-coded values (URLs, model names, etc.)
- **Now**: central configuration in YAML format
  - All parameters in `config/config.yaml`
  - Easily changeable settings
  - Support for environment-specific config

### 3. Data cleaning module ✅
- **Before**: duplicate links, noisy skills
- **Now**: comprehensive data cleaning
  - Duplicate removal
  - Skill filtering (stopwords, min length)
  - Empty description check
  - Data quality reporting

### 4. Logging system ✅
- **Before**: print statements, no logs
- **Now**: professional logging
  - Level-based logging (DEBUG, INFO, WARNING, ERROR)
  - File and console output
  - Timestamp and module information
  - Progress tracking

### 5. Code organization ✅
- **Before**: notebooks, repeated code
- **Now**:
  - Reusable classes
  - Type hints
  - Docstrings
  - Clean code principles

### 6. Documentation ✅
- **Before**: no documentation
- **Now**:
  - Comprehensive README.md
  - In-code documentation
  - Usage examples
  - Description of the project structure

### 7. S score improvements ✅
- **Before**: S scores in a very narrow range (58.8-59.9)
- **Now**:
  - Cliché detection and penalty
  - Specific detail bonus (dates, budget, number of participants)
  - Sentiment analysis (subjectivity, robotic language detection)
  - Wide score range (0-100) with sigmoid normalization
  - S scores now range from 0.48 to 96.68

### 8. R score improvements ✅
- **Before**: the same R score for all projects
- **Now**:
  - A separate R score for each project (share of the project's requirements verified in the CV)
  - Bonus/penalty from the organization information (project count, verification status)
  - Missing data penalty (missing address, website)
  - R scores now range from 0 to 69.67 (different for each project)

### 9. Z-Score normalization ✅
- **Before**: the M score was very dominant (mean 97.56), the R and S scores had no effect
- **Now**:
  - All scores (M, S, R) are normalized with the Z-Score
  - The M score no longer suppresses the other scores
  - Small differences become more visible
  - Scores with a standard deviation of 0 are detected automatically
  - Balanced score contributions: M=25.92, S=15.00, R=10.15

### 10. CSV output improvements ✅
- **Before**: only the final scores were saved
- **Now**:
  - Raw scores are saved (`*_Raw` columns)
  - Normalized scores are saved (`*_Normalized` columns)
  - 14 columns in total: basic information + raw + normalized + final scores

## 📊 Comparison

| Feature | Before | Now |
|---------|--------|------|
| Code organization | Notebooks | Modular package |
| Configuration | Hard-coded | YAML config |
| Logging | Print | Professional logging |
| Data cleaning | Manual | Automatic module |
| Documentation | None | Comprehensive |
| S score range | 58.8-59.9 (narrow) | 0.48-96.68 (wide) |
| R score | Same for all projects | Different for each project |
| Score normalization | None | Z-Score normalization |
| Score balance | M score dominant | Balanced contributions |
| CSV output | 8 columns | 14 columns (raw + normalized) |
| Reusability | Low | High |
| Testability | Hard | Easy |

## 🚀 Usage

### Old method (notebook)
```python
# inside p1.123.ipynb
# All code in a single cell...
```

### New method (modular)
```python
from src.pipeline import ProjectPipeline

pipeline = ProjectPipeline()
df_final = pipeline.run_full_pipeline()
```

or step by step:
```python
from src.data_collection.youth_portal_scraper import YouthPortalScraper
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

config = load_config()
logger = setup_logger()
scraper = YouthPortalScraper(config, logger)
links = scraper.get_project_links()
```

## 📝 Next Steps (Suggestions)

### High priority
1. ✅ Modular structure - **COMPLETED**
2. ✅ Config file - **COMPLETED**
3. ✅ Data cleaning - **COMPLETED**
4. ✅ Logging - **COMPLETED**
5. ⏳ **Performance optimization** (parallel processing, batch processing)
6. ⏳ Writing **unit tests**
7. ⏳ **Web interface** (Flask/FastAPI)

### Medium priority
- Model fine-tuning
- More advanced fairness metrics
- API endpoints
- Docker containerization

### Low priority
- Distributed computing
- Real-time processing
- Multi-language support

## 🔧 Technical Details

### New folder structure
```
src/
├── data_collection/
│   ├── youth_portal_scraper.py
│   ├── project_detail_scraper.py
│   └── cv_parser.py
├── skill_extraction/
│   ├── skill_extractor.py
│   └── vectorizer.py
├── scoring/
│   ├── matching_score.py
│   ├── sincerity_score.py
│   ├── reliability_score.py
│   └── final_ranking.py
├── utils/
│   ├── config_loader.py
│   ├── logger.py
│   └── data_cleaner.py
└── pipeline.py
```

### Config structure
```yaml
data_sources:
  youth_portal: {...}
  europass_cv: {...}
  motivation_letter: {...}

models:
  skill_extraction: {...}
  embedding: {...}
  ai_detection: {...}
  nli: {...}

scoring_weights:
  matching_score: 0.5
  sincerity_score: 0.3
  reliability_score: 0.2
```

## 📈 Performance Improvements (Future)

1. **Parallel processing**: fetch project details in parallel with `multiprocessing`
2. **Batch processing**: batch processing for vectorization (already present but can be optimized)
3. **Caching**: cache the vectors to avoid recomputation
4. **GPU optimization**: optimize the use of MPS/CUDA

## 🐛 Known Issues and Solutions

1. **AI detection model**: gives an AI score of 99.95% (the model may produce a false positive)
   - **Status**: still present, but less influential in the S score (balanced by the NLP analysis)
   - **Solution**: an alternative model or fine-tuning may be needed

2. **Low R-Score**: previously 58.49 (some skills cannot be verified)
   - **Status**: ✅ SOLVED - a separate R score is now computed for each project
   - **Now**: R scores range from 0 to 69.67, different for each project

3. **Dominant M score**: previously the M score suppressed the other scores
   - **Status**: ✅ SOLVED - balanced contributions with Z-Score normalization

4. **Narrow S score range**: previously 58.8-59.9
   - **Status**: ✅ SOLVED - widened to 0.48-96.68 with NLP analysis and sigmoid normalization

5. **Performance**: ~28 minutes for 952 projects (IP1)
   - **Status**: still present
   - **Solution**: parallel processing, batch optimization

## 💡 Tips

1. **Changing the config**: always edit `config/config.yaml`
2. **Log tracking**: check `logs/project.log`
3. **Debugging**: set the logger level to `DEBUG`
4. **Module testing**: each module can be tested on its own

## 📚 Resources

- [README.md](../README.md) - General usage guide
- [config/config.yaml](../config/config.yaml) - All settings
- [src/pipeline.py](../src/pipeline.py) - Main pipeline
