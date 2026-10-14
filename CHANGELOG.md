# Changelog

All notable changes are documented in this file.

## [2.0.0] - 2024-12-09

### ✨ New Features

#### Z-Score Normalization
- All scores (M, S, R) are normalized with the Z-Score
- The M score no longer suppresses the other scores
- Small differences become more visible
- Scores with a standard deviation of 0 are detected automatically

#### S Score Improvements
- Cliché detection and penalty added
- Specific detail bonus (dates, budget, number of participants)
- Sentiment analysis (subjectivity, robotic language detection)
- Wide score range (0-100) with sigmoid normalization
- S scores now range from 0.48 to 96.68 (previously 58.8-59.9)

#### R Score Improvements
- A separate R score is computed for each project
- Bonus/penalty from the organization information (project count, verification status)
- Missing data penalty (missing address, website)
- R scores now range from 0 to 69.67 (different for each project)

#### CSV Output Improvements
- Raw scores are saved (`*_Raw` columns)
- Normalized scores are saved (`*_Normalized` columns)
- 14 columns in total: basic information + raw + normalized + final scores

### 🔧 Changes

- `final_ranking.py`: Z-Score normalization added
- `sincerity_score.py`: NLP analysis extended
- `reliability_score.py`: project-specific R score computation
- `run_scoring_only.py`: saving of raw and normalized scores
- `pipeline.py`: updated CSV output

### 📚 Documentation

- `README.md`: new features added
- `RUNNING_INSTRUCTIONS.md`: Z-Score normalization and new CSV columns
- `DEVELOPMENT_NOTES.md`: all improvements documented
- `S_SCORE_IMPROVEMENTS.md`: details of the S score improvements
- `R_SCORE_IMPROVEMENTS.md`: details of the R score improvements
- `Z_SCORE_NORMALIZATION.md`: details of the Z-Score normalization

### 🐛 Fixes

- M score dominance problem fixed
- Narrow S score range problem fixed
- Constant R score problem fixed
- Error check added for scores with a standard deviation of 0

## [1.0.0] - 2024-12-01

### ✨ First Release

- Modular Python package structure
- YAML configuration file
- Data cleaning module
- Professional logging system
- Comprehensive documentation
- All IP steps (IP1-IP5)
