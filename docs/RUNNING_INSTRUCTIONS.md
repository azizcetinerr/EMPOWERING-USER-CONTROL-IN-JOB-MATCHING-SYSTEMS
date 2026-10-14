# Running Instructions

## ✅ Improvements Made

1. **The S score is computed for all projects**
   - `sincerity_scoring.calculate_all: true` in `config.yaml` (default)
   - No project is skipped
   - Advanced NLP analysis: cliché detection, specific detail bonus, sentiment analysis
   - Wide score range (0-100) with sigmoid normalization

2. **The R score is computed separately for each project**
   - For each project, the share of that project's requirements verified in the CV
   - Bonus/penalty from the organization information (project count, verification status)
   - Missing data penalty

3. **Z-Score normalization**
   - All scores (M, S, R) are normalized with the Z-Score
   - The M score no longer suppresses the other scores
   - Small differences become more visible
   - Scores with a standard deviation of 0 are detected automatically

4. **All scores are stored as data**
   - 14 columns in the CSV output:
     - Raw scores: `M_Score_Raw`, `S_Score_Raw`, `R_Score_Raw`
     - Normalized scores: `M_Score_Normalized`, `S_Score_Normalized`, `R_Score_Normalized`
     - Scores in use: `M_Score`, `S_Score`, `R_Score` (normalized)
     - Final scores: `Final_Score`, `Fair_Score`
   - No score is missing

5. **Statistics and checks added**
   - Score statistics are shown at every step
   - Missing scores are checked
   - Score contributions are analyzed

## 🚀 Running Steps

### 1. Install the dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### 2. Check the config file

Open `config/config.yaml` and check:
- `sincerity_scoring.calculate_all: true` (S score for all projects)
- Is the CV URL correct?
- Is there a motivation letter?

### 3. Run the pipeline

```bash
python src/pipeline.py
```

If the `IP1_output_*.pkl` files already exist, the pipeline uses them as a cache and only collects data for new projects.

## 📊 Expected Outputs

### CSV file (`outputs/results/IP5_final_fair_ranking.csv`)

Each row will have these 14 columns:

**Basic information:**
- `link`: Project URL
- `full_description`: Project description
- `extracted_skills`: Extracted skills

**Raw scores (original values):**
- `M_Score_Raw`: Raw matching score (0-100)
- `S_Score_Raw`: Raw sincerity score (0-100)
- `R_Score_Raw`: Raw reliability score (0-100)

**Normalized scores (with Z-Score):**
- `M_Score_Normalized`: Normalized M score
- `S_Score_Normalized`: Normalized S score
- `R_Score_Normalized`: Normalized R score

**Scores in use (normalized):**
- `M_Score`: M score used in the final computation (normalized)
- `S_Score`: S score used in the final computation (normalized)
- `R_Score`: R score used in the final computation (normalized, different for each project)

**Final scores:**
- `Final_Score`: Weighted mean (M×0.5 + S×0.3 + R×0.2)
- `Fair_Score`: Final score + fairness bonus

### Log file (`logs/project.log`)

All operations are logged:
- Start/end of each step
- Progress reports
- Score statistics
- Error messages

## ⏱️ Time Estimates

- **IP1 (data collection)**: ~30-40 minutes (952 projects)
- **IP2 (M score)**: ~1 minute
- **IP3 (S score)**: ~10-15 minutes (for all projects)
- **IP4 (R score)**: ~2-3 minutes
- **IP5 (final ranking)**: ~1 minute

**Total**: ~45-60 minutes

## 🔍 Checklist

After running, check:

- [ ] Are all projects in the CSV file? (952 projects)
- [ ] Does every project have an M_Score?
- [ ] Does every project have an S_Score? (no NaN?)
- [ ] Does every project have an R_Score?
- [ ] Does every project have a Fair_Score?
- [ ] Do the scores differ? (are they not all the same?)

## 🐛 Troubleshooting

```bash
pip install pyyaml
```

### "ModuleNotFoundError: No module named 'pandas'"
```bash
pip install -r requirements.txt
```

### "S score missing" warning
- Make sure `calculate_all: true` is set in `config.yaml`
- Wait for IP3 to complete

### It runs very slowly
- The S score computation takes long (embeddings for all projects)
- Be patient, or set `calculate_all: false` to compute only for the first N projects

## 📝 Example Output

```
✅ ALL SCORING STEPS COMPLETED!

📊 SUMMARY:
   Total number of projects: 952
   M score (normalized): 0.00 - 53.15 (mean: 51.83)
   S score (normalized): 20.36 - 78.26 (mean: 50.00)
   R score (normalized): 0.00 - 96.25 (mean: 50.74) (different for each project)
   Fair score: 6.13 - 73.01 (mean: 55.84)

📈 SCORE CONTRIBUTIONS:
   M contribution: 25.92 (50.0% weight)
   S contribution: 15.00 (30.0% weight)
   R contribution: 10.15 (20.0% weight)

🏆 TOP 5 PROJECTS:
   1. [Fair: 73.01] [M: 52.9, S: 74.3, R: 96.2] 47139
   2. [Fair: 71.18] [M: 53.0, S: 74.0, R: 87.5] 48151
   3. [Fair: 69.22] [M: 53.1, S: 76.7, R: 73.3] 34309
   ...
```

## ✅ Success Criteria

The pipeline is considered successful if:
1. ✅ All 952 projects were processed
2. ✅ No project is missing a score
3. ✅ The scores differ (they are not all the same)
4. ✅ The CSV file has 14 columns (raw + normalized + final)
5. ✅ Z-Score normalization was applied (the M score no longer suppresses the others)
6. ✅ A separate R score was computed for each project
7. ✅ No errors in the log file
