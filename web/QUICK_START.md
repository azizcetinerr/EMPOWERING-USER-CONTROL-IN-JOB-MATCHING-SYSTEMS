# Quick Start Guide

## Status Check

When the application starts, you will see these messages:

### ✅ Fast mode (recommended)
```
✅ IP1 outputs found - fast mode active
   (Project scraping will be skipped, only the CV analysis will run)
```

### ⚠️ Slow mode (first run)
```
⚠️  IP1 outputs not found - slow mode
   (The first analysis may take long; project data will be collected)
```

## Creating the IP1 Outputs for Fast Mode

### Method 1: With the pipeline (recommended)

1. Go to the project root directory in the terminal:
```bash
cd /path/to/EMPOWERING-USER-CONTROL-IN-JOB-MATCHING-SYSTEMS
```

2. Run the pipeline (once):
```bash
python src/pipeline.py
```

This command:
- Collects the project data (~5-10 minutes)
- Extracts the skills
- Computes the vectors
- Creates the `IP1_output_listings_with_skills.pkl` file

3. Restart the web application:
```bash
python web/app.py
```

You will now see the "fast mode active" message!

### Method 2: From the web interface (first analysis)

1. Open the web interface: `http://localhost:5001`
2. Enter your CV link
3. The first analysis may take long (~10-15 minutes) but the IP1 outputs will be created
4. Subsequent analyses will be much faster (~2-3 minutes)

## Performance Comparison

| Mode | First analysis | Subsequent analyses |
|-------|------------|-------------------|
| **Slow mode** | ~10-15 minutes | ~2-3 minutes |
| **Fast mode** | ~2-3 minutes | ~1-2 minutes |

## Tips

1. **If you are using it for the first time**: run the pipeline once, then use the web interface
2. **If you only want a CV analysis**: continue from the web interface (the first analysis takes long)
3. **The project list was updated**: the cache is updated automatically and only new projects are scraped

## Troubleshooting

**I see the "IP1 outputs not found" message:**
- Normal; this message appears on the first run
- The first analysis may take long but subsequent analyses will be fast

**"Loading models" takes very long:**
- The model is being downloaded for the first time (depends on the internet connection)
- It will be much faster on subsequent starts

**The analysis takes very long:**
- First analysis: normal (~10-15 minutes)
- Subsequent analyses: much faster (~2-3 minutes)
