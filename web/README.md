# Web Interface

This folder contains all files related to the web interface.

## File Structure

```
web/
├── app.py              # Flask web application
├── templates/          # HTML templates
│   ├── index.html     # Home page
│   └── results.html   # Results page
├── static/            # Static files (CSS, JS, images)
└── README.md          # This file
```

## Installation

1. Install the required packages (in the project root directory):
```bash
pip install -r requirements.txt
```

or only for Flask:
```bash
pip install flask pyyaml
```

## Usage

1. Start the web application:
```bash
cd web
python app.py
```

or from the project root directory:

```bash
python web/app.py
```

2. Open this address in your browser:
```
http://localhost:5000
```

## Usage Steps

1. **Enter the CV link**: enter your Europass CV sharing link on the home page
   - Example format: `https://europa.eu/europass/eportfolio/api/eprofile/shared-profile/...`

2. **Analyze**: click the "Analyze" button

3. **Follow the progress**:
   - A progress bar and messages are shown during processing
   - When processing completes, you are redirected to the results page automatically

4. **Review the results**:
   - The top 10 projects are listed
   - The M, S, R and Final scores are shown for each project
   - The project details can be reached through the link

## Features

- ✅ Modern and user-friendly interface
- ✅ Real-time progress tracking
- ✅ Detailed score display (M, S, R, Final)
- ✅ Top 10 project list
- ✅ Responsive design (mobile-friendly)
- ✅ Error handling and user feedback

## How Does It Work?

1. The user enters the Europass CV link
2. The system analyzes the CV and extracts the skills
3. Matching is done against all projects
4. The M (matching), S (sincerity) and R (reliability) scores are computed
5. The projects are ranked by the final score
6. The 10 most suitable projects are shown to the user

## Performance Optimizations

The web application includes the following optimizations (without sacrificing quality):

1. **Parallel project scraping**:
   - Parallel scraping with 5 threads
   - ~5x faster project collection
   
2. **Smart project caching**:
   - Existing projects are loaded from the cache
   - Only new projects are scraped
   - Example: if there are 100 projects and 5 new ones are added, only those 5 are scraped
   
3. **Optimized S score computation**:
   - The S score is computed only for the top 50 projects (the best matches)
   - ~10-20x faster, quality preserved
   
4. **Model preloading**: models are loaded and cached at application start
5. **Using existing IP1 outputs**: if the `IP1_output_listings_with_skills.pkl` file exists, project scraping is skipped

**For details**: see `PERFORMANCE_OPTIMIZATIONS.md`.

### First run (slow)
- Project data is collected (~5-10 minutes)
- Models are loaded
- The CV is analyzed
- The scores are computed

### Subsequent runs (fast)
- Only the CV is analyzed (~30 seconds - 2 minutes)
- Cached project data is used
- Models are already loaded

### To run faster

1. **Prepare the IP1 outputs**: run the full pipeline once first:
   ```bash
   python src/pipeline.py
   ```
   This creates the `IP1_output_listings_with_skills.pkl` file.

2. **Scoring only**: the web application now only analyzes the CV and runs much faster.

## Notes

- The first analysis may take long (because of model loading)
- The CV link must be a valid Europass sharing link
- Do not close the browser window during the analysis
- Since project data is cached, clear the cache when you want to update the project list

## Troubleshooting

**Error: "CV URL is required"**
- Make sure you entered a valid CV link

**Error: "Session not found"**
- Refresh the page and try again

**The analysis takes very long**
- This is normal, especially on the first run when the models are loaded
- Follow the progress bar
