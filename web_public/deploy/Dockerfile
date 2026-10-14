# FAIR-PASS web - Hugging Face Spaces (Docker SDK) image
# This file is copied to the Space root directory by prepare_space.py.

FROM python:3.12-slim

# Selenium + Chromium are needed because the Europass CV page is loaded with JavaScript
RUN apt-get update \
 && apt-get install -y --no-install-recommends chromium chromium-driver fonts-liberation ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# HF Spaces runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    CHROMEDRIVER_PATH=/usr/bin/chromedriver \
    CHROME_BIN=/usr/bin/chromium \
    PORT=7860
WORKDIR /home/user/app

# 1) Dependencies (cache layer: not reinstalled when the code changes)
COPY --chown=user web_public/requirements.txt web_public/requirements.txt
RUN pip install --no-cache-dir --user --extra-index-url https://download.pytorch.org/whl/cpu \
    -r web_public/requirements.txt
RUN python -m spacy download en_core_web_lg

# 2) Bake the transformer models into the image (so the first user does not wait for a download)
COPY --chown=user web_public/deploy/download_models.py web_public/deploy/download_models.py
RUN python web_public/deploy/download_models.py

# 3) Application code and data
COPY --chown=user . .

EXPOSE 7860
# Single worker: the analysis state is kept in memory; concurrency comes from threads
CMD ["gunicorn", "web_public.app:app", "--bind", "0.0.0.0:7860", "--workers", "1", "--threads", "8", "--timeout", "0"]
