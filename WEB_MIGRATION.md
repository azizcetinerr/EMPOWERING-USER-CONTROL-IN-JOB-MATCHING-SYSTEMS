# Web Deployment Guide

There are several options for publishing this project on the web. The easiest and free options:

## 🚀 Option 1: Render (recommended - easiest)

**Advantages:**
- ✅ Free tier available
- ✅ Automatic deployment from GitHub
- ✅ Very easy setup
- ✅ Automatic SSL certificate

**Steps:**

1. **Upload to GitHub:**
```bash
cd /path/to/EMPOWERING-USER-CONTROL-IN-JOB-MATCHING-SYSTEMS
git init
git add .
git commit -m "Initial commit"
# Create a new repository on GitHub, then:
git remote add origin https://github.com/USER_NAME/REPO_NAME.git
git push -u origin main
```

2. **Go to Render.com:**
   - Go to https://render.com
   - Select "New +" → "Web Service"
   - Connect your GitHub repository
   - Settings:
     - **Name**: `tubitak-project` (any name you like)
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn web.app:app --bind 0.0.0.0:$PORT`
     - **Root Directory**: `/` (or leave empty)

3. **Add environment variables** (in the Render dashboard):
   ```
   PORT=5001
   PYTHONUNBUFFERED=1
   ```

4. **Deploy!** Render will deploy automatically.

**Note:** on the Render free tier, the application goes to sleep if it is not used for 15 minutes. The first request may take 30-60 seconds.

---

## 🚂 Option 2: Railway (very easy)

**Advantages:**
- ✅ Free tier available ($5 credit/month)
- ✅ Automatic deployment from GitHub
- ✅ Very easy setup
- ✅ Does not go to sleep (faster)

**Steps:**

1. **Upload to GitHub** (as above)

2. **Go to Railway.app:**
   - Go to https://railway.app
   - "New Project" → "Deploy from GitHub repo"
   - Select your repository

3. **In Settings:**
   - **Start Command**: `gunicorn web.app:app --bind 0.0.0.0:$PORT`
   - **Root Directory**: `/` (or empty)

4. **Deploy!**

---

## ✈️ Option 3: Fly.io (modern, fast)

**Advantages:**
- ✅ Free tier available
- ✅ Global CDN
- ✅ Very fast
- ✅ Easy management with the CLI

**Steps:**

1. **Install the Fly.io CLI:**
```bash
curl -L https://fly.io/install.sh | sh
```

2. **Login:**
```bash
fly auth login
```

3. **Initialize the project:**
```bash
cd /path/to/EMPOWERING-USER-CONTROL-IN-JOB-MATCHING-SYSTEMS
fly launch
```

4. **Deploy:**
```bash
fly deploy
```

---

## 🔧 Production Preparation

### 1. Add Gunicorn (to requirements.txt):

```bash
echo "gunicorn" >> requirements.txt
```

### 2. Create a Procfile (for Render/Railway):

Create a `Procfile` file:
```
web: gunicorn web.app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 300
```

### 3. Create runtime.txt (for the Python version):

Create a `runtime.txt` file:
```
python-3.11.0
```

### 4. Create .env.example (optional):

Create a `.env.example` file:
```
PORT=5001
FLASK_ENV=production
```

---

## ⚠️ Important Notes

### 1. **Model files:**
- The models will be downloaded on the first run (from Hugging Face)
- This may take long (5-10 minutes)
- After the first deployment the models will be cached

### 2. **Memory usage:**
- The models are kept in RAM
- Free tiers may have a memory limit
- Use `workers=1` if needed

### 3. **File system:**
- The file system is ephemeral on Render/Railway
- Checkpoints and caches may be lost
- Use external storage (S3, etc.) for important data

### 4. **Timeout:**
- The first analysis may take long (10-15 minutes)
- Increase the timeout: `--timeout 600` (10 minutes)

### 5. **Session management:**
- Currently kept in memory
- Redis or a database should be used in production
- Works for now but is lost on restart

---

## 📝 Quick Start (Render)

1. **Add gunicorn to requirements.txt:**
```bash
echo "gunicorn" >> requirements.txt
```

2. **Create the Procfile:**
```bash
echo "web: gunicorn web.app:app --bind 0.0.0.0:\$PORT --workers 2 --timeout 600" > Procfile
```

3. **Push to GitHub:**
```bash
git add .
git commit -m "Production ready"
git push
```

4. **Deploy on Render** (steps above)

---

## 🐳 Deploying with Docker (advanced)

A `Dockerfile` can be created if you want to use Docker. With Docker:
- It runs on any platform
- More controlled environment
- You can test it locally

---

## 💡 Recommendation

**Use Render to get started:**
- Easiest
- Free
- GitHub integration
- Automatic SSL

**Move to Railway later:**
- Faster (does not go to sleep)
- More resources
- Better performance

---

## 🆘 Troubleshooting

**"ModuleNotFoundError: No module named 'flask'"**
- Check whether flask is in `requirements.txt`
- Check the build logs

**"Application error"**
- Check the logs (Render/Railway dashboard)
- May be a timeout problem (increase the timeout)

**"Memory limit exceeded"**
- Reduce the number of workers: `--workers 1`
- Clear the model cache

**"Timeout"**
- The first analysis takes long; increase the timeout: `--timeout 600`

---

## 📞 Help

If there is a problem:
1. Check the build logs
2. Check the runtime logs
3. Test locally: `gunicorn web.app:app --bind 0.0.0.0:5001`
