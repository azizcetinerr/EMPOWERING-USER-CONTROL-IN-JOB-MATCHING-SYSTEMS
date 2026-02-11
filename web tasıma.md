# Web Deployment Kılavuzu

Bu projeyi web'de yayınlamak için birkaç seçenek var. En kolay ve ücretsiz seçenekler:

## 🚀 Seçenek 1: Render (Önerilen - En Kolay)

**Avantajlar:**
- ✅ Ücretsiz tier var
- ✅ GitHub ile otomatik deploy
- ✅ Çok kolay kurulum
- ✅ SSL sertifikası otomatik

**Adımlar:**

1. **GitHub'a yükle:**
```bash
cd /Users/acetiner/Desktop/Tubitak_Proje/Tubitak_Proje_Yeni
git init
git add .
git commit -m "Initial commit"
# GitHub'da yeni repo oluştur, sonra:
git remote add origin https://github.com/KULLANICI_ADI/REPO_ADI.git
git push -u origin main
```

2. **Render.com'a git:**
   - https://render.com adresine git
   - "New +" → "Web Service" seç
   - GitHub repo'nu bağla
   - Ayarlar:
     - **Name**: `tubitak-proje` (istediğin isim)
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn web.app:app --bind 0.0.0.0:$PORT`
     - **Root Directory**: `/` (veya boş bırak)

3. **Environment Variables ekle** (Render dashboard'da):
   ```
   PORT=5001
   PYTHONUNBUFFERED=1
   ```

4. **Deploy!** Render otomatik olarak deploy edecek.

**Not:** Render ücretsiz tier'da uygulama 15 dakika kullanılmazsa uykuya geçer. İlk istekte 30-60 saniye sürebilir.

---

## 🚂 Seçenek 2: Railway (Çok Kolay)

**Avantajlar:**
- ✅ Ücretsiz tier var ($5 kredi/ay)
- ✅ GitHub ile otomatik deploy
- ✅ Çok kolay kurulum
- ✅ Uykuya geçmez (daha hızlı)

**Adımlar:**

1. **GitHub'a yükle** (yukarıdaki gibi)

2. **Railway.app'e git:**
   - https://railway.app adresine git
   - "New Project" → "Deploy from GitHub repo"
   - Repo'nu seç

3. **Settings'te:**
   - **Start Command**: `gunicorn web.app:app --bind 0.0.0.0:$PORT`
   - **Root Directory**: `/` (veya boş)

4. **Deploy!**

---

## ✈️ Seçenek 3: Fly.io (Modern, Hızlı)

**Avantajlar:**
- ✅ Ücretsiz tier var
- ✅ Global CDN
- ✅ Çok hızlı
- ✅ CLI ile kolay yönetim

**Adımlar:**

1. **Fly.io CLI yükle:**
```bash
curl -L https://fly.io/install.sh | sh
```

2. **Login:**
```bash
fly auth login
```

3. **Projeyi initialize et:**
```bash
cd /Users/acetiner/Desktop/Tubitak_Proje/Tubitak_Proje_Yeni
fly launch
```

4. **Deploy:**
```bash
fly deploy
```

---

## 🔧 Production Hazırlığı

### 1. Gunicorn ekle (requirements.txt'ye):

```bash
echo "gunicorn" >> requirements.txt
```

### 2. Procfile oluştur (Render/Railway için):

`Procfile` dosyası oluştur:
```
web: gunicorn web.app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 300
```

### 3. Runtime.txt oluştur (Python versiyonu için):

`runtime.txt` dosyası oluştur:
```
python-3.11.0
```

### 4. .env.example oluştur (opsiyonel):

`.env.example` dosyası oluştur:
```
PORT=5001
FLASK_ENV=production
```

---

## ⚠️ Önemli Notlar

### 1. **Model Dosyaları:**
- Modeller ilk çalıştırmada indirilecek (HuggingFace'den)
- Bu uzun sürebilir (5-10 dakika)
- İlk deploy'dan sonra modeller cache'lenecek

### 2. **Memory Kullanımı:**
- Modeller RAM'de tutuluyor
- Ücretsiz tier'larda memory limiti olabilir
- Gerekirse `workers=1` kullan

### 3. **File System:**
- Render/Railway'de dosya sistemi geçici
- Checkpoint'ler ve cache'ler kaybolabilir
- Önemli veriler için external storage (S3, etc.) kullan

### 4. **Timeout:**
- İlk analiz uzun sürebilir (10-15 dakika)
- Timeout'u artır: `--timeout 600` (10 dakika)

### 5. **Session Management:**
- Şu an memory'de tutuluyor
- Production'da Redis veya database kullanılmalı
- Şimdilik çalışır ama restart'ta kaybolur

---

## 📝 Hızlı Başlangıç (Render)

1. **requirements.txt'ye gunicorn ekle:**
```bash
echo "gunicorn" >> requirements.txt
```

2. **Procfile oluştur:**
```bash
echo "web: gunicorn web.app:app --bind 0.0.0.0:\$PORT --workers 2 --timeout 600" > Procfile
```

3. **GitHub'a push et:**
```bash
git add .
git commit -m "Production ready"
git push
```

4. **Render'da deploy et** (yukarıdaki adımlar)

---

## 🐳 Docker ile Deploy (İleri Seviye)

Docker kullanmak istersen `Dockerfile` oluşturabilirim. Docker ile:
- Herhangi bir platformda çalışır
- Daha kontrollü environment
- Local'de test edebilirsin

---

## 💡 Öneri

**Başlangıç için Render kullan:**
- En kolay
- Ücretsiz
- GitHub entegrasyonu
- Otomatik SSL

**Daha sonra Railway'a geç:**
- Daha hızlı (uykuya geçmez)
- Daha fazla kaynak
- Daha iyi performans

---

## 🆘 Sorun Giderme

**"ModuleNotFoundError: No module named 'flask'"**
- `requirements.txt`'de flask var mı kontrol et
- Build log'ları kontrol et

**"Application error"**
- Logs'u kontrol et (Render/Railway dashboard)
- Timeout sorunu olabilir (timeout'u artır)

**"Memory limit exceeded"**
- Workers sayısını azalt: `--workers 1`
- Model cache'ini temizle

**"Timeout"**
- İlk analiz uzun sürüyor, timeout'u artır: `--timeout 600`

---

## 📞 Yardım

Sorun olursa:
1. Build logs'u kontrol et
2. Runtime logs'u kontrol et
3. Local'de test et: `gunicorn web.app:app --bind 0.0.0.0:5001`
