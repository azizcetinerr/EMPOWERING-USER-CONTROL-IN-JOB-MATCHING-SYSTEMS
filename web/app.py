#!/usr/bin/env python3
"""
İŞ EŞLEŞTİRME SİSTEMLERİNDE KULLANICI KONTROLÜNÜN GÜÇLENDİRİLMESİ - Web Arayüzü
CV linki ile proje eşleştirme sistemi
"""

import os
# Tokenizers parallelism uyarısını düzelt
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from pathlib import Path
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import threading
import pandas as pd
import numpy as np

# Proje kök dizinini path'e ekle (web klasörünün bir üst dizini)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.pipeline import ProjectPipeline
from src.data_collection.cv_parser import EuropassCVParser
from src.skill_extraction.skill_extractor import SkillExtractorModule
from src.skill_extraction.vectorizer import SkillVectorizer
from src.scoring.matching_score import MatchingScoreCalculator
from src.scoring.sincerity_score import SincerityScoreCalculator
from src.scoring.reliability_score import ReliabilityScoreCalculator
from src.scoring.final_ranking import FinalRankingCalculator

# Flask app'i template klasörü ile birlikte başlat
app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'tubitak-proje-secret-key-2024'  # Production'da değiştirin!

# Global değişkenler
processing_status = {}
processing_results = {}

# Önbellek: Proje verileri ve modeller
_cached_projects = None
_cached_models = {}
_cached_api_data = None  # API'den gelen ham veriler
_api_cache_file = None   # Cache dosyası yolu
_config = None
_logger = None

def get_api_cache_file():
    """API cache dosyası yolunu döndür"""
    global _api_cache_file
    if _api_cache_file is None:
        _api_cache_file = project_root / "outputs" / "api_cache.json"
        _api_cache_file.parent.mkdir(parents=True, exist_ok=True)
    return _api_cache_file

def load_api_cache():
    """API cache'ini diskten yükle"""
    import json
    cache_file = get_api_cache_file()
    
    if not cache_file.exists():
        return None
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        return cache_data
    except Exception as e:
        if _logger:
            _logger.warning(f"API cache dosyası okunamadı: {e}")
        return None

def save_api_cache(api_data: dict, project_count: int):
    """API verilerini cache'e kaydet"""
    import json
    from datetime import datetime
    import hashlib
    
    cache_file = get_api_cache_file()
    
    # API verilerinin hash'ini al (değişiklik kontrolü için)
    api_data_str = json.dumps(api_data, sort_keys=True)
    data_hash = hashlib.md5(api_data_str.encode()).hexdigest()
    
    cache_data = {
        'timestamp': datetime.now().isoformat(),
        'project_count': project_count,
        'data_hash': data_hash,
        'api_data': api_data
    }
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
        if _logger:
            _logger.info(f"✅ API cache kaydedildi: {project_count} proje ({cache_file})")
        return True
    except Exception as e:
        if _logger:
            _logger.error(f"API cache kaydedilemedi: {e}")
        return False

def check_api_for_updates():
    """API'deki veriler ile cache'deki verileri karşılaştır, fark varsa güncelle"""
    global _cached_api_data, _config, _logger
    
    if _config is None:
        _config = load_config()
    if _logger is None:
        _logger = setup_logger()
    
    # Cache'i yükle
    cached_data = load_api_cache()
    
    # API'den veri çek
    try:
        from src.data_collection.youth_portal_scraper import YouthPortalScraper
        scraper = YouthPortalScraper(_config, _logger)
        api_data = scraper.get_project_data_from_api()
        
        if api_data is None:
            _logger.warning("⚠️  API'den veri alınamadı, cache kullanılacak")
            if cached_data:
                _cached_api_data = cached_data.get('api_data')
            return False
        
        project_count = api_data.get('hits', {}).get('total', {}).get('value', 0)
        
        # Cache yoksa direkt kaydet
        if cached_data is None:
            _logger.info("📥 İlk kez API cache oluşturuluyor...")
            save_api_cache(api_data, project_count)
            _cached_api_data = api_data
            return True
        
        # Hash karşılaştırması
        import json
        import hashlib
        api_data_str = json.dumps(api_data, sort_keys=True)
        current_hash = hashlib.md5(api_data_str.encode()).hexdigest()
        cached_hash = cached_data.get('data_hash')
        
        if current_hash != cached_hash:
            _logger.info(f"🔄 API verilerinde değişiklik tespit edildi (cache: {cached_data.get('project_count')}, API: {project_count}), güncelleniyor...")
            save_api_cache(api_data, project_count)
            _cached_api_data = api_data
            # Cache'teki projeleri de invalidate et (yeniden işlenecek)
            global _cached_projects
            _cached_projects = None
            return True
        else:
            _logger.info(f"✅ API verilerinde değişiklik yok, cache kullanılıyor ({project_count} proje)")
            _cached_api_data = cached_data.get('api_data')
            return False
            
    except Exception as e:
        _logger.error(f"API güncelleme kontrolü hatası: {e}", exc_info=True)
        # Hata durumunda cache'i kullan
        if cached_data:
            _cached_api_data = cached_data.get('api_data')
        return False

def get_cached_projects():
    """Önbelleğe alınmış proje verilerini döndür veya yükle (CSV ve pickle kontrol eder)"""
    global _cached_projects, _cached_api_data, _config, _logger
    
    # İlk yüklemede API cache'i kontrol et
    if _cached_api_data is None:
        check_api_for_updates()
    
    if _cached_projects is not None:
        return _cached_projects
    
    if _config is None:
        _config = load_config()
    if _logger is None:
        _logger = setup_logger()
    
    from src.utils.config_loader import get_output_path
    
    # Önce CSV dosyasını kontrol et
    csv_projects_file = get_output_path(_config, 'project_details')
    ip1_projects_file = project_root / _config['output']['files']['projects_with_skills']
    
    # Pickle dosyası varsa onu kullan (daha eksiksiz)
    if ip1_projects_file.exists():
        try:
            _cached_projects = pd.read_pickle(ip1_projects_file)
            
            # Becerilerin varlığını kontrol et ve logla
            if 'cikarilan_beceriler' in _cached_projects.columns:
                skills_count = _cached_projects['cikarilan_beceriler'].notna().sum()
                valid_skills = _cached_projects['cikarilan_beceriler'].apply(
                    lambda x: isinstance(x, list) and len(x) > 0 if isinstance(x, list) else False
                ).sum()
                _logger.info(f"✅ {len(_cached_projects)} proje pickle'dan yüklendi")
                _logger.info(f"   📊 {valid_skills}/{len(_cached_projects)} projede geçerli beceri var (önbellekten)")
            else:
                _logger.warning(f"⚠️  Pickle'da beceri sütunu yok!")
            
            # Vektörler yoksa ekle
            if "beceri_vektorleri" not in _cached_projects.columns:
                _logger.info("   Vektörler yok, hesaplanıyor...")
                vectorizer = _get_model('vectorizer')
                _cached_projects = vectorizer.vectorize_dataframe(_cached_projects)
            else:
                _logger.info("   ✅ Vektörler mevcut (önbellekten)")
            
            return _cached_projects
        except Exception as e:
            _logger.warning(f"Pickle dosyası okunamadı: {e}")
    
    # CSV dosyası varsa onu kullan
    if csv_projects_file.exists():
        try:
            _cached_projects = pd.read_csv(csv_projects_file, encoding='utf-8-sig')
            _logger.info(f"✅ {len(_cached_projects)} proje CSV'den yüklendi")
            # Becerileri çıkar ve vektörleri hesapla
            if 'cikarilan_beceriler' not in _cached_projects.columns:
                skill_extractor = _get_model('skill_extractor')
                _cached_projects = skill_extractor.extract_skills_from_dataframe(_cached_projects)
            if "beceri_vektorleri" not in _cached_projects.columns:
                vectorizer = _get_model('vectorizer')
                _cached_projects = vectorizer.vectorize_dataframe(_cached_projects)
            return _cached_projects
        except Exception as e:
            _logger.warning(f"CSV dosyası okunamadı: {e}")
    
    return None

def get_cached_models():
    """Modelleri lazy loading ile yükle (sadece gerektiğinde)"""
    global _cached_models, _config, _logger
    
    if _cached_models:
        return _cached_models
    
    if _config is None:
        _config = load_config()
    if _logger is None:
        _logger = setup_logger()
    
    # Lazy loading: Modelleri sadece gerektiğinde yükle
    # Başlangıçta None olarak ayarla, ilk kullanımda yüklenecek
    _cached_models = {
        'vectorizer': None,
        'matching_calc': None,
        'sincerity_calc': None,
        'reliability_calc': None,
        'ranking_calc': None,
        'skill_extractor': None,
        '_config': _config,
        '_logger': _logger
    }
    
    return _cached_models

def _get_model(model_key: str):
    """Lazy loading ile modeli yükle"""
    global _cached_models
    
    if _cached_models.get(model_key) is not None:
        return _cached_models[model_key]
    
    _config = _cached_models.get('_config')
    _logger = _cached_models.get('_logger')
    
    if _config is None or _logger is None:
        _config = load_config()
        _logger = setup_logger()
        _cached_models['_config'] = _config
        _cached_models['_logger'] = _logger
    
    # Modeli yükle
    if model_key == 'vectorizer':
        _logger.info("Vectorizer modeli yükleniyor...")
        _cached_models[model_key] = SkillVectorizer(_config, _logger)
    elif model_key == 'matching_calc':
        _logger.info("Matching calculator yükleniyor...")
        _cached_models[model_key] = MatchingScoreCalculator(_config, _logger)
    elif model_key == 'sincerity_calc':
        _logger.info("Sincerity calculator yükleniyor (bu biraz uzun sürebilir)...")
        _cached_models[model_key] = SincerityScoreCalculator(_config, _logger)
    elif model_key == 'reliability_calc':
        _logger.info("Reliability calculator yükleniyor...")
        _cached_models[model_key] = ReliabilityScoreCalculator(_config, _logger)
    elif model_key == 'ranking_calc':
        _logger.info("Ranking calculator yükleniyor...")
        _cached_models[model_key] = FinalRankingCalculator(_config, _logger)
    elif model_key == 'skill_extractor':
        _logger.info("Skill extractor yükleniyor...")
        _cached_models[model_key] = SkillExtractorModule(_config, _logger)
    
    return _cached_models[model_key]

def _clean_description(text, max_length=200):
    """Açıklama metnini temizler ve kısaltır"""
    if not text or not isinstance(text, str):
        return ''
    
    import re
    
    # Önce temel temizleme yap
    cleaned = _clean_language_codes(text)
    
    # Ek temizlik: Başlık ve açıklama birbirine karışmışsa, ilk cümleyi bul
    # Eğer açıklama çok kısaysa veya başlık gibi görünüyorsa, ilk gerçek cümleyi bul
    
    # İlk cümleyi bul (nokta, soru işareti, ünlem ile biten)
    sentences = re.split(r'[.!?]+\s+', cleaned)
    if len(sentences) > 1:
        # İlk anlamlı cümleyi bul (en az 20 karakter)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and not sentence.lower().startswith(('apply', 'share', 'sign', 'join', 'activity', 'deadline')):
                cleaned = sentence
                break
    
    # Maksimum uzunluğa göre kısalt, kelime sınırında kes
    if len(cleaned) > max_length:
        truncated = cleaned[:max_length - 3]
        last_space = truncated.rfind(' ')
        if last_space > max_length * 0.7:  # En az %70'ini al
            return truncated[:last_space] + '...'
        return truncated + '...'
    
    return cleaned

def _clean_language_codes(text):
    """Dil kodlarını ve gereksiz karakterleri metinden temizler, birleşik kelimeleri ayırır, sayfa metinlerini kaldırır, düzgün cümleler üretir"""
    if not text or not isinstance(text, str):
        return text if text else ''
    
    import re
    
    # 0. Sayfa metinlerini temizle (Apply, Share this page, Sign in/Join gibi)
    page_ui_texts = [
        r'\bApply\b', r'\bShare this page\b', r'\bSign in\s*/?\s*Join\b', 
        r'\bSign in\b', r'\bJoin the Corps\b', r'\bJoin\b.*Corps',
        r'\bLooking for participants\b', r'\bActivity dates\b', r'\bActivity location\b',
        r'\bActivity type\b', r'\bActivity topics\b', r'\bDeadline for applications\b',
        r'\bNo application deadline\b', r'\bPlease explain\b.*\bbox below\b',
        r'\bThe feedback collection\b.*\banonymous\b', r'\bPlease do not submit\b',
        r'\bDiscover more\b.*\boneuropa\.eu\b', r'\bIn order to check\b.*\bSign In\b'
    ]
    for pattern in page_ui_texts:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # 0.5. Bozuk karakterleri temizle (Ų, ų gibi yalnız kalan karakterler)
    text = re.sub(r'[ŲųŪūŮůŰűŲųŴŵŶŷŸŹźŻżŽžſ]+', '', text)
    # İzole kalan tek karakterli Unicode karakterleri (normal harfler değil)
    text = re.sub(r'\b[^\w\s]\s+', '', text)  # Kelime başında izole özel karakterler
    text = re.sub(r'\s+[^\w\s]\b', '', text)  # Kelime sonunda izole özel karakterler
    
    # 1. Dil kodlarını ve onlara bağlı karakterleri temizle
    # BGбългарскиCSčeština gibi pattern'ler (2-3 harfli kod + non-ASCII karakterler)
    text = re.sub(r'[A-Z]{2,3}[а-яА-ЯёЁa-záčďéěíňóřšťúůýžàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿığşüöçąćęłńóśźżăîşț]+', '', text)
    
    # DEL, EL gibi kısaltmalar + Greek characters (ελληνικά)
    text = re.sub(r'[A-Z]{2,3}[ελληνικάα-ωΑ-Ω]+', '', text)
    
    # Tekrarlayan büyük harfli kodlar (BGCSDADE...) - bunlar genelde dil seçici
    text = re.sub(r'\b[A-Z]{2,10}\b(?=[а-яА-ЯёЁa-záčďéěíňóřšťúůýžελληνικάα-ωΑ-Ωăîşț])', '', text)
    
    # 1.5. İzole kalan büyük harfli kodları temizle (VZW, NGO, ESC gibi - ama bunlar bağlamda anlamlı olabilir, dikkatli ol)
    # Sadece kelime sınırlarında ve boşluklarla çevrili olanları temizle
    text = re.sub(r'\s+[A-Z]{2,5}\s+', ' ', text)  # Ortada izole kodlar
    
    # 2. Başta/sonda kalan gereksiz karakterler ve kodlar
    text = re.sub(r'^[A-Z]{2,10}\s+', '', text)  # Başta büyük harfli kodlar
    text = re.sub(r'\s+[A-Z]{2,10}$', '', text)  # Sonda büyük harfli kodlar
    text = re.sub(r'^[^\w]+\s*', '', text)  # Başta özel karakterler
    text = re.sub(r'\s*[^\w]+$', '', text)  # Sonda özel karakterler
    
    # 3. Birleşmiş kelimeleri ayır (camelCase, PascalCase, ve özel durumlar)
    # Önce birleşmiş büyük harfli kelimeleri ayır (örn: "BordersGiosef" -> "Borders Giosef")
    text = re.sub(r'([A-Z][a-z]+)([A-Z][a-z]+)', r'\1 \2', text)
    # Küçük harften sonra büyük harf geliyorsa boşluk ekle (Descriptionvzw -> Description vzw)
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # Tek harfli kelimelerden sonra (A, I gibi) boşluk ekle (ama "AI", "IT" gibi kısaltmaları koru)
    text = re.sub(r'\b([A-Z])([A-Z][a-z]{2,})', r'\1 \2', text)
    
    # 3.5. Özel birleşik durumlar (Descriptionvzw, ApplyDescription gibi)
    # Description, Apply gibi kelimelerden sonra gelen birleşik metinleri ayır
    text = re.sub(r'\b(Description|Apply|Share|Sign|Join|Activity|Deadline|Looking)([A-Z][a-z]+)', r'\1 \2', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(Description|Apply|Share|Sign|Join|Activity|Deadline|Looking)([a-z]{3,})', r'\1 \2', text, flags=re.IGNORECASE)
    
    # 4. Fazla boşlukları ve gereksiz karakterleri temizle
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'^\s+|\s+$', '', text)  # Başta/sonda boşluk
    
    # 5. Başta kalan tek harfli izole harfleri ve özel karakterleri temizle (E, D, L, Ų gibi)
    text = re.sub(r'^[A-ZŲŪŮŰŴŶŸŹŻŽſ]\s+', '', text)
    
    # 6. Cümle başını büyük harfe çevir (eğer küçük harfle başlıyorsa)
    if text and len(text) > 0:
        if text[0].islower():
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        # Bozuk karakterle başlıyorsa temizle
        elif ord(text[0]) > 127 and text[0] not in 'ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþ':
            # Normal Unicode karakter değilse, ikinci karakterden başla
            if len(text) > 1:
                text = text[1:].strip()
                if text and text[0].islower():
                    text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    return text.strip()

def _get_project_title(row):
    """Proje başlığını alır (proje_adi yoksa açıklamadan çıkarır)"""
    import re
    
    # Önce proje_adi'yi kontrol et (eğer varsa temizle)
    title = row.get('proje_adi')
    if title and str(title).strip() and str(title) != 'Proje Başlığı Bulunamadı':
        cleaned_title = _clean_language_codes(str(title).strip())
        # Eğer temizlenmiş başlık anlamlıysa (en az 5 karakter) kullan
        if len(cleaned_title) > 5:
            # Maksimum 150 karakter, cümle sonunu kesme
            if len(cleaned_title) > 150:
                # Son kelimeyi kesme, en yakın boşluğa kadar al
                truncated = cleaned_title[:147]
                last_space = truncated.rfind(' ')
                if last_space > 100:  # En az 100 karakter al
                    return truncated[:last_space] + '...'
                return cleaned_title[:150]
            return cleaned_title
    
    # Açıklamadan ilk anlamlı satırı al
    description = str(row.get('tam_aciklama', ''))
    if description:
        # Önce tam açıklamayı temizle
        clean_desc = _clean_language_codes(description)
        
        # İlk satırı al
        first_line = clean_desc.split('\n')[0].strip()
        
        # İlk anlamlı cümleyi bul (noktalama veya 50+ karakter ile biten)
        # Eğer temizlenmiş satır anlamlıysa (20'den fazla karakter) kullan
        if len(first_line) > 20:
            # Maksimum 150 karakter, cümle sonunu kesme
            if len(first_line) > 150:
                truncated = first_line[:147]
                last_space = truncated.rfind(' ')
                if last_space > 100:
                    return truncated[:last_space] + '...'
                return first_line[:150]
            return first_line
        
        # Yoksa temizlenmiş açıklamanın ilk anlamlı kısmını al
        if len(clean_desc) > 20:
            if len(clean_desc) > 150:
                truncated = clean_desc[:147]
                last_space = truncated.rfind(' ')
                if last_space > 100:
                    return truncated[:last_space] + '...'
                return clean_desc[:150]
            return clean_desc
    
    return 'Proje Başlığı Yok'

def run_pipeline_with_cv_url(cv_url: str, session_id: str):
    """CV URL ile pipeline'ı çalıştırır (optimize edilmiş versiyon)"""
    global _config, _logger
    
    # Session ID'nin processing_status'ta olduğundan emin ol
    if session_id not in processing_status:
        processing_status[session_id] = {
            'status': 'processing',
            'message': 'Başlatılıyor...',
            'progress': 5
        }
    
    try:
        # Durumu güncelle
        processing_status[session_id].update({
            'status': 'processing',
            'message': 'Başlatılıyor...',
            'progress': 5
        })
        
        # Config ve logger'ı yükle
        if _config is None:
            _config = load_config()
        if _logger is None:
            _logger = setup_logger()
        
        # ✅ RESUME: Checkpoint kontrolü - kaldığı yerden devam et
        checkpoint_dir = project_root / "outputs" / "checkpoints"
        checkpoint_file = checkpoint_dir / f"checkpoint_{session_id}.pkl"
        
        skip_to_ip3 = False
        if checkpoint_file.exists():
            try:
                import pickle
                _logger.info(f"🔄 Checkpoint bulundu, kaldığı yerden devam ediliyor: {checkpoint_file}")
                
                with open(checkpoint_file, 'rb') as f:
                    checkpoint_data = pickle.load(f)
                
                df_projects = checkpoint_data['df_projects']
                candidate_vectors = checkpoint_data['candidate_vectors']
                cv_text = checkpoint_data['cv_text']
                candidate_skills = checkpoint_data['candidate_skills']
                checkpoint_stage = checkpoint_data.get('checkpoint_stage', 'IP2_COMPLETED')
                
                _logger.info(f"✅ Checkpoint yüklendi: Stage={checkpoint_stage}, Proje sayısı={len(df_projects)}")
                _logger.info(f"   IP1 ve IP2 atlanıyor, {checkpoint_stage} aşamasından devam ediliyor...")
                
                # IP1 ve IP2 atlandı, IP3'ten devam et
                processing_status[session_id]['message'] = 'Checkpoint yüklendi, IP3\'ten devam ediliyor...'
                processing_status[session_id]['progress'] = 70
                
                # Checkpoint'i sil (bir kez kullanıldı)
                checkpoint_file.unlink()
                _logger.info(f"🗑️  Checkpoint kullanıldı ve silindi: {checkpoint_file}")
                
                # IP3'e geç (aşağıdaki IP3 koduna atla)
                skip_to_ip3 = True
                
            except Exception as e:
                _logger.warning(f"⚠️  Checkpoint yüklenemedi: {e}, normal pipeline ile devam ediliyor")
                skip_to_ip3 = False
        
        if not skip_to_ip3:
            # Normal pipeline - IP1'den başla
            # Modelleri önbelleğe al
            processing_status[session_id]['message'] = 'Modeller yükleniyor...'
            processing_status[session_id]['progress'] = 10
            models = get_cached_models()
            
            # Önbelleğe alınmış proje verilerini kontrol et ve API'yi kontrol et
            processing_status[session_id]['message'] = 'Proje verileri ve API kontrol ediliyor...'
            processing_status[session_id]['progress'] = 15
            # API kontrolü (arka planda, cache kullanılır)
            check_api_for_updates()
            df_projects = get_cached_projects()
            
            if df_projects is None:
                # Proje verileri yok, tam pipeline çalıştır (önbellek destekli)
                processing_status[session_id]['message'] = 'Proje verileri kontrol ediliyor (mevcut olanlar önbellekten yüklenecek)...'
                processing_status[session_id]['progress'] = 20
                pipeline = ProjectPipeline()
                pipeline.config['data_sources']['europass_cv']['url'] = cv_url
                # Cached API data'yı pipeline'a geçir
                if _cached_api_data:
                    pipeline.config['_cached_api_data'] = _cached_api_data
                df_projects, candidate_vectors, cv_text, candidate_skills = pipeline.run_ip1_data_collection()
                
                # Önbelleği güncelle
                global _cached_projects
                _cached_projects = df_projects
            else:
                # Proje verileri var, pipeline'a önceden yüklenmiş projeleri ver
                # Eğer beceriler ve vektörler varsa, pipeline sadece CV parse edecek
                _logger.info(f"🔍 get_cached_projects() döndü: {len(df_projects)} proje")
                if 'cikarilan_beceriler' in df_projects.columns:
                    skills_count = df_projects['cikarilan_beceriler'].notna().sum()
                    _logger.info(f"   📊 {skills_count}/{len(df_projects)} projede beceri sütunu var")
                if 'beceri_vektorleri' in df_projects.columns:
                    vectors_count = df_projects['beceri_vektorleri'].notna().sum()
                    _logger.info(f"   📊 {vectors_count}/{len(df_projects)} projede vektör sütunu var")
                
                processing_status[session_id]['message'] = 'Proje verileri önbellekten yüklendi, CV analiz ediliyor...'
                processing_status[session_id]['progress'] = 20
                pipeline = ProjectPipeline()
                pipeline.config['data_sources']['europass_cv']['url'] = cv_url
                # Cached API data'yı pipeline'a geçir
                if _cached_api_data:
                    pipeline.config['_cached_api_data'] = _cached_api_data
                _logger.info(f"🚀 Pipeline'a preloaded_projects geçiriliyor: {len(df_projects)} proje")
                df_projects, candidate_vectors, cv_text, candidate_skills = pipeline.run_ip1_data_collection(preloaded_projects=df_projects)
            # Not: Artık pipeline içinde preloaded_projects kontrolü yapılıyor
            # Eğer projeler önbellekten yüklendiyse ve beceriler/vektörler varsa,
            # pipeline sadece CV parse edecek ve beceri çıkarma/vektörizasyon atlanacak
            
            # IP1 tamamlandı, projeler dosyaya kaydedildi
            # Şimdi IP2'ye geçiliyor
            _logger.info("✅ IP1 tamamlandı! Projeler dosyaya kaydedildi. IP2'ye geçiliyor...")
            
            # IP2 ve IP4: M ve R skorlarını paralel hesapla
            processing_status[session_id]['message'] = 'IP2 & IP4: M ve R skorları paralel hesaplanıyor (2 iş eş zamanlı)...'
            processing_status[session_id]['progress'] = 50
            
            from concurrent.futures import ThreadPoolExecutor
            
            matching_calc = _get_model('matching_calc')
            reliability_calc = _get_model('reliability_calc')
            
            def calculate_m_score():
                """M skorunu hesapla"""
                return matching_calc.calculate_all_scores(df_projects.copy(), candidate_vectors)
            
            def calculate_r_score():
                """R skorunu hesapla (M skorundan bağımsız)"""
                # CV paragraflarını önbelleğe al (her seferinde parse etmeyelim)
                cv_paragraphs = [p for p in cv_text.split('\n') if len(p) > 20]
                # Çok uzun paragrafları kısalt (performans için)
                cv_paragraphs = [p[:500] for p in cv_paragraphs]  # Max 500 karakter
                return reliability_calc.calculate_project_reliability_scores(
                    df_projects.copy(), 
                    cv_paragraphs,
                    skill_column="cikarilan_beceriler"
                )
            
            # Paralel hesaplama (2 thread)
            with ThreadPoolExecutor(max_workers=2) as executor:
                m_future = executor.submit(calculate_m_score)
                r_future = executor.submit(calculate_r_score)
                
                # Sonuçları bekle ve birleştir
                df_projects = m_future.result()
                df_projects_r = r_future.result()
                
                # R skorunu ekle
                if 'R_Skoru' in df_projects_r.columns:
                    df_projects['R_Skoru'] = df_projects_r['R_Skoru']
            
            # ✅ CHECKPOINT: IP2 sonrası durumu kaydet (kaldığı yerden devam için)
            try:
                checkpoint_dir = project_root / "outputs" / "checkpoints"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                # Session ID ile checkpoint dosyası oluştur
                checkpoint_file = checkpoint_dir / f"checkpoint_{session_id}.pkl"
                
                import pickle
                checkpoint_data = {
                    'df_projects': df_projects,
                    'candidate_vectors': candidate_vectors,
                    'cv_text': cv_text,
                    'candidate_skills': candidate_skills,
                    'checkpoint_stage': 'IP2_COMPLETED',  # Hangi aşamada olduğumuzu belirt
                    'timestamp': pd.Timestamp.now().isoformat()
                }
                
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                
                _logger.info(f"💾 Checkpoint kaydedildi: {checkpoint_file} (IP2 tamamlandı)")
            except Exception as e:
                _logger.warning(f"⚠️  Checkpoint kaydedilemedi: {e}")
                # Checkpoint kaydedilemese bile devam et
        
        # IP2 tamamlandı (normal pipeline'dan veya checkpoint'ten), IP3'e geç
        processing_status[session_id]['progress'] = 70
        processing_status[session_id]['message'] = 'IP2 & IP4 tamamlandı! IP3 başlatılıyor...'
        
        # IP3: S Skoru (M skorundan sonra, sıralama için gerekli)
        processing_status[session_id]['message'] = 'IP3: Samimiyet skoru hesaplanıyor (tüm projeler için)...'
        processing_status[session_id]['progress'] = 75
        # Tüm projeler için S skoru hesapla (kullanıcı talebi: S skoru hesaplanmamış proje olmasın)
        calculate_all = True  # Tüm projeler için hesapla
        sincerity_calc = _get_model('sincerity_calc')  # Bu model yükleme sırasında uzun sürebilir
        df_projects = sincerity_calc.calculate_all_scores(df_projects, calculate_all=calculate_all)
        
        # IP5: Final Ranking
        processing_status[session_id]['message'] = 'IP5: Final sıralama hesaplanıyor...'
        processing_status[session_id]['progress'] = 85
        ranking_calc = _get_model('ranking_calc')
        df_final = ranking_calc.calculate_ranking(df_projects)
        
        processing_status[session_id]['message'] = 'Sonuçlar hazırlanıyor...'
        processing_status[session_id]['progress'] = 95
        
        # Sonuçları hazırla (hata durumunda bile status'u set et)
        try:
            results = {
                'total_projects': len(df_final),
                'top_projects': []
            }
            
            # Top 10 projeyi hazırla
            top_10 = df_final.head(10)
            for idx, row in top_10.iterrows():
                try:
                    # Project ID'yi link'ten çıkar (son kısım)
                    link = row.get('link', '')
                    project_id = link.split('/')[-1].split('_')[0] if link else str(idx)
                    
                    project_data = {
                        'rank': idx + 1,
                        'project_id': project_id,
                        'link': link,
                        'title': _get_project_title(row),
                        'description': _clean_description(str(row.get('tam_aciklama', ''))),
                        'm_score': round(float(row.get('M_Skoru', 0)), 2),
                        's_score': round(float(row.get('S_Skoru')), 2) if pd.notna(row.get('S_Skoru')) else None,
                        'r_score': round(float(row.get('R_Skoru', 0)), 2),
                        'final_score': round(float(row.get('Adil_Skor', 0)), 2),
                        'skills': row.get('cikarilan_beceriler', [])[:5] if isinstance(row.get('cikarilan_beceriler', []), list) else []
                    }
                    results['top_projects'].append(project_data)
                except Exception as e:
                    _logger.warning(f"Proje {idx} sonuç hazırlama hatası: {e}")
                    continue
            
            # Tüm alakalı projeleri hazırla (Adil_Skor > 40 olanlar veya top 50)
            all_relevant = df_final[df_final['Adil_Skor'] > 40].head(100)  # İlk 100 alakalı proje
            results['all_relevant_projects'] = []
            for idx, row in all_relevant.iterrows():
                try:
                    # Project ID'yi link'ten çıkar (son kısım)
                    link = row.get('link', '')
                    project_id = link.split('/')[-1].split('_')[0] if link else str(idx)
                    
                    project_data = {
                        'rank': idx + 1,
                        'project_id': project_id,
                        'link': link,
                        'title': _get_project_title(row),
                        'description': _clean_description(str(row.get('tam_aciklama', ''))),
                        'm_score': round(float(row.get('M_Skoru', 0)), 2),
                        's_score': round(float(row.get('S_Skoru')), 2) if pd.notna(row.get('S_Skoru')) else None,
                        'r_score': round(float(row.get('R_Skoru', 0)), 2),
                        'final_score': round(float(row.get('Adil_Skor', 0)), 2),
                        'skills': row.get('cikarilan_beceriler', [])[:5] if isinstance(row.get('cikarilan_beceriler', []), list) else []
                    }
                    results['all_relevant_projects'].append(project_data)
                except Exception as e:
                    _logger.warning(f"Proje {idx} sonuç hazırlama hatası: {e}")
                    continue
            
            # İstatistikler
            results['stats'] = {
                'm_score': {
                    'min': round(float(df_final['M_Skoru'].min()), 2),
                    'max': round(float(df_final['M_Skoru'].max()), 2),
                    'avg': round(float(df_final['M_Skoru'].mean()), 2)
                },
                's_score': {
                    'min': round(float(df_final['S_Skoru'].min()), 2) if pd.notna(df_final['S_Skoru'].min()) else None,
                    'max': round(float(df_final['S_Skoru'].max()), 2) if pd.notna(df_final['S_Skoru'].max()) else None,
                    'avg': round(float(df_final['S_Skoru'].mean()), 2) if pd.notna(df_final['S_Skoru'].mean()) else None
                },
                'r_score': {
                    'min': round(float(df_final['R_Skoru'].min()), 2),
                    'max': round(float(df_final['R_Skoru'].max()), 2),
                    'avg': round(float(df_final['R_Skoru'].mean()), 2)
                },
                'final_score': {
                    'min': round(float(df_final['Adil_Skor'].min()), 2),
                    'max': round(float(df_final['Adil_Skor'].max()), 2),
                    'avg': round(float(df_final['Adil_Skor'].mean()), 2)
                }
            }
            
            processing_results[session_id] = results
        except Exception as e:
            _logger.error(f"Sonuç hazırlama hatası: {e}", exc_info=True)
            # Sonuçlar hazırlanamadı ama pipeline tamamlandı, basit bir sonuç döndür
            processing_results[session_id] = {
                'total_projects': len(df_final),
                'top_projects': [],
                'stats': {},
                'warning': 'Sonuçlar tam olarak hazırlanamadı, ancak analiz tamamlandı.'
            }
        
        # Status'u MUTLAKA set et (sonuçlar hazırlansa da hazırlanmasa da)
        processing_status[session_id]['status'] = 'completed'
        processing_status[session_id]['message'] = 'Tamamlandı!'
        processing_status[session_id]['progress'] = 100
        _logger.info(f"✅ Pipeline başarıyla tamamlandı (session: {session_id}, proje sayısı: {len(df_final)})")
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        _logger.error(f"❌ Pipeline hatası (session: {session_id}): {error_trace}")
        
        # Session ID'nin processing_status'ta olduğundan emin ol
        if session_id not in processing_status:
            processing_status[session_id] = {
                'status': 'error',
                'message': f'Hata: {str(e)}',
                'progress': 0
            }
        else:
            processing_status[session_id].update({
                'status': 'error',
                'message': f'Hata: {str(e)}',
                'progress': 0
            })
        processing_results[session_id] = {'error': str(e)}
    finally:
        # Her durumda status'un ayarlandığından emin ol
        if session_id in processing_status:
            final_status = processing_status[session_id].get('status', 'unknown')
            _logger.info(f"🔍 Final status kontrolü (session: {session_id}): {final_status}, progress: {processing_status[session_id].get('progress', 0)}")
            
            # Eğer status hala 'processing' veya 'queued' ise ve progress 95+ ise, muhtemelen tamamlandı ama status set edilmemiş
            if final_status in ['processing', 'queued'] and processing_status[session_id].get('progress', 0) >= 95:
                _logger.warning(f"⚠️  Status 'completed' olarak işaretlenmemiş ama progress %95+. Düzeltiliyor...")
                processing_status[session_id]['status'] = 'completed'
                processing_status[session_id]['message'] = 'Tamamlandı! (otomatik düzeltme)'
                processing_status[session_id]['progress'] = 100


@app.route('/')
def index():
    """Ana sayfa"""
    return render_template('index.html')


@app.route('/submit', methods=['POST'])
def submit_cv():
    """CV URL'ini al ve işleme başlat"""
    data = request.get_json()
    cv_url = data.get('cv_url', '').strip()
    
    if not cv_url:
        return jsonify({'error': 'CV URL gerekli'}), 400
    
    # Session ID oluştur
    import uuid
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    
    # İşleme durumunu başlat (thread başlamadan önce)
    processing_status[session_id] = {
        'status': 'queued',
        'message': 'Kuyruğa alındı...',
        'progress': 0
    }
    
    # Pipeline'ı arka planda çalıştır
    try:
        thread = threading.Thread(
            target=run_pipeline_with_cv_url,
            args=(cv_url, session_id),
            name=f"Pipeline-{session_id[:8]}"
        )
        thread.daemon = True
        thread.start()
        
        # Thread'in başladığını doğrula
        if thread.is_alive():
            return jsonify({
                'success': True,
                'session_id': session_id,
                'message': 'İşleme başlatıldı'
            })
        else:
            # Thread başlamadıysa hata
            processing_status[session_id] = {
                'status': 'error',
                'message': 'Thread başlatılamadı',
                'progress': 0
            }
            return jsonify({
                'error': 'İşleme başlatılamadı'
            }), 500
    except Exception as e:
        processing_status[session_id] = {
            'status': 'error',
            'message': f'Hata: {str(e)}',
            'progress': 0
        }
        return jsonify({
            'error': f'İşleme başlatılamadı: {str(e)}'
        }), 500


@app.route('/status/<session_id>')
def get_status(session_id):
    """İşleme durumunu döndür"""
    if session_id not in processing_status:
        return jsonify({'error': 'Session bulunamadı'}), 404
    
    status = processing_status[session_id].copy()
    
    # Otomatik düzeltme: Eğer progress 95+ ise ve status hala 'processing' ise
    if status.get('status') in ['processing', 'queued'] and status.get('progress', 0) >= 95:
        # Sonuçlar varsa, muhtemelen tamamlandı
        if session_id in processing_results:
            status['status'] = 'completed'
            status['message'] = 'Tamamlandı! (otomatik düzeltme)'
            status['progress'] = 100
            # Global status'u da güncelle
            processing_status[session_id].update(status)
            _logger.info(f"🔧 Status otomatik düzeltildi (session: {session_id})")
    
    # Eğer tamamlandıysa sonuçları da ekle
    if status['status'] == 'completed' and session_id in processing_results:
        status['results'] = processing_results[session_id]
    
    return jsonify(status)


@app.route('/results/<session_id>')
def show_results(session_id):
    """Sonuçları göster"""
    if session_id not in processing_results:
        return redirect(url_for('index'))
    
    results = processing_results[session_id]
    return render_template('results.html', results=results, session_id=session_id)


@app.route('/feedback', methods=['POST'])
def submit_feedback():
    """Kullanıcı geri bildirimini kaydet"""
    try:
        data = request.get_json()
        project_id = data.get('project_id')
        feedback_type = data.get('feedback_type')  # 'positive' veya 'negative'
        session_id = data.get('session_id')
        
        if not all([project_id, feedback_type, session_id]):
            return jsonify({'success': False, 'error': 'Eksik parametreler'}), 400
        
        if feedback_type not in ['positive', 'negative']:
            return jsonify({'success': False, 'error': 'Geçersiz feedback tipi'}), 400
        
        # Feedback dosyası yolu
        feedback_file = project_root / "user_feedback.json"
        
        # Mevcut feedback'leri yükle
        feedbacks = []
        if feedback_file.exists():
            try:
                import json
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    feedbacks = json.load(f)
            except Exception as e:
                _logger.warning(f"Feedback dosyası okunamadı: {e}")
                feedbacks = []
        
        # Proje link'ini bul (session'dan)
        project_link = ''
        if session_id in processing_results:
            # Top 10 ve all_relevant_projects'te ara
            for project in processing_results[session_id].get('top_projects', []):
                if str(project.get('project_id', '')) == str(project_id):
                    project_link = project.get('link', '')
                    break
            if not project_link:
                for project in processing_results[session_id].get('all_relevant_projects', []):
                    if str(project.get('project_id', '')) == str(project_id):
                        project_link = project.get('link', '')
                        break
        
        # Yeni feedback ekle
        from datetime import datetime
        new_feedback = {
            'project_id': project_id,
            'project_link': project_link,
            'feedback_type': feedback_type,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat()
        }
        feedbacks.append(new_feedback)
        
        # Feedback'leri kaydet
        import json
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(feedbacks, f, indent=2, ensure_ascii=False)
        
        _logger.info(f"✅ Feedback kaydedildi: {project_id} - {feedback_type} (session: {session_id})")
        
        return jsonify({
            'success': True,
            'message': 'Geri bildiriminiz kaydedildi'
        })
        
    except Exception as e:
        _logger.error(f"Feedback kaydetme hatası: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Geri bildirim kaydedilemedi: {str(e)}'
        }), 500


@app.route('/clear-cache', methods=['POST'])
def clear_cache():
    """Önbelleği temizle (proje verilerini yeniden yüklemek için)"""
    global _cached_projects, _cached_models
    _cached_projects = None
    _cached_models = {}
    return jsonify({'success': True, 'message': 'Önbellek temizlendi'})


@app.route('/cache-status')
def cache_status():
    """Önbellek durumunu kontrol et"""
    global _cached_projects, _cached_models
    
    ip1_file = project_root / "IP1_Ciktisi_Projeler_Becerili.pkl"
    has_ip1 = ip1_file.exists()
    has_cached_projects = _cached_projects is not None
    has_cached_models = len(_cached_models) > 0
    
    return jsonify({
        'ip1_file_exists': has_ip1,
        'projects_cached': has_cached_projects,
        'models_cached': has_cached_models,
        'message': 'Hızlı mod aktif' if (has_ip1 and has_cached_models) else 'Yavaş mod (ilk çalıştırma)'
    })


if __name__ == '__main__':
    import os
    
    # Port'u environment variable'dan al veya varsayılan olarak 5001 kullan
    port = int(os.environ.get('PORT', 5001))
    
    print("=" * 80)
    print("İŞ EŞLEŞTİRME SİSTEMLERİNDE KULLANICI KONTROLÜNÜN GÜÇLENDİRİLMESİ - Web Arayüzü Başlatılıyor...")
    print("=" * 80)
    
    # Modelleri lazy loading ile yükle (sadece gerektiğinde)
    print("\nModeller lazy loading ile yüklenecek (sadece gerektiğinde)")
    print("   Bu sayede başlangıç çok daha hızlı olacak")
    try:
        # Sadece model yapısını hazırla, modelleri yükleme
        get_cached_models()
        print("✅ Model yapısı hazırlandı (modeller gerektiğinde yüklenecek)")
    except Exception as e:
        print(f"⚠️  Model yapısı hazırlanırken hata: {e}")
    
    # API cache'ini kontrol et ve yükle
    print("\n📥 API cache kontrol ediliyor...")
    try:
        check_api_for_updates()
        cache_data = load_api_cache()
        if cache_data:
            print(f"✅ API cache yüklendi: {cache_data.get('project_count', 0)} proje")
            print(f"   Son güncelleme: {cache_data.get('timestamp', 'Bilinmiyor')}")
        else:
            print("⚠️  API cache bulunamadı, ilk istekte oluşturulacak")
    except Exception as e:
        print(f"⚠️  API cache kontrolü hatası: {e}")
    
    # IP1 çıktılarını kontrol et
    ip1_file = project_root / "IP1_Ciktisi_Projeler_Becerili.pkl"
    if ip1_file.exists():
        print("✅ IP1 çıktıları bulundu - Hızlı mod aktif")
        print("   (Proje scraping'i atlanacak, sadece CV analizi yapılacak)")
    else:
        print("⚠️  IP1 çıktıları bulunamadı - Yavaş mod")
        print("   (İlk analiz uzun sürebilir, proje verileri toplanacak)")
        print("   İpucu: Önce 'python src/pipeline.py' çalıştırarak IP1 çıktılarını oluşturun")
    
    print(f"\nTarayıcınızda http://localhost:{port} adresini açın")
    print("Çıkmak için Ctrl+C tuşlarına basın\n")
    
    # Production modunda çalıştır (debug=False) - Dosya değişikliklerinde otomatik restart olmasın
    # Debug modu pipeline thread'lerini kesebilir ve session'ları kaybedebilir
    # Not: Production'da gunicorn kullanılmalı, app.run() sadece development için
    app.run(debug=False, host='0.0.0.0', port=port)
