"""
SkillNER kullanarak becerileri çıkaran modül
"""

import spacy
from spacy.matcher import PhraseMatcher
from skillNer.skill_extractor_class import SkillExtractor
from skillNer.general_params import SKILL_DB
import pandas as pd
from typing import List
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time

from ..utils.data_cleaner import filter_skills


class SkillExtractorModule:
    """SkillNER beceri çıkarma sınıfı"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # Filtreleme ayarları
        data_config = config.get('data_processing', {})
        self.min_skill_length = data_config.get('min_skill_length', 2)
        self.skill_stopwords = set(data_config.get('skill_stopwords', []))
        
        model_config = config.get('models', {}).get('skill_extraction', {})
        self.spacy_model = model_config.get('spacy_model', 'en_core_web_sm')
        
        # SkillNER kullan
        # Spacy modelini yükle
        try:
            self.nlp = spacy.load(self.spacy_model)
            self.logger.info(f"Spacy modeli yüklendi: {self.spacy_model}")
        except IOError:
            self.logger.error(f"Spacy modeli bulunamadı: {self.spacy_model}")
            raise
        
        # SkillExtractor'ı başlat
        try:
            self.skill_extractor = SkillExtractor(self.nlp, SKILL_DB, PhraseMatcher)
            self.logger.info("SkillExtractor başarıyla başlatıldı")
        except Exception as e:
            self.logger.error(f"SkillExtractor başlatılırken hata: {e}")
            raise
        
        # Performans optimizasyon parametreleri (config'den al)
        skill_config = self.config.get('models', {}).get('skill_extraction', {})
        self.max_text_length = skill_config.get('max_text_length', 8000)  # 0 = sınırsız
        self.max_ngram_results = skill_config.get('max_ngram_results', 40)  # 0 = sınırsız
    
    def extract_skills_from_text(self, text: str) -> List[str]:
        """
        Metinden becerileri çıkarır.
        
        Args:
            text: İşlenecek metin
        
        Returns:
            Çıkarılan beceri listesi
        """
        # SkillNER kullan
        try:
            # Performans optimizasyonu: Çok uzun metinleri kısalt
            original_text = text
            if self.max_text_length > 0 and len(text) > self.max_text_length:
                text = text[:self.max_text_length]
                self.logger.debug(f"Metin {len(original_text)} -> {len(text)} karaktere kısaltıldı (performans için)")
            
            annotations = self.skill_extractor.annotate(text)
            
            skills = []
            
            # Tam eşleşmeler (daha hızlı, öncelikli)
            for match in annotations['results']['full_matches']:
                skills.append(match['doc_node_value'])
            
            # Ngram eşleşmeler (performans için limitli)
            ngram_results = annotations.get('results', {}).get('ngram_scored', [])
            # En iyi N ngram'ı al (performans için)
            if self.max_ngram_results > 0 and len(ngram_results) > self.max_ngram_results:
                ngram_results = ngram_results[:self.max_ngram_results]
            for match in ngram_results:
                skills.append(match['doc_node_value'])
            
            # Benzersiz yap ve normalize et
            skills = list(set([s.lower() for s in skills]))
            
            # Filtrele
            skills = filter_skills(skills, self.min_skill_length, self.skill_stopwords)
            
            return skills
            
        except (IndexError, ValueError, KeyError) as e:
            # skillNer kütüphanesinden gelen beklenen hatalar (bazı metinler için)
            # Bu hatalar normaldir ve boş liste döndürülerek işlem devam eder
            self.logger.debug(f"Beceri çıkarma hatası (metin atlandı): {type(e).__name__}")
            return []
        except Exception as e:
            # Beklenmeyen hatalar için uyarı ver
            self.logger.warning(f"Beceri çıkarma hatası: {e}")
            return []
    
    def extract_skills_from_dataframe(self, df: pd.DataFrame, text_column: str = "tam_aciklama", parallel: bool = True) -> pd.DataFrame:
        """
        DataFrame'deki tüm metinlerden becerileri çıkarır (paralel veya sequential).
        
        Args:
            df: İşlenecek DataFrame
            text_column: Metin sütunu adı
            parallel: Paralel işleme kullan (True) veya sequential (False) - IGNORED, her zaman paralel kullanılır
        
        Returns:
            Beceri listesi eklenmiş DataFrame
        """
        total = len(df)
        self.logger.info(f"{total} adet metin işleniyor...")
        
        # Her zaman paralel kullan (10'dan fazla proje varsa) - parallel parametresi artık kullanılmıyor
        if total > 10:
            self.logger.info(f"🚀 Paralel beceri çıkarma modu aktif (10'dan fazla proje)")
            return self._extract_skills_parallel(df, text_column)
        else:
            self.logger.info(f"📝 Sequential beceri çıkarma modu (10 veya daha az proje)")
            return self._extract_skills_sequential(df, text_column)
    
    def _extract_skills_sequential(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Sequential beceri çıkarma (eski yöntem)"""
        skill_lists = []
        total = len(df)
        error_count = 0
        
        for idx, text in enumerate(df[text_column].astype(str)):
            try:
                skills = self.extract_skills_from_text(text)
                skill_lists.append(skills)
            except Exception as e:
                error_count += 1
                skill_lists.append([])
            
            # İlerleme raporu (daha sık göster - sequential mode için)
            if (idx + 1) % 25 == 0:
                self.logger.info(f"İlerleme: {idx+1}/{total} metin işlendi ({(idx+1)*100//total}%)")
        
        # Hata özeti
        if error_count > 0:
            self.logger.info(f"⚠️  {error_count}/{total} metin için beceri çıkarma hatası (normal, bazı metinler işlenemeyebilir)")
        
        df_result = df.copy()
        df_result["cikarilan_beceriler"] = skill_lists
        
        self.logger.info(f"✅ Tüm metinler işlendi")
        return df_result
    
    def _extract_skills_parallel(self, df: pd.DataFrame, text_column: str) -> pd.DataFrame:
        """Paralel beceri çıkarma (hızlı yöntem)"""
        total = len(df)
        
        # Worker sayısını config'den al (skill_extraction_workers varsa onu kullan, yoksa num_workers)
        perf_config = self.config.get('performance', {})
        num_workers = perf_config.get('skill_extraction_workers') or perf_config.get('num_workers', 8)
        # CPU sayısına göre sınırla (çok fazla thread memory sorunu yaratabilir)
        import os
        cpu_count = os.cpu_count() or 4
        # Optimal thread sayısı: CPU sayısı veya config değeri, max 16 (daha fazla thread memory sorunu yaratabilir)
        max_workers = min(num_workers, cpu_count * 2, 16)  # Max 16 thread
        
        self.logger.info(f"🚀 Paralel beceri çıkarma başlatılıyor: {max_workers} thread")
        
        # Her worker için ayrı model instance'ı oluştur
        def create_worker_extractor():
            """Her thread için ayrı model instance'ı oluşturur"""
            nlp = spacy.load(self.spacy_model)
            skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)
            return skill_extractor
        
        # Worker pool oluştur - Sequential yükleme (memory için daha güvenli)
        self.logger.info(f"   ⏳ {max_workers} worker modeli yükleniyor...")
        worker_extractors = []
        
        for i in range(max_workers):
            try:
                extractor = create_worker_extractor()
                worker_extractors.append(extractor)
                if (i + 1) % 4 == 0 or (i + 1) == max_workers:
                    self.logger.info(f"   ✅ {i + 1}/{max_workers} worker modeli yüklendi")
            except Exception as e:
                self.logger.warning(f"   ⚠️  Worker {i+1} yüklenemedi: {e}, daha az worker ile devam ediliyor")
                if len(worker_extractors) == 0:
                    self.logger.error("❌ Hiç worker modeli yüklenemedi, sequential moda geçiliyor")
                    return self._extract_skills_sequential(df, text_column)
                break
        
        max_workers = len(worker_extractors)  # Gerçek worker sayısı
        self.logger.info(f"   ✅ {max_workers} worker modeli hazır, paralel işleme başlıyor")
        
        worker_lock = threading.Lock()
        worker_idx = 0
        
        def get_worker_extractor():
            """Round-robin ile worker extractor döndürür"""
            nonlocal worker_idx
            with worker_lock:
                extractor = worker_extractors[worker_idx]
                worker_idx = (worker_idx + 1) % max_workers
                return extractor
        
        # İşlenecek verileri hazırla
        texts = df[text_column].astype(str).tolist()
        
        def process_single_text(idx_text_pair):
            """Tek bir metni işler"""
            idx, text = idx_text_pair
            try:
                # SkillNER kullan
                worker_extractor = get_worker_extractor()
                
                # Performans optimizasyonu: Çok uzun metinleri kısalt
                max_text_len = self.config.get('models', {}).get('skill_extraction', {}).get('max_text_length', 8000)
                if max_text_len > 0 and len(text) > max_text_len:
                    text = text[:max_text_len]
                
                # Becerileri çıkar
                annotations = worker_extractor.annotate(text)
                
                skills = []
                # Tam eşleşmeler (daha hızlı, öncelikli)
                for match in annotations['results']['full_matches']:
                    skills.append(match['doc_node_value'])
                
                # Ngram eşleşmeler (performans için limitli)
                ngram_results = annotations.get('results', {}).get('ngram_scored', [])
                max_ngram = self.config.get('models', {}).get('skill_extraction', {}).get('max_ngram_results', 40)
                # En iyi N ngram'ı al (performans için)
                if max_ngram > 0 and len(ngram_results) > max_ngram:
                    ngram_results = ngram_results[:max_ngram]
                for match in ngram_results:
                    skills.append(match['doc_node_value'])
                
                # Benzersiz yap ve normalize et (optimize edilmiş)
                skills = list(set([s.lower() for s in skills]))
                
                # Filtrele
                skills = filter_skills(skills, self.min_skill_length, self.skill_stopwords)
                
                return (idx, skills)
            except (IndexError, ValueError, KeyError):
                # Beklenen hatalar (normal)
                return (idx, [])
            except Exception as e:
                # Beklenmeyen hatalar
                self.logger.debug(f"Beceri çıkarma hatası (idx {idx}): {type(e).__name__}")
                return (idx, [])
        
        # Paralel işleme - as_completed ile gerçek paralel işleme
        skill_lists = [None] * total
        error_count = 0
        completed = 0
        start_time = time.time()
        
        self.logger.info(f"🚀 Paralel beceri çıkarma başlatıldı: {total} metin, {max_workers} thread")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Tüm görevleri gönder (hepsi aynı anda başlar)
            future_to_idx = {
                executor.submit(process_single_text, (idx, text)): idx 
                for idx, text in enumerate(texts)
            }
            
            # Sonuçları topla (as_completed ile - tamamlanan ilk gelen alınır)
            for future in as_completed(future_to_idx):
                original_idx = future_to_idx[future]
                try:
                    idx, skills = future.result()
                    skill_lists[idx] = skills
                    completed += 1
                    
                    # İlerleme raporu (her 20'de bir veya tamamlanınca - çok sık log yavaşlatır)
                    if completed % 20 == 0 or completed == total:
                        elapsed = time.time() - start_time
                        remaining = total - completed
                        avg_time = elapsed / completed if completed > 0 else 0
                        estimated_remaining = avg_time * remaining / max_workers
                        self.logger.info(
                            f"📊 İlerleme: {completed}/{total} metin işlendi "
                            f"({completed*100//total}%) | "
                            f"Tahmini kalan: {int(estimated_remaining//60)}dk {int(estimated_remaining%60)}sn | "
                            f"Paralel: {max_workers} thread"
                        )
                except Exception as e:
                    error_count += 1
                    skill_lists[original_idx] = []
                    self.logger.debug(f"Beceri çıkarma hatası (idx {original_idx}): {type(e).__name__}")
        
        # Hata özeti
        if error_count > 0:
            self.logger.info(f"⚠️  {error_count}/{total} metin için beceri çıkarma hatası (normal, bazı metinler işlenemeyebilir)")
        
        df_result = df.copy()
        df_result["cikarilan_beceriler"] = skill_lists
        
        self.logger.info(f"✅ Tüm metinler paralel olarak işlendi ({max_workers} thread)")
        return df_result

