#!/usr/bin/env python3
"""
Mevcut IP5 sonuçlarından metrikleri hesaplayıp kaydetme scripti
"""
import sys
from pathlib import Path

# Proje root'unu ekle
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json

# Config ve logger yükle
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.scoring.final_ranking import FinalRankingCalculator

def load_existing_results():
    """Mevcut IP5 sonuçlarını yükle"""
    print("📂 Mevcut sonuçlar yükleniyor...")
    
    # IP5 CSV'yi yükle
    csv_file = project_root / "outputs/results/IP5_Nihai_Adil_Siralama.csv"
    if not csv_file.exists():
        print(f"❌ IP5 CSV dosyası bulunamadı: {csv_file}")
        return None, None
    
    df_ranked = pd.read_csv(csv_file)
    print(f"✅ IP5 CSV yüklendi: {len(df_ranked)} proje")
    
    # Vektörleri yükle (diversity için gerekli)
    vector_file = project_root / "data/processed/IP1_Ciktisi_Projeler_Vektorlu.pkl"
    vectors_dict = None
    if vector_file.exists():
        try:
            with open(vector_file, 'rb') as f:
                vectors_dict = pickle.load(f)
            if isinstance(vectors_dict, pd.DataFrame):
                print(f"✅ Vektörler yüklendi: {len(vectors_dict)} proje (DataFrame)")
            elif isinstance(vectors_dict, (dict, list)):
                print(f"✅ Vektörler yüklendi: {len(vectors_dict)} proje ({type(vectors_dict).__name__})")
            else:
                print(f"✅ Vektörler yüklendi: {type(vectors_dict).__name__}")
            
            # DataFrame'e vektörleri ekle
            if 'link' in df_ranked.columns:
                if isinstance(vectors_dict, dict):
                    # Dict ise link'e göre map et
                    df_ranked['beceri_vektorleri'] = df_ranked['link'].map(lambda x: vectors_dict.get(str(x)))
                    missing_count = df_ranked['beceri_vektorleri'].isna().sum()
                    if missing_count > 0:
                        print(f"⚠️  {missing_count} proje için vektör bulunamadı")
                elif isinstance(vectors_dict, pd.DataFrame):
                    # DataFrame ise link'e göre merge et
                    if 'link' in vectors_dict.columns and 'beceri_vektorleri' in vectors_dict.columns:
                        df_ranked = df_ranked.merge(
                            vectors_dict[['link', 'beceri_vektorleri']], 
                            on='link', 
                            how='left'
                        )
                        missing_count = df_ranked['beceri_vektorleri'].isna().sum()
                        if missing_count > 0:
                            print(f"⚠️  {missing_count} proje için vektör bulunamadı")
                    elif 'beceri_vektorleri' in vectors_dict.columns and len(vectors_dict) == len(df_ranked):
                        # Index'e göre ekle (eğer sayı eşleşiyorsa)
                        df_ranked['beceri_vektorleri'] = vectors_dict['beceri_vektorleri'].values
                    else:
                        print(f"⚠️  DataFrame formatı beklenenle uyuşmuyor")
                elif isinstance(vectors_dict, list):
                    # List ise index'e göre ekle (sadece sayı eşleşiyorsa)
                    if len(vectors_dict) == len(df_ranked):
                        df_ranked['beceri_vektorleri'] = vectors_dict[:len(df_ranked)]
                    else:
                        print(f"⚠️  Vektör sayısı ({len(vectors_dict)}) proje sayısı ({len(df_ranked)}) ile eşleşmiyor - link bazlı eşleştirme gerekli")
                else:
                    print(f"⚠️  Vektör formatı tanınmadı: {type(vectors_dict)}")
                    print(f"    İçerik örneği: {str(vectors_dict)[:200]}")
        except Exception as e:
            print(f"⚠️  Vektörler yüklenemedi (diversity hesaplanamayabilir): {e}")
    else:
        print(f"⚠️  Vektör dosyası bulunamadı (diversity hesaplanamayabilir): {vector_file}")
    
    return df_ranked, vectors_dict

def calculate_and_save_metrics():
    """Metrikleri hesapla ve kaydet"""
    print("\n" + "="*60)
    print("📊 METRİKLERİ HESAPLAYIP KAYDETME")
    print("="*60 + "\n")
    
    # Sonuçları yükle
    df_ranked, vectors_dict = load_existing_results()
    if df_ranked is None:
        print("❌ Sonuçlar yüklenemedi, metrikler hesaplanamıyor")
        return
    
    # Config ve logger yükle
    config = load_config()
    log_config = config.get('logging', {})
    logger = setup_logger(
        name=log_config.get('name', 'tubitak_proje'),
        log_level=log_config.get('level', 'INFO'),
        log_file=log_config.get('file', None),
        console=log_config.get('console', True),
        log_format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # FinalRankingCalculator oluştur
    ranking_calc = FinalRankingCalculator(config, logger)
    
    # Metrikleri hesapla
    print("\n🔢 Metrikler hesaplanıyor...")
    try:
        relevance, diversity, confidence, rc, uc = ranking_calc.calculate_performance_metrics(
            df_ranked,
            top_k=5,
            vector_column="beceri_vektorleri",
            m_score_column="M_Skoru"
        )
        
        print(f"\n✅ Metrikler hesaplandı:")
        print(f"   Relevance Accuracy: {relevance:.4f}")
        print(f"   Diversity Score: {diversity:.4f}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   RC (Recommendation Coverage): {rc:.4f}")
        print(f"   UC (User Coverage): {uc:.4f}")
        
        # Metrikleri kaydet
        print("\n💾 Metrikler kaydediliyor...")
        ranking_calc._save_metrics_report(relevance, diversity, confidence, rc, uc, df_ranked)
        ranking_calc._create_metrics_visualization(relevance, diversity, confidence, rc, uc)
        
        print("\n✅ Metrikler başarıyla kaydedildi!")
        print(f"   📁 Dizin: {project_root / 'outputs' / 'metrics'}")
        
    except Exception as e:
        print(f"\n❌ Metrik hesaplama hatası: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    calculate_and_save_metrics()
