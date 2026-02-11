#!/usr/bin/env python3
"""
MMR (Maximal Marginal Relevance) test script'i
Mevcut sonuçları yükleyip MMR'ın çalışıp çalışmadığını test eder
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Proje root'unu bul
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.scoring.final_ranking import FinalRankingCalculator

def test_mmr():
    """MMR'ı test et"""
    print("=" * 80)
    print("🔍 MMR (Maximal Marginal Relevance) TEST")
    print("=" * 80)
    
    # Config yükle
    config_path = project_root / "config" / "config.yaml"
    config = load_config(str(config_path))
    
    # Logger ayarla
    logger_config = config.get('logging', {})
    logger = setup_logger(
        name=logger_config.get('name', 'tubitak_proje'),
        log_level=logger_config.get('level', 'INFO'),
        log_file=logger_config.get('file', 'logs/proje.log'),
        console=logger_config.get('console', True),
        log_format=logger_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # MMR ayarlarını kontrol et
    mmr_config = config.get('diversity_aware_ranking', {})
    mmr_enabled = mmr_config.get('enabled', False)
    mmr_top_k = mmr_config.get('top_k', 10)
    mmr_lambda = mmr_config.get('lambda', 0.5)
    
    print(f"\n📋 MMR Ayarları:")
    print(f"  enabled: {mmr_enabled}")
    print(f"  top_k: {mmr_top_k}")
    print(f"  lambda: {mmr_lambda}")
    
    if not mmr_enabled:
        print("\n⚠️  MMR devre dışı! config.yaml'da diversity_aware_ranking.enabled: true yapın.")
        return
    
    # Mevcut sonuçları yükle
    csv_file = project_root / "outputs" / "results" / "IP5_Nihai_Adil_Siralama.csv"
    if not csv_file.exists():
        print(f"\n❌ Sonuç dosyası bulunamadı: {csv_file}")
        return
    
    print(f"\n📂 Sonuçlar yükleniyor: {csv_file}")
    df_ranked = pd.read_csv(csv_file)
    print(f"✅ {len(df_ranked)} proje yüklendi")
    
    # Vektörleri yükle (birkaç olası konumda ara)
    vector_files = [
        project_root / "data/processed/IP1_Ciktisi_Projeler_Vektorlu.pkl",  # Proje root'unda (en yaygın)
        project_root / "outputs" / "processed" / "IP1_Ciktisi_Projeler_Vektorlu.pkl",
    ]
    
    vectors_loaded = False
    for vector_file in vector_files:
        if vector_file.exists():
            print(f"\n📂 Vektörler yükleniyor: {vector_file}")
            try:
                with open(vector_file, 'rb') as f:
                    vectors_data = pickle.load(f)
                
                print(f"   Vektör veri tipi: {type(vectors_data).__name__}")
                if isinstance(vectors_data, pd.DataFrame):
                    print(f"   DataFrame satır sayısı: {len(vectors_data)}")
                    if 'link' in vectors_data.columns:
                        print(f"   DataFrame sütunları: {list(vectors_data.columns)}")
                elif isinstance(vectors_data, dict):
                    print(f"   Dict key sayısı: {len(vectors_data)}")
                    if len(vectors_data) > 0:
                        print(f"   İlk key örneği: {list(vectors_data.keys())[0]}")
                elif isinstance(vectors_data, list):
                    print(f"   List uzunluğu: {len(vectors_data)}")
                
                # Vektörleri DataFrame'e ekle
                if 'link' not in df_ranked.columns:
                    print("⚠️  'link' sütunu bulunamadı, vektörler eklenemedi")
                    continue
                
                if isinstance(vectors_data, pd.DataFrame):
                    # DataFrame ise 'link' ile merge et
                    if 'link' in df_ranked.columns and 'link' in vectors_data.columns:
                        # Önce mevcut 'beceri_vektorleri' sütununu kaldır (varsa)
                        if 'beceri_vektorleri' in df_ranked.columns:
                            df_ranked = df_ranked.drop(columns=['beceri_vektorleri'])
                        
                        # Merge et
                        df_ranked = df_ranked.merge(
                            vectors_data[['link', 'beceri_vektorleri']],
                            on='link',
                            how='left'
                        )
                        vectors_loaded_count = df_ranked['beceri_vektorleri'].notna().sum()
                        print(f"✅ Vektörler merge edildi (DataFrame): {vectors_loaded_count}/{len(df_ranked)} projede vektör var")
                        vectors_loaded = True
                        break
                    elif 'beceri_vektorleri' in vectors_data.columns and len(vectors_data) == len(df_ranked):
                        # Index'e göre ekle (eğer sayı eşleşiyorsa)
                        df_ranked['beceri_vektorleri'] = vectors_data['beceri_vektorleri'].values
                        vectors_loaded_count = df_ranked['beceri_vektorleri'].notna().sum()
                        print(f"✅ Vektörler index ile eklendi (DataFrame): {vectors_loaded_count}/{len(df_ranked)} projede vektör var")
                        vectors_loaded = True
                        break
                    else:
                        print(f"⚠️  DataFrame formatı beklenenle uyuşmuyor (link: {'link' in vectors_data.columns}, beceri_vektorleri: {'beceri_vektorleri' in vectors_data.columns}, uzunluk: {len(vectors_data)} vs {len(df_ranked)})")
                elif isinstance(vectors_data, dict):
                    # Dict ise link'e göre map et (string key'ler için)
                    # Hem string hem de orijinal formatı dene
                    def get_vector(link_val):
                        if pd.isna(link_val):
                            return None
                        link_str = str(link_val)
                        # Önce direkt string key ile dene
                        if link_str in vectors_data:
                            return vectors_data[link_str]
                        # Sonra tüm key'leri string'e çevirip dene
                        for k, v in vectors_data.items():
                            if str(k) == link_str:
                                return v
                        return None
                    
                    df_ranked['beceri_vektorleri'] = df_ranked['link'].apply(get_vector)
                    vectors_loaded_count = df_ranked['beceri_vektorleri'].notna().sum()
                    print(f"✅ Vektörler map edildi (dict): {vectors_loaded_count}/{len(df_ranked)} projede vektör var")
                    vectors_loaded = True
                    break
                elif isinstance(vectors_data, list):
                    # List ise index'e göre ekle (sadece sayı eşleşiyorsa)
                    if len(vectors_data) == len(df_ranked):
                        df_ranked['beceri_vektorleri'] = vectors_data[:len(df_ranked)]
                        vectors_loaded_count = len([v for v in df_ranked['beceri_vektorleri'] if v is not None])
                        print(f"✅ Vektörler list olarak eklendi: {len(vectors_data)} vektör")
                        vectors_loaded = True
                        break
                    else:
                        print(f"⚠️  Vektör listesi uzunluğu ({len(vectors_data)}) proje sayısı ({len(df_ranked)}) ile eşleşmiyor - link bazlı eşleştirme gerekli")
                else:
                    print(f"⚠️  Vektör verisi beklenmeyen tip: {type(vectors_data)}")
                    print(f"    İçerik örneği: {str(vectors_data)[:200]}")
            except Exception as e:
                import traceback
                print(f"⚠️  Vektör dosyası okunamadı: {e}")
                print(f"    Detay: {traceback.format_exc()[:300]}")
                continue
    
    if not vectors_loaded:
        print(f"\n⚠️  Vektör dosyası bulunamadı veya yüklenemedi!")
        print(f"   Kontrol edilen dosyalar:")
        for vf in vector_files:
            print(f"     - {vf} ({'✓ VAR' if vf.exists() else '✗ YOK'})")
        print(f"\n   Diversity hesaplaması ve MMR için vektörler gerekli!")
        return
    
    # FinalRankingCalculator oluştur
    ranking_calc = FinalRankingCalculator(config, logger)
    
    # MMR öncesi diversity hesapla
    print("\n" + "=" * 80)
    print("📊 MMR ÖNCESİ METRİKLER")
    print("=" * 80)
    
    relevance_before, diversity_before, confidence_before, rc_before, uc_before = ranking_calc.calculate_performance_metrics(
        df_ranked,
        top_k=5,
        vector_column="beceri_vektorleri",
        m_score_column="M_Skoru"
    )
    
    print(f"  Relevance: {relevance_before:.4f}")
    print(f"  Diversity: {diversity_before:.4f}")
    print(f"  Confidence: {confidence_before:.4f}")
    print(f"  RC: {rc_before:.4f}")
    print(f"  UC: {uc_before:.4f}")
    
    # MMR uygula
    print("\n" + "=" * 80)
    print("🔄 MMR UYGULANIYOR")
    print("=" * 80)
    
    if 'beceri_vektorleri' not in df_ranked.columns:
        print("❌ 'beceri_vektorleri' sütunu yok! MMR çalışamaz.")
        return
    
    df_mmr = ranking_calc._apply_mmr_ranking(
        df_ranked.copy(),
        top_k=mmr_top_k,
        lambda_param=mmr_lambda,
        vector_column="beceri_vektorleri"
    )
    
    # MMR sonrası diversity hesapla
    print("\n" + "=" * 80)
    print("📊 MMR SONRASI METRİKLER")
    print("=" * 80)
    
    relevance_after, diversity_after, confidence_after, rc_after, uc_after = ranking_calc.calculate_performance_metrics(
        df_mmr,
        top_k=5,
        vector_column="beceri_vektorleri",
        m_score_column="M_Skoru"
    )
    
    print(f"  Relevance: {relevance_after:.4f}")
    print(f"  Diversity: {diversity_after:.4f}  ({diversity_after - diversity_before:+.4f})")
    print(f"  Confidence: {confidence_after:.4f} ({confidence_after - confidence_before:+.4f})")
    print(f"  RC: {rc_after:.4f}")
    print(f"  UC: {uc_after:.4f}")
    
    # Sonuç özeti
    print("\n" + "=" * 80)
    print("📈 SONUÇ ÖZETİ")
    print("=" * 80)
    print(f"Diversity değişimi: {diversity_before:.4f} → {diversity_after:.4f} ({diversity_after - diversity_before:+.4f})")
    
    if diversity_after > diversity_before:
        print("✅ MMR Diversity Score'u artırdı!")
    elif diversity_after == diversity_before:
        print("⚠️  MMR Diversity Score'u değiştirmedi (projeler zaten farklı veya çok benzer)")
    else:
        print("⚠️  MMR Diversity Score'u düşürdü (beklenmeyen durum)")
    
    # Top-5 projeleri karşılaştır
    print("\n" + "=" * 80)
    print("🔍 TOP-5 PROJELER KARŞILAŞTIRMASI")
    print("=" * 80)
    
    print("\nMMR Öncesi (İlk 5):")
    before_links = []
    for i, (idx, row) in enumerate(df_ranked.head(5).iterrows(), 1):
        link = row.get('link', 'N/A')
        before_links.append(link)
        print(f"  {i}. Adil_Skor={row.get('Adil_Skor', 0):.2f}, "
              f"M_Skoru={row.get('M_Skoru', 0):.2f}, "
              f"Link={str(link)[:60]}...")
    
    print("\nMMR Sonrası (İlk 5):")
    after_links = []
    for i, (idx, row) in enumerate(df_mmr.head(5).iterrows(), 1):
        link = row.get('link', 'N/A')
        after_links.append(link)
        print(f"  {i}. Adil_Skor={row.get('Adil_Skor', 0):.2f}, "
              f"M_Skoru={row.get('M_Skoru', 0):.2f}, "
              f"Link={str(link)[:60]}...")
    
    # Projelerin değişip değişmediğini kontrol et
    changed_count = sum(1 for b, a in zip(before_links[:5], after_links[:5]) if b != a)
    print(f"\n📊 Sıralama Değişikliği: {changed_count}/5 proje pozisyonu değişti")
    
    # Vektör benzerliğini analiz et
    if 'beceri_vektorleri' in df_mmr.columns:
        print("\n" + "=" * 80)
        print("🔬 VEKTÖR BENZERLİK ANALİZİ (Top-5)")
        print("=" * 80)
        
        top5_mmr = df_mmr.head(5)
        vectors_top5 = []
        valid_indices = []
        
        for idx, row in top5_mmr.iterrows():
            vec = row.get('beceri_vektorleri')
            if vec is not None:
                try:
                    vec_array = np.array(vec)
                    if vec_array.ndim > 1:
                        vec_array = vec_array.flatten()
                    norm = np.linalg.norm(vec_array)
                    if norm > 0:
                        vec_array = vec_array / norm
                        vectors_top5.append(vec_array)
                        valid_indices.append(idx)
                except:
                    pass
        
        if len(vectors_top5) >= 2:
            vectors_matrix = np.vstack(vectors_top5)
            similarity_matrix = cosine_similarity(vectors_matrix)
            np.fill_diagonal(similarity_matrix, 0.0)
            
            mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
            if mask.sum() > 0:
                avg_sim = similarity_matrix[mask].mean()
                min_sim = similarity_matrix[mask].min()
                max_sim = similarity_matrix[mask].max()
                print(f"  Ortalama benzerlik: {avg_sim:.4f}")
                print(f"  Min benzerlik: {min_sim:.4f}")
                print(f"  Max benzerlik: {max_sim:.4f}")
                print(f"  Diversity Score (1 - avg): {1.0 - avg_sim:.4f}")
                
                if avg_sim > 0.95:
                    print(f"\n⚠️  Top-5 projeler ÇOK BENZER (benzerlik > 0.95)!")
                    print(f"   Lambda değerini düşürmeyi deneyin (örn. 0.3 veya 0.4)")
                elif avg_sim > 0.9:
                    print(f"\n⚠️  Top-5 projeler benzer (benzerlik > 0.9)")
                    print(f"   Lambda değerini düşürmeyi deneyin (örn. 0.4)")
    
    print("\n💡 ÖNERİLER:")
    if diversity_after == 0.0:
        print("  1. Lambda değerini düşürün (config.yaml): lambda: 0.3 veya 0.4")
        print("  2. Bu, daha fazla diversity ağırlığı verir (relevance yerine)")
        print("  3. Pipeline'ı yeniden çalıştırın veya test script'ini lambda ile tekrar çalıştırın")
    
    print("\n✅ Test tamamlandı!")

if __name__ == "__main__":
    test_mmr()
