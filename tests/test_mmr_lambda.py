#!/usr/bin/env python3
"""
MMR Lambda parametresi test script'i
Farklı lambda değerleri ile MMR'ı test eder ve en iyi Diversity Score'u bulur
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Proje root'unu bul
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.scoring.final_ranking import FinalRankingCalculator

def test_different_lambdas():
    """Farklı lambda değerleri ile MMR'ı test et"""
    print("=" * 80)
    print("🔍 MMR LAMBDA PARAMETRESİ TEST")
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
        console=False,  # Sessiz mod
        log_format=logger_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # MMR ayarlarını kontrol et
    mmr_config = config.get('diversity_aware_ranking', {})
    mmr_top_k = mmr_config.get('top_k', 10)
    
    # Mevcut sonuçları yükle
    csv_file = project_root / "outputs" / "results" / "IP5_Nihai_Adil_Siralama.csv"
    if not csv_file.exists():
        print(f"\n❌ Sonuç dosyası bulunamadı: {csv_file}")
        return
    
    print(f"\n📂 Sonuçlar yükleniyor: {csv_file}")
    df_ranked = pd.read_csv(csv_file)
    print(f"✅ {len(df_ranked)} proje yüklendi")
    
    # Vektörleri yükle
    vector_files = [
        project_root / "IP1_Ciktisi_Projeler_Vektorlu.pkl",
        project_root / "outputs" / "processed" / "IP1_Ciktisi_Projeler_Vektorlu.pkl",
        project_root / "outputs" / "IP1_Ciktisi_Projeler_Vektorlu.pkl",
    ]
    
    vectors_loaded = False
    for vector_file in vector_files:
        if vector_file.exists():
            print(f"\n📂 Vektörler yükleniyor: {vector_file}")
            try:
                with open(vector_file, 'rb') as f:
                    vectors_data = pickle.load(f)
                
                if 'link' not in df_ranked.columns:
                    continue
                
                if isinstance(vectors_data, dict):
                    def get_vector(link_val):
                        if pd.isna(link_val):
                            return None
                        link_str = str(link_val)
                        if link_str in vectors_data:
                            return vectors_data[link_str]
                        for k, v in vectors_data.items():
                            if str(k) == link_str:
                                return v
                        return None
                    
                    df_ranked['beceri_vektorleri'] = df_ranked['link'].apply(get_vector)
                    vectors_loaded_count = df_ranked['beceri_vektorleri'].notna().sum()
                    print(f"✅ Vektörler yüklendi: {vectors_loaded_count}/{len(df_ranked)} projede vektör var")
                    vectors_loaded = True
                    break
                elif isinstance(vectors_data, pd.DataFrame):
                    if 'link' in vectors_data.columns and 'beceri_vektorleri' in vectors_data.columns:
                        if 'beceri_vektorleri' in df_ranked.columns:
                            df_ranked = df_ranked.drop(columns=['beceri_vektorleri'])
                        # Merge yaparken link'leri string'e çevir (uyumsuzluk olmasın)
                        vectors_df = vectors_data[['link', 'beceri_vektorleri']].copy()
                        vectors_df['link'] = vectors_df['link'].astype(str)
                        df_ranked['link'] = df_ranked['link'].astype(str)
                        df_ranked = df_ranked.merge(
                            vectors_df,
                            on='link',
                            how='left'
                        )
                        vectors_loaded_count = df_ranked['beceri_vektorleri'].notna().sum()
                        print(f"✅ Vektörler yüklendi: {vectors_loaded_count}/{len(df_ranked)} projede vektör var")
                        vectors_loaded = True
                        break
            except Exception as e:
                continue
    
    if not vectors_loaded or 'beceri_vektorleri' not in df_ranked.columns:
        print("❌ Vektörler yüklenemedi!")
        return
    
    # FinalRankingCalculator oluştur
    ranking_calc = FinalRankingCalculator(config, logger)
    
    # Farklı lambda değerlerini test et (0.0 = sadece diversity)
    lambda_values = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]
    
    print("\n" + "=" * 80)
    print("📊 LAMBDA DEĞERLERİ TEST EDİLİYOR")
    print("=" * 80)
    print(f"{'Lambda':<10} {'Diversity':<12} {'Relevance':<12} {'Değişim':<12} {'Not'}")
    print("-" * 80)
    
    baseline_diversity = 0.0
    baseline_relevance = 0.0
    
    # Önce baseline (MMR olmadan)
    baseline_relevance, baseline_diversity, _, _, _ = ranking_calc.calculate_performance_metrics(
        df_ranked,
        top_k=5,
        vector_column="beceri_vektorleri",
        m_score_column="M_Skoru"
    )
    
    print(f"{'Baseline':<10} {baseline_diversity:<12.4f} {baseline_relevance:<12.2f} {'-':<12} {'MMR yok'}")
    
    best_lambda = None
    best_diversity = baseline_diversity
    
    results = []
    
    for lambda_val in lambda_values:
        # MMR uygula
        df_mmr = ranking_calc._apply_mmr_ranking(
            df_ranked.copy(),
            top_k=mmr_top_k,
            lambda_param=lambda_val,
            vector_column="beceri_vektorleri"
        )
        
        # Metrikleri hesapla
        relevance, diversity, _, _, _ = ranking_calc.calculate_performance_metrics(
            df_mmr,
            top_k=5,
            vector_column="beceri_vektorleri",
            m_score_column="M_Skoru"
        )
        
        diversity_change = diversity - baseline_diversity
        relevance_change = relevance - baseline_relevance
        
        results.append({
            'lambda': lambda_val,
            'diversity': diversity,
            'relevance': relevance,
            'diversity_change': diversity_change,
            'relevance_change': relevance_change
        })
        
        note = ""
        if diversity > best_diversity:
            best_diversity = diversity
            best_lambda = lambda_val
            note = "⭐ EN İYİ"
        elif diversity == baseline_diversity:
            note = "⚠️  Değişmedi"
        
        print(f"{lambda_val:<10.1f} {diversity:<12.4f} {relevance:<12.2f} {diversity_change:+.4f}      {note}")
    
    # Özet
    print("\n" + "=" * 80)
    print("📈 SONUÇ ÖZETİ")
    print("=" * 80)
    
    if best_lambda is not None and best_diversity > baseline_diversity:
        print(f"✅ En iyi Lambda: {best_lambda}")
        print(f"   Diversity Score: {baseline_diversity:.4f} → {best_diversity:.4f} ({best_diversity - baseline_diversity:+.4f})")
        print(f"\n💡 Önerilen config ayarı:")
        print(f"   diversity_aware_ranking:")
        print(f"     enabled: true")
        print(f"     top_k: {mmr_top_k}")
        print(f"     lambda: {best_lambda}  # En iyi Diversity Score için")
    else:
        print("⚠️  Hiçbir lambda değeri Diversity Score'u artıramadı")
        print("   Bu durum, top-K projelerin gerçekten çok benzer olduğunu gösterir")
        print("   Lambda değerini daha da düşürmeyi deneyin (0.1, 0.05, 0.0) veya top_k'yı artırın")
    
    print("\n✅ Test tamamlandı!")

if __name__ == "__main__":
    test_different_lambdas()
