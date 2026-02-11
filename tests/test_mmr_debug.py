#!/usr/bin/env python3
"""
MMR debug script'i - MMR'ın neden Diversity Score'u düşürdüğünü analiz eder
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Proje root'unu bul
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.scoring.final_ranking import FinalRankingCalculator

def debug_mmr():
    """MMR'ı debug et"""
    print("=" * 80)
    print("🔍 MMR DEBUG - Neden Diversity Score Düşüyor?")
    print("=" * 80)
    
    # Config yükle
    config_path = project_root / "config" / "config.yaml"
    config = load_config(str(config_path))
    
    # Logger
    logger_config = config.get('logging', {})
    logger = setup_logger(
        name=logger_config.get('name', 'tubitak_proje'),
        log_level=logger_config.get('level', 'INFO'),
        log_file=logger_config.get('file', 'logs/proje.log'),
        console=False,
        log_format=logger_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # CSV yükle
    csv_file = project_root / "outputs" / "results" / "IP5_Nihai_Adil_Siralama.csv"
    if not csv_file.exists():
        print(f"❌ CSV bulunamadı: {csv_file}")
        return
    
    df_ranked = pd.read_csv(csv_file)
    print(f"✅ CSV yüklendi: {len(df_ranked)} proje")
    
    # Vektörleri yükle
    vector_file = project_root / "IP1_Ciktisi_Projeler_Vektorlu.pkl"
    if vector_file.exists():
        with open(vector_file, 'rb') as f:
            vectors_data = pickle.load(f)
        
        if isinstance(vectors_data, dict):
            df_ranked['beceri_vektorleri'] = df_ranked['link'].apply(
                lambda x: vectors_data.get(str(x)) if pd.notna(x) else None
            )
        elif isinstance(vectors_data, pd.DataFrame):
            if 'link' in vectors_data.columns and 'beceri_vektorleri' in vectors_data.columns:
                # Merge yaparken link'leri string'e çevir
                vectors_df = vectors_data[['link', 'beceri_vektorleri']].copy()
                vectors_df['link'] = vectors_df['link'].astype(str)
                df_ranked['link'] = df_ranked['link'].astype(str)
                df_ranked = df_ranked.merge(
                    vectors_df,
                    on='link',
                    how='left'
                )
    
    ranking_calc = FinalRankingCalculator(config, logger)
    
    # Baseline metrikleri
    print("\n📊 BASELINE (MMR ÖNCESİ)")
    print("=" * 80)
    relevance_b, diversity_b, conf_b, rc_b, uc_b = ranking_calc.calculate_performance_metrics(
        df_ranked, top_k=5, vector_column="beceri_vektorleri", m_score_column="M_Skoru"
    )
    print(f"Diversity: {diversity_b:.6f}")
    print(f"Relevance: {relevance_b:.2f}")
    
    # Top-5 projeleri göster
    print("\nTop-5 Projeler (Baseline):")
    for i, (idx, row) in enumerate(df_ranked.head(5).iterrows(), 1):
        print(f"  {i}. Adil_Skor={row.get('Adil_Skor', 0):.2f}, Link={row.get('link', 'N/A')[:50]}...")
    
    # Lambda=0.0 ile MMR (sadece diversity)
    print("\n🔄 MMR (lambda=0.0 - Sadece Diversity)")
    print("=" * 80)
    df_mmr = ranking_calc._apply_mmr_ranking(
        df_ranked.copy(),
        top_k=10,
        lambda_param=0.0,  # Sadece diversity
        vector_column="beceri_vektorleri"
    )
    
    # MMR sonrası metrikleri
    print("\n📊 MMR SONRASI (lambda=0.0)")
    print("=" * 80)
    relevance_a, diversity_a, conf_a, rc_a, uc_a = ranking_calc.calculate_performance_metrics(
        df_mmr, top_k=5, vector_column="beceri_vektorleri", m_score_column="M_Skoru"
    )
    print(f"Diversity: {diversity_a:.6f} (değişim: {diversity_a - diversity_b:+.6f})")
    print(f"Relevance: {relevance_a:.2f} (değişim: {relevance_a - relevance_b:+.2f})")
    
    # Top-5 projeleri göster
    print("\nTop-5 Projeler (MMR sonrası):")
    for i, (idx, row) in enumerate(df_mmr.head(5).iterrows(), 1):
        print(f"  {i}. Adil_Skor={row.get('Adil_Skor', 0):.2f}, Link={row.get('link', 'N/A')[:50]}...")
    
    # Projelerin değişip değişmediğini kontrol et
    baseline_links = [str(row.get('link', '')) for _, row in df_ranked.head(5).iterrows()]
    mmr_links = [str(row.get('link', '')) for _, row in df_mmr.head(5).iterrows()]
    
    changed = sum(1 for b, m in zip(baseline_links, mmr_links) if b != m)
    print(f"\n📊 Değişiklik: {changed}/5 proje pozisyonu değişti")
    
    if diversity_a < diversity_b:
        print("\n⚠️  SORUN: MMR Diversity Score'u DÜŞÜRDÜ!")
        print("   Bu beklenmeyen bir durum. MMR diversity'yi artırmalı.")
    elif diversity_a > diversity_b:
        print("\n✅ MMR Diversity Score'u artırdı!")
    else:
        print("\n⚠️  MMR Diversity Score'u değiştirmedi")

if __name__ == "__main__":
    debug_mmr()
