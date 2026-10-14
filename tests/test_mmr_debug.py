#!/usr/bin/env python3
"""
MMR debug script - analyzes why MMR decreases the Diversity Score
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

# Find the project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.scoring.final_ranking import FinalRankingCalculator

def debug_mmr():
    """Debug MMR"""
    print("=" * 80)
    print("🔍 MMR DEBUG - Why does the Diversity Score decrease?")
    print("=" * 80)
    
    # Load the config
    config_path = project_root / "config" / "config.yaml"
    config = load_config(str(config_path))
    
    # Logger
    logger_config = config.get('logging', {})
    logger = setup_logger(
        name=logger_config.get('name', 'tubitak_project'),
        log_level=logger_config.get('level', 'INFO'),
        log_file=logger_config.get('file', 'logs/project.log'),
        console=False,
        log_format=logger_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Load the CSV
    csv_file = project_root / "outputs" / "results" / "IP5_final_fair_ranking.csv"
    if not csv_file.exists():
        print(f"❌ CSV not found: {csv_file}")
        return
    
    df_ranked = pd.read_csv(csv_file)
    print(f"✅ CSV loaded: {len(df_ranked)} projects")
    
    # Load the vectors
    vector_file = project_root / "IP1_output_listings_vectorized.pkl"
    if vector_file.exists():
        with open(vector_file, 'rb') as f:
            vectors_data = pickle.load(f)
        
        if isinstance(vectors_data, dict):
            df_ranked['skill_vectors'] = df_ranked['link'].apply(
                lambda x: vectors_data.get(str(x)) if pd.notna(x) else None
            )
        elif isinstance(vectors_data, pd.DataFrame):
            if 'link' in vectors_data.columns and 'skill_vectors' in vectors_data.columns:
                # Convert the links to strings when merging
                vectors_df = vectors_data[['link', 'skill_vectors']].copy()
                vectors_df['link'] = vectors_df['link'].astype(str)
                df_ranked['link'] = df_ranked['link'].astype(str)
                df_ranked = df_ranked.merge(
                    vectors_df,
                    on='link',
                    how='left'
                )
    
    ranking_calc = FinalRankingCalculator(config, logger)
    
    # Baseline metrikleri
    print("\n📊 BASELINE (BEFORE MMR)")
    print("=" * 80)
    relevance_b, diversity_b, conf_b, rc_b, uc_b = ranking_calc.calculate_performance_metrics(
        df_ranked, top_k=5, vector_column="skill_vectors", m_score_column="M_Score"
    )
    print(f"Diversity: {diversity_b:.6f}")
    print(f"Relevance: {relevance_b:.2f}")
    
    # Show the top-5 projects
    print("\nTop-5 projects (baseline):")
    for i, (idx, row) in enumerate(df_ranked.head(5).iterrows(), 1):
        print(f"  {i}. Fair_Score={row.get('Fair_Score', 0):.2f}, Link={row.get('link', 'N/A')[:50]}...")
    
    # MMR with lambda=0.0 (diversity only)
    print("\n🔄 MMR (lambda=0.0 - diversity only)")
    print("=" * 80)
    df_mmr = ranking_calc._apply_mmr_ranking(
        df_ranked.copy(),
        top_k=10,
        lambda_param=0.0,  # Diversity only
        vector_column="skill_vectors"
    )
    
    # Metrics after MMR
    print("\n📊 AFTER MMR (lambda=0.0)")
    print("=" * 80)
    relevance_a, diversity_a, conf_a, rc_a, uc_a = ranking_calc.calculate_performance_metrics(
        df_mmr, top_k=5, vector_column="skill_vectors", m_score_column="M_Score"
    )
    print(f"Diversity: {diversity_a:.6f} (change: {diversity_a - diversity_b:+.6f})")
    print(f"Relevance: {relevance_a:.2f} (change: {relevance_a - relevance_b:+.2f})")
    
    # Show the top-5 projects
    print("\nTop-5 projects (after MMR):")
    for i, (idx, row) in enumerate(df_mmr.head(5).iterrows(), 1):
        print(f"  {i}. Fair_Score={row.get('Fair_Score', 0):.2f}, Link={row.get('link', 'N/A')[:50]}...")
    
    # Check whether the projects changed
    baseline_links = [str(row.get('link', '')) for _, row in df_ranked.head(5).iterrows()]
    mmr_links = [str(row.get('link', '')) for _, row in df_mmr.head(5).iterrows()]
    
    changed = sum(1 for b, m in zip(baseline_links, mmr_links) if b != m)
    print(f"\n📊 Change: {changed}/5 project positions changed")
    
    if diversity_a < diversity_b:
        print("\n⚠️  ISSUE: MMR DECREASED the Diversity Score!")
        print("   This is an unexpected outcome. MMR should increase diversity.")
    elif diversity_a > diversity_b:
        print("\n✅ MMR increased the Diversity Score!")
    else:
        print("\n⚠️  MMR did not change the Diversity Score")

if __name__ == "__main__":
    debug_mmr()
