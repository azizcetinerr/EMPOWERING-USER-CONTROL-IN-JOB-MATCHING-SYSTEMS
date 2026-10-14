#!/usr/bin/env python3
"""
MMR lambda parameter test script
Tests MMR with different lambda values and finds the best Diversity Score
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Find the project root
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.scoring.final_ranking import FinalRankingCalculator

def test_different_lambdas():
    """Test MMR with different lambda values"""
    print("=" * 80)
    print("🔍 MMR LAMBDA PARAMETER TEST")
    print("=" * 80)
    
    # Load the config
    config_path = project_root / "config" / "config.yaml"
    config = load_config(str(config_path))
    
    # Set up the logger
    logger_config = config.get('logging', {})
    logger = setup_logger(
        name=logger_config.get('name', 'tubitak_project'),
        log_level=logger_config.get('level', 'INFO'),
        log_file=logger_config.get('file', 'logs/project.log'),
        console=False,  # Silent mode
        log_format=logger_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Check the MMR settings
    mmr_config = config.get('diversity_aware_ranking', {})
    mmr_top_k = mmr_config.get('top_k', 10)
    
    # Load the existing results
    csv_file = project_root / "outputs" / "results" / "IP5_final_fair_ranking.csv"
    if not csv_file.exists():
        print(f"\n❌ Result file not found: {csv_file}")
        return
    
    print(f"\n📂 Loading results: {csv_file}")
    df_ranked = pd.read_csv(csv_file)
    print(f"✅ {len(df_ranked)} projects loaded")
    
    # Load the vectors
    vector_files = [
        project_root / "IP1_output_listings_vectorized.pkl",
        project_root / "outputs" / "processed" / "IP1_output_listings_vectorized.pkl",
        project_root / "outputs" / "IP1_output_listings_vectorized.pkl",
    ]
    
    vectors_loaded = False
    for vector_file in vector_files:
        if vector_file.exists():
            print(f"\n📂 Loading vectors: {vector_file}")
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
                    
                    df_ranked['skill_vectors'] = df_ranked['link'].apply(get_vector)
                    vectors_loaded_count = df_ranked['skill_vectors'].notna().sum()
                    print(f"✅ Vectors loaded: {vectors_loaded_count}/{len(df_ranked)} projects have vectors")
                    vectors_loaded = True
                    break
                elif isinstance(vectors_data, pd.DataFrame):
                    if 'link' in vectors_data.columns and 'skill_vectors' in vectors_data.columns:
                        if 'skill_vectors' in df_ranked.columns:
                            df_ranked = df_ranked.drop(columns=['skill_vectors'])
                        # Convert the links to strings when merging (to avoid mismatches)
                        vectors_df = vectors_data[['link', 'skill_vectors']].copy()
                        vectors_df['link'] = vectors_df['link'].astype(str)
                        df_ranked['link'] = df_ranked['link'].astype(str)
                        df_ranked = df_ranked.merge(
                            vectors_df,
                            on='link',
                            how='left'
                        )
                        vectors_loaded_count = df_ranked['skill_vectors'].notna().sum()
                        print(f"✅ Vectors loaded: {vectors_loaded_count}/{len(df_ranked)} projects have vectors")
                        vectors_loaded = True
                        break
            except Exception as e:
                continue
    
    if not vectors_loaded or 'skill_vectors' not in df_ranked.columns:
        print("❌ Could not load the vectors!")
        return
    
    # Create the FinalRankingCalculator
    ranking_calc = FinalRankingCalculator(config, logger)
    
    # Test different lambda values (0.0 = diversity only)
    lambda_values = [0.9, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0]
    
    print("\n" + "=" * 80)
    print("📊 TESTING LAMBDA VALUES")
    print("=" * 80)
    print(f"{'Lambda':<10} {'Diversity':<12} {'Relevance':<12} {'Change':<12} {'Note'}")
    print("-" * 80)
    
    baseline_diversity = 0.0
    baseline_relevance = 0.0
    
    # First the baseline (without MMR)
    baseline_relevance, baseline_diversity, _, _, _ = ranking_calc.calculate_performance_metrics(
        df_ranked,
        top_k=5,
        vector_column="skill_vectors",
        m_score_column="M_Score"
    )
    
    print(f"{'Baseline':<10} {baseline_diversity:<12.4f} {baseline_relevance:<12.2f} {'-':<12} {'No MMR'}")
    
    best_lambda = None
    best_diversity = baseline_diversity
    
    results = []
    
    for lambda_val in lambda_values:
        # Apply MMR
        df_mmr = ranking_calc._apply_mmr_ranking(
            df_ranked.copy(),
            top_k=mmr_top_k,
            lambda_param=lambda_val,
            vector_column="skill_vectors"
        )
        
        # Compute the metrics
        relevance, diversity, _, _, _ = ranking_calc.calculate_performance_metrics(
            df_mmr,
            top_k=5,
            vector_column="skill_vectors",
            m_score_column="M_Score"
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
            note = "⭐ BEST"
        elif diversity == baseline_diversity:
            note = "⚠️  Unchanged"
        
        print(f"{lambda_val:<10.1f} {diversity:<12.4f} {relevance:<12.2f} {diversity_change:+.4f}      {note}")
    
    # Summary
    print("\n" + "=" * 80)
    print("📈 RESULT SUMMARY")
    print("=" * 80)
    
    if best_lambda is not None and best_diversity > baseline_diversity:
        print(f"✅ Best lambda: {best_lambda}")
        print(f"   Diversity Score: {baseline_diversity:.4f} → {best_diversity:.4f} ({best_diversity - baseline_diversity:+.4f})")
        print(f"\n💡 Recommended config setting:")
        print(f"   diversity_aware_ranking:")
        print(f"     enabled: true")
        print(f"     top_k: {mmr_top_k}")
        print(f"     lambda: {best_lambda}  # For the best Diversity Score")
    else:
        print("⚠️  No lambda value increased the Diversity Score")
        print("   This shows that the top-K projects really are very similar")
        print("   Try lowering lambda further (0.1, 0.05, 0.0) or increasing top_k")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_different_lambdas()
