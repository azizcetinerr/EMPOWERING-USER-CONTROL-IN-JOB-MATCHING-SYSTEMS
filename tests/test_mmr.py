#!/usr/bin/env python3
"""
MMR (Maximal Marginal Relevance) test script'i
Loads the existing results and tests whether MMR works
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Find the project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.scoring.final_ranking import FinalRankingCalculator

def test_mmr():
    """Test MMR"""
    print("=" * 80)
    print("🔍 MMR (Maximal Marginal Relevance) TEST")
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
        console=logger_config.get('console', True),
        log_format=logger_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Check the MMR settings
    mmr_config = config.get('diversity_aware_ranking', {})
    mmr_enabled = mmr_config.get('enabled', False)
    mmr_top_k = mmr_config.get('top_k', 10)
    mmr_lambda = mmr_config.get('lambda', 0.5)
    
    print(f"\n📋 MMR settings:")
    print(f"  enabled: {mmr_enabled}")
    print(f"  top_k: {mmr_top_k}")
    print(f"  lambda: {mmr_lambda}")
    
    if not mmr_enabled:
        print("\n⚠️  MMR disabled! Set diversity_aware_ranking.enabled: true in config.yaml.")
        return
    
    # Load the existing results
    csv_file = project_root / "outputs" / "results" / "IP5_final_fair_ranking.csv"
    if not csv_file.exists():
        print(f"\n❌ Result file not found: {csv_file}")
        return
    
    print(f"\n📂 Loading results: {csv_file}")
    df_ranked = pd.read_csv(csv_file)
    print(f"✅ {len(df_ranked)} projects loaded")
    
    # Load the vectors (search several possible locations)
    vector_files = [
        project_root / "data/processed/IP1_output_listings_vectorized.pkl",  # In the project root (most common)
        project_root / "outputs" / "processed" / "IP1_output_listings_vectorized.pkl",
    ]
    
    vectors_loaded = False
    for vector_file in vector_files:
        if vector_file.exists():
            print(f"\n📂 Loading vectors: {vector_file}")
            try:
                with open(vector_file, 'rb') as f:
                    vectors_data = pickle.load(f)
                
                print(f"   Vector data type: {type(vectors_data).__name__}")
                if isinstance(vectors_data, pd.DataFrame):
                    print(f"   DataFrame row count: {len(vectors_data)}")
                    if 'link' in vectors_data.columns:
                        print(f"   DataFrame columns: {list(vectors_data.columns)}")
                elif isinstance(vectors_data, dict):
                    print(f"   Dict key count: {len(vectors_data)}")
                    if len(vectors_data) > 0:
                        print(f"   First key sample: {list(vectors_data.keys())[0]}")
                elif isinstance(vectors_data, list):
                    print(f"   List length: {len(vectors_data)}")
                
                # Add the vectors to the DataFrame
                if 'link' not in df_ranked.columns:
                    print("⚠️  'link' column not found, vectors could not be added")
                    continue
                
                if isinstance(vectors_data, pd.DataFrame):
                    # If it is a DataFrame, merge on 'link'
                    if 'link' in df_ranked.columns and 'link' in vectors_data.columns:
                        # First drop the existing 'skill_vectors' column (if any)
                        if 'skill_vectors' in df_ranked.columns:
                            df_ranked = df_ranked.drop(columns=['skill_vectors'])
                        
                        # Merge et
                        df_ranked = df_ranked.merge(
                            vectors_data[['link', 'skill_vectors']],
                            on='link',
                            how='left'
                        )
                        vectors_loaded_count = df_ranked['skill_vectors'].notna().sum()
                        print(f"✅ Vectors merged (DataFrame): {vectors_loaded_count}/{len(df_ranked)} projects have vectors")
                        vectors_loaded = True
                        break
                    elif 'skill_vectors' in vectors_data.columns and len(vectors_data) == len(df_ranked):
                        # Add by index (if the counts match)
                        df_ranked['skill_vectors'] = vectors_data['skill_vectors'].values
                        vectors_loaded_count = df_ranked['skill_vectors'].notna().sum()
                        print(f"✅ Vectors added by index (DataFrame): {vectors_loaded_count}/{len(df_ranked)} projects have vectors")
                        vectors_loaded = True
                        break
                    else:
                        print(f"⚠️  DataFrame format does not match the expected one (link: {'link' in vectors_data.columns}, skill_vectors: {'skill_vectors' in vectors_data.columns}, length: {len(vectors_data)} vs {len(df_ranked)})")
                elif isinstance(vectors_data, dict):
                    # If it is a dict, map by link (for string keys)
                    # Try both the string and the original format
                    def get_vector(link_val):
                        if pd.isna(link_val):
                            return None
                        link_str = str(link_val)
                        # First try the string key directly
                        if link_str in vectors_data:
                            return vectors_data[link_str]
                        # Then convert all keys to strings and try
                        for k, v in vectors_data.items():
                            if str(k) == link_str:
                                return v
                        return None
                    
                    df_ranked['skill_vectors'] = df_ranked['link'].apply(get_vector)
                    vectors_loaded_count = df_ranked['skill_vectors'].notna().sum()
                    print(f"✅ Vectors mapped (dict): {vectors_loaded_count}/{len(df_ranked)} projects have vectors")
                    vectors_loaded = True
                    break
                elif isinstance(vectors_data, list):
                    # If it is a list, add by index (only if the counts match)
                    if len(vectors_data) == len(df_ranked):
                        df_ranked['skill_vectors'] = vectors_data[:len(df_ranked)]
                        vectors_loaded_count = len([v for v in df_ranked['skill_vectors'] if v is not None])
                        print(f"✅ Vectors added as a list: {len(vectors_data)} vectors")
                        vectors_loaded = True
                        break
                    else:
                        print(f"⚠️  Vector list length ({len(vectors_data)}) does not match the number of projects ({len(df_ranked)}) - link-based matching required")
                else:
                    print(f"⚠️  Unexpected vector data type: {type(vectors_data)}")
                    print(f"    Content sample: {str(vectors_data)[:200]}")
            except Exception as e:
                import traceback
                print(f"⚠️  Could not read the vector file: {e}")
                print(f"    Detail: {traceback.format_exc()[:300]}")
                continue
    
    if not vectors_loaded:
        print(f"\n⚠️  Vector file not found or could not be loaded!")
        print(f"   Files checked:")
        for vf in vector_files:
            print(f"     - {vf} ({'✓ EXISTS' if vf.exists() else '✗ MISSING'})")
        print(f"\n   Vectors are required for the diversity computation and MMR!")
        return
    
    # Create the FinalRankingCalculator
    ranking_calc = FinalRankingCalculator(config, logger)
    
    # Compute diversity before MMR
    print("\n" + "=" * 80)
    print("📊 METRICS BEFORE MMR")
    print("=" * 80)
    
    relevance_before, diversity_before, confidence_before, rc_before, uc_before = ranking_calc.calculate_performance_metrics(
        df_ranked,
        top_k=5,
        vector_column="skill_vectors",
        m_score_column="M_Score"
    )
    
    print(f"  Relevance: {relevance_before:.4f}")
    print(f"  Diversity: {diversity_before:.4f}")
    print(f"  Confidence: {confidence_before:.4f}")
    print(f"  RC: {rc_before:.4f}")
    print(f"  UC: {uc_before:.4f}")
    
    # Apply MMR
    print("\n" + "=" * 80)
    print("🔄 APPLYING MMR")
    print("=" * 80)
    
    if 'skill_vectors' not in df_ranked.columns:
        print("❌ No 'skill_vectors' column! MMR cannot run.")
        return
    
    df_mmr = ranking_calc._apply_mmr_ranking(
        df_ranked.copy(),
        top_k=mmr_top_k,
        lambda_param=mmr_lambda,
        vector_column="skill_vectors"
    )
    
    # Compute diversity after MMR
    print("\n" + "=" * 80)
    print("📊 METRICS AFTER MMR")
    print("=" * 80)
    
    relevance_after, diversity_after, confidence_after, rc_after, uc_after = ranking_calc.calculate_performance_metrics(
        df_mmr,
        top_k=5,
        vector_column="skill_vectors",
        m_score_column="M_Score"
    )
    
    print(f"  Relevance: {relevance_after:.4f}")
    print(f"  Diversity: {diversity_after:.4f}  ({diversity_after - diversity_before:+.4f})")
    print(f"  Confidence: {confidence_after:.4f} ({confidence_after - confidence_before:+.4f})")
    print(f"  RC: {rc_after:.4f}")
    print(f"  UC: {uc_after:.4f}")
    
    # Result summary
    print("\n" + "=" * 80)
    print("📈 RESULT SUMMARY")
    print("=" * 80)
    print(f"Diversity change: {diversity_before:.4f} → {diversity_after:.4f} ({diversity_after - diversity_before:+.4f})")
    
    if diversity_after > diversity_before:
        print("✅ MMR increased the Diversity Score!")
    elif diversity_after == diversity_before:
        print("⚠️  MMR did not change the Diversity Score (the projects are already different or very similar)")
    else:
        print("⚠️  MMR decreased the Diversity Score (unexpected outcome)")
    
    # Compare the top-5 projects
    print("\n" + "=" * 80)
    print("🔍 TOP-5 PROJECT COMPARISON")
    print("=" * 80)
    
    print("\nBefore MMR (first 5):")
    before_links = []
    for i, (idx, row) in enumerate(df_ranked.head(5).iterrows(), 1):
        link = row.get('link', 'N/A')
        before_links.append(link)
        print(f"  {i}. Fair_Score={row.get('Fair_Score', 0):.2f}, "
              f"M_Score={row.get('M_Score', 0):.2f}, "
              f"Link={str(link)[:60]}...")
    
    print("\nAfter MMR (first 5):")
    after_links = []
    for i, (idx, row) in enumerate(df_mmr.head(5).iterrows(), 1):
        link = row.get('link', 'N/A')
        after_links.append(link)
        print(f"  {i}. Fair_Score={row.get('Fair_Score', 0):.2f}, "
              f"M_Score={row.get('M_Score', 0):.2f}, "
              f"Link={str(link)[:60]}...")
    
    # Check whether the projects changed
    changed_count = sum(1 for b, a in zip(before_links[:5], after_links[:5]) if b != a)
    print(f"\n📊 Ranking change: {changed_count}/5 project positions changed")
    
    # Analyze vector similarity
    if 'skill_vectors' in df_mmr.columns:
        print("\n" + "=" * 80)
        print("🔬 VECTOR SIMILARITY ANALYSIS (Top-5)")
        print("=" * 80)
        
        top5_mmr = df_mmr.head(5)
        vectors_top5 = []
        valid_indices = []
        
        for idx, row in top5_mmr.iterrows():
            vec = row.get('skill_vectors')
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
                print(f"  Mean similarity: {avg_sim:.4f}")
                print(f"  Min similarity: {min_sim:.4f}")
                print(f"  Max similarity: {max_sim:.4f}")
                print(f"  Diversity Score (1 - avg): {1.0 - avg_sim:.4f}")
                
                if avg_sim > 0.95:
                    print(f"\n⚠️  The top-5 projects are VERY SIMILAR (similarity > 0.95)!")
                    print(f"   Try lowering the lambda value (e.g. 0.3 or 0.4)")
                elif avg_sim > 0.9:
                    print(f"\n⚠️  The top-5 projects are similar (similarity > 0.9)")
                    print(f"   Try lowering the lambda value (e.g. 0.4)")
    
    print("\n💡 RECOMMENDATIONS:")
    if diversity_after == 0.0:
        print("  1. Lower the lambda value (config.yaml): lambda: 0.3 or 0.4")
        print("  2. This gives more weight to diversity (instead of relevance)")
        print("  3. Rerun the pipeline or rerun the test script with lambda")
    
    print("\n✅ Test completed!")

if __name__ == "__main__":
    test_mmr()
