#!/usr/bin/env python3
"""
Diversity Score debug script'i
Analyzes why the Diversity Score stays at 0.0
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

def debug_diversity():
    """Diversity Score'u debug et"""
    print("=" * 80)
    print("🔍 DIVERSITY SCORE DEBUG")
    print("=" * 80)
    
    # Load the CSV
    csv_file = project_root / "outputs" / "results" / "IP5_final_fair_ranking.csv"
    if not csv_file.exists():
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    df_ranked = pd.read_csv(csv_file)
    print(f"✅ CSV loaded: {len(df_ranked)} projects")
    
    # Load the vectors
    vector_files = [
        project_root / "data/processed/IP1_output_listings_vectorized.pkl",
        project_root / "outputs" / "processed" / "IP1_output_listings_vectorized.pkl",
    ]
    
    vectors_loaded = False
    for vector_file in vector_files:
        if vector_file.exists():
            print(f"\n📂 Loading vectors: {vector_file}")
            try:
                with open(vector_file, 'rb') as f:
                    vectors_data = pickle.load(f)
                
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
                    print(f"✅ Vectors loaded: {vectors_loaded_count}/{len(df_ranked)}")
                    vectors_loaded = True
                    break
                elif isinstance(vectors_data, pd.DataFrame):
                    if 'link' in vectors_data.columns and 'skill_vectors' in vectors_data.columns:
                        df_ranked = df_ranked.merge(
                            vectors_data[['link', 'skill_vectors']],
                            on='link',
                            how='left'
                        )
                        vectors_loaded_count = df_ranked['skill_vectors'].notna().sum()
                        print(f"✅ Vectors loaded: {vectors_loaded_count}/{len(df_ranked)}")
                        vectors_loaded = True
                        break
            except Exception as e:
                print(f"⚠️  Error: {e}")
                continue
    
    if not vectors_loaded or 'skill_vectors' not in df_ranked.columns:
        print("❌ Could not load the vectors!")
        return
    
    # Analyze the top-5 projects
    top5 = df_ranked.head(5)
    print(f"\n📊 TOP-5 PROJECT ANALYSIS")
    print("=" * 80)
    
    vectors_list = []
    valid_links = []
    
    for idx, row in top5.iterrows():
        link = row.get('link', 'N/A')
        vec = row.get('skill_vectors')
        
        print(f"\nProject {idx}:")
        print(f"  Link: {link[:60]}...")
        print(f"  Fair_Score: {row.get('Fair_Score', 0):.2f}")
        print(f"  Vector: {type(vec)}")
        
        if vec is None:
            print("  ⚠️  NO vector!")
            continue
        
        try:
            vec_array = np.array(vec)
            print(f"  Vector shape: {vec_array.shape}")
            print(f"  Vector dtype: {vec_array.dtype}")
            print(f"  Vector min/max: {vec_array.min():.4f} / {vec_array.max():.4f}")
            print(f"  Vector norm: {np.linalg.norm(vec_array):.4f}")
            
            if vec_array.ndim > 1:
                vec_array = vec_array.flatten()
                print(f"  Flattened shape: {vec_array.shape}")
            
            norm = np.linalg.norm(vec_array)
            if norm > 0:
                vec_normalized = vec_array / norm
                vectors_list.append(vec_normalized)
                valid_links.append(link)
                print(f"  ✅ Normalized (norm={norm:.4f})")
            else:
                print(f"  ⚠️  Zero norm!")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            print(traceback.format_exc()[:200])
    
    print(f"\n📊 VECTOR SIMILARITY ANALYSIS")
    print("=" * 80)
    print(f"Number of valid vectors: {len(vectors_list)}")
    
    if len(vectors_list) < 2:
        print("❌ Not enough vectors (at least 2 needed)")
        return
    
    # Compute cosine similarity
    try:
        vectors_matrix = np.vstack(vectors_list)
        print(f"Vector matrix shape: {vectors_matrix.shape}")
        
        similarity_matrix = cosine_similarity(vectors_matrix)
        print(f"Similarity matrix shape: {similarity_matrix.shape}")
        
        # Set the diagonal to 0
        np.fill_diagonal(similarity_matrix, 0.0)
        
        # Upper triangular matrix
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        similarities = similarity_matrix[mask]
        
        if len(similarities) > 0:
            avg_sim = similarities.mean()
            min_sim = similarities.min()
            max_sim = similarities.max()
            std_sim = similarities.std()
            
            print(f"\nSimilarity statistics:")
            print(f"  Mean: {avg_sim:.6f}")
            print(f"  Min: {min_sim:.6f}")
            print(f"  Max: {max_sim:.6f}")
            print(f"  Std: {std_sim:.6f}")
            
            diversity = 1.0 - avg_sim
            print(f"\nDiversity Score: {diversity:.6f}")
            
            if avg_sim > 0.999:
                print("\n⚠️  VERY HIGH SIMILARITY (>0.999)!")
                print("   The projects have almost identical vectors")
                print("   This shows that the projects are really very similar")
            elif avg_sim > 0.99:
                print("\n⚠️  High similarity (>0.99)")
            elif avg_sim > 0.95:
                print("\n⚠️  Similarity is high (>0.95)")
            else:
                print("\n✅ Normal similarity level")
            
            # Show the pairwise similarities
            print(f"\nPairwise similarities:")
            for i in range(len(vectors_list)):
                for j in range(i+1, len(vectors_list)):
                    sim = similarity_matrix[i, j]
                    print(f"  {i} <-> {j}: {sim:.6f} (Link {i}: {valid_links[i][:30]}... <-> Link {j}: {valid_links[j][:30]}...)")
        else:
            print("❌ Could not compute the similarity")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_diversity()
