#!/usr/bin/env python3
"""
Script for diagnosing critical issues
1. Why is Diversity 0.0028? (duplicate content check)
2. Why is Confidence 0.04? (why are the M scores identical?)
3. Why is RC 0.0053? (why are only 5 projects recommended?)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Find the project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.scoring.final_ranking import FinalRankingCalculator

def analyze_diversity_issue(df_ranked):
    """Why is Diversity 0.0028? Duplicate content check"""
    print("\n" + "=" * 80)
    print("🔍 ISSUE 1: DIVERSITY = 0.0028 ANALYSIS")
    print("=" * 80)
    
    # Get the top-5 projects
    top5 = df_ranked.head(5).copy()
    
    print(f"\nTop-5 projects:")
    for i, (idx, row) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. Link: {str(row.get('link', ''))[:60]}...")
        if 'full_description' in row:
            desc = str(row['full_description'])[:100] if pd.notna(row['full_description']) else "N/A"
            print(f"     Description: {desc}...")
        if 'extracted_skills' in row:
            skills = row['extracted_skills']
            if isinstance(skills, list):
                print(f"     Skills ({len(skills)}): {', '.join(skills[:10])}")
            else:
                print(f"     Skills: {skills}")
    
    # 1. Duplicate content check
    print("\n📊 1. DUPLICATE CONTENT CHECK:")
    if 'full_description' in top5.columns:
        descriptions = top5['full_description'].fillna('').astype(str)
        unique_descriptions = descriptions.nunique()
        print(f"   Number of unique descriptions: {unique_descriptions}/5")
        if unique_descriptions < 5:
            print(f"   ⚠️  {5 - unique_descriptions} projects have the same description!")
            # Find identical descriptions
            desc_counts = descriptions.value_counts()
            for desc, count in desc_counts.items():
                if count > 1:
                    print(f"   ⚠️  Description '{desc[:50]}...' is repeated {count} times")
    
    # 2. Generic words check
    print("\n📊 2. GENERIC WORDS CHECK:")
    if 'extracted_skills' in top5.columns:
        all_skills = []
        for skills in top5['extracted_skills']:
            if isinstance(skills, list):
                all_skills.extend(skills)
        
        if all_skills:
            skill_counts = Counter(all_skills)
            print(f"   Total number of skills: {len(all_skills)}")
            print(f"   Number of unique skills: {len(skill_counts)}")
            print(f"   Most frequent skills:")
            for skill, count in skill_counts.most_common(15):
                print(f"     - {skill}: {count} times ({count*100/len(all_skills):.1f}%)")
            
            # Generic words
            generic_skills = ['communication', 'teamwork', 'collaboration', 'leadership', 
                            'problem solving', 'time management', 'organization']
            generic_count = sum(skill_counts.get(skill.lower(), 0) for skill in generic_skills)
            print(f"\n   Total generic skills: {generic_count}/{len(all_skills)} ({generic_count*100/len(all_skills):.1f}%)")
            if generic_count > len(all_skills) * 0.5:
                print(f"   ⚠️  More than 50% of the skills are generic!")
    
    # 3. Vector similarity check
    print("\n📊 3. VECTOR SIMILARITY CHECK:")
    if 'skill_vectors' in top5.columns:
        vectors_list = []
        valid_indices = []
        
        for idx, row in top5.iterrows():
            vector = row['skill_vectors']
            if vector is None:
                continue
            
            try:
                if isinstance(vector, float) and np.isnan(vector):
                    continue
                
                vec_array = np.array(vector)
                if vec_array.ndim == 2:
                    vec_array = vec_array.mean(axis=0)
                elif vec_array.ndim > 2:
                    vec_array = vec_array.flatten()
                
                if vec_array.ndim == 1:
                    norm = np.linalg.norm(vec_array)
                    if norm > 1e-8:
                        vectors_list.append(vec_array / norm)
                        valid_indices.append(idx)
            except:
                continue
        
        if len(vectors_list) >= 2:
            vectors_matrix = np.vstack(vectors_list)
            similarity_matrix = cosine_similarity(vectors_matrix)
            np.fill_diagonal(similarity_matrix, 0.0)
            
            mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
            similarities = similarity_matrix[mask]
            
            print(f"   Number of processable vectors: {len(vectors_list)}/5")
            print(f"   Mean similarity: {similarities.mean():.6f}")
            print(f"   Min similarity: {similarities.min():.6f}")
            print(f"   Max similarity: {similarities.max():.6f}")
            print(f"   Std deviation: {similarities.std():.6f}")
            
            if similarities.mean() > 0.95:
                print(f"   ⚠️  Similarity above 95%! The vectors are almost identical!")
            elif similarities.mean() > 0.80:
                print(f"   ⚠️  Similarity above 80%! The vectors are very similar!")
        else:
            print(f"   ❌ Not enough vectors: {len(vectors_list)}/5")

def analyze_confidence_issue(df_ranked):
    """Why is Confidence 0.04? Why are the M scores identical?"""
    print("\n" + "=" * 80)
    print("🔍 ISSUE 2: CONFIDENCE = 0.04 ANALYSIS (M SCORES)")
    print("=" * 80)
    
    # Check the M scores
    m_score_cols = ['M_Score', 'M_Score_Raw', 'M_Score_Normalized']
    available_cols = [col for col in m_score_cols if col in df_ranked.columns]
    
    if not available_cols:
        print("   ❌ M_Score column not found!")
        return
    
    for col in available_cols:
        scores = df_ranked[col].dropna()
        if len(scores) == 0:
            continue
        
        print(f"\n📊 {col} statistics:")
        print(f"   Total projects: {len(scores)}")
        print(f"   Min: {scores.min():.4f}")
        print(f"   Max: {scores.max():.4f}")
        print(f"   Mean: {scores.mean():.4f}")
        print(f"   Medyan: {scores.median():.4f}")
        print(f"   Std deviation: {scores.std():.4f}")
        print(f"   Number of unique values: {scores.nunique()}")
        
        if scores.std() < 0.1:
            print(f"   ⚠️  Standard deviation is very low! The scores are almost identical!")
        
        # Show the first 10 scores
        print(f"   First 10 scores: {scores.head(10).tolist()}")
        
        # Number of projects with the same score
        score_counts = scores.value_counts()
        duplicate_scores = score_counts[score_counts > 1]
        if len(duplicate_scores) > 0:
            print(f"   ⚠️  {len(duplicate_scores)} distinct scores are repeated across {duplicate_scores.sum()} projects")
            print(f"   Most repeated scores:")
            for score, count in duplicate_scores.head(5).items():
                print(f"     - {score:.4f}: {count} projects")
    
    # Show the top-5 M scores in detail
    print(f"\n📊 Top-5 M score details:")
    top5 = df_ranked.head(5)
    for i, (idx, row) in enumerate(top5.iterrows(), 1):
        print(f"   {i}. ", end="")
        for col in available_cols:
            val = row.get(col, None)
            if pd.notna(val):
                print(f"{col}={val:.4f} ", end="")
        print()

def analyze_rc_issue(df_ranked):
    """Why is RC 0.0053? Why are only 5 projects recommended?"""
    print("\n" + "=" * 80)
    print("🔍 ISSUE 3: RC = 0.0053 ANALYSIS (RECOMMENDATION COVERAGE)")
    print("=" * 80)
    
    total_projects = len(df_ranked)
    print(f"\n📊 General statistics:")
    print(f"   Total number of projects: {total_projects}")
    
    # Top-K check
    print(f"\n📊 Top-K analysis:")
    for k in [5, 10, 20, 50, 100]:
        top_k = df_ranked.head(k)
        rc = k / total_projects
        print(f"   Top-{k}: RC = {rc:.4f} ({k}/{total_projects})")
        
        # How many of this top-K have a sufficient score?
        if 'Fair_Score' in df_ranked.columns:
            threshold = 40.0
            eligible_in_topk = len(top_k[top_k['Fair_Score'] > threshold])
            eligible_total = len(df_ranked[df_ranked['Fair_Score'] > threshold])
            print(f"      With a sufficient score (Fair_Score > {threshold}): {eligible_in_topk}/{eligible_total} projects")
    
    # Fair_Score distribution
    if 'Fair_Score' in df_ranked.columns:
        print(f"\n📊 Fair_Score distribution:")
        scores = df_ranked['Fair_Score'].dropna()
        print(f"   Total projects: {len(scores)}")
        print(f"   Min: {scores.min():.2f}")
        print(f"   Max: {scores.max():.2f}")
        print(f"   Mean: {scores.mean():.2f}")
        print(f"   Medyan: {scores.median():.2f}")
        print(f"   Std deviation: {scores.std():.2f}")
        
        # Projects above the threshold
        thresholds = [30, 40, 50, 60, 70]
        for thresh in thresholds:
            count = len(scores[scores > thresh])
            pct = count * 100 / len(scores)
            print(f"   Fair_Score > {thresh}: {count} projects ({pct:.1f}%)")
    
    # Is the top-5 always the same projects?
    print(f"\n📊 Top-5 project stability:")
    top5_links = df_ranked.head(5)['link'].tolist()
    print(f"   Top-5 links:")
    for i, link in enumerate(top5_links, 1):
        print(f"     {i}. {str(link)[:60]}...")

def main():
    """Main analysis function"""
    print("=" * 80)
    print("🔍 CRITICAL ISSUES DIAGNOSTIC REPORT")
    print("=" * 80)
    
    # Load the config
    config_path = project_root / "config" / "config.yaml"
    config = load_config(str(config_path))
    
    # Logger
    logger_config = config.get('logging', {})
    logger = setup_logger(
        name=logger_config.get('name', 'diagnosis'),
        log_level='WARNING',
        console=False
    )
    
    # Load the CSV
    csv_file = project_root / "outputs" / "results" / "IP5_final_fair_ranking.csv"
    if not csv_file.exists():
        print(f"\n❌ CSV not found: {csv_file}")
        return
    
    df_ranked = pd.read_csv(csv_file)
    print(f"\n✅ CSV loaded: {len(df_ranked)} projects")
    
    # Load the vectors
    vector_file = project_root / "data/processed/IP1_output_listings_vectorized.pkl"
    if vector_file.exists():
        with open(vector_file, 'rb') as f:
            vectors_data = pickle.load(f)
        
        if isinstance(vectors_data, pd.DataFrame):
            vectors_df = vectors_data[['link', 'skill_vectors']].copy()
            vectors_df['link'] = vectors_df['link'].astype(str)
            df_ranked['link'] = df_ranked['link'].astype(str)
            df_ranked = df_ranked.merge(vectors_df, on='link', how='left')
            print(f"✅ Vectors loaded: {df_ranked['skill_vectors'].notna().sum()}/{len(df_ranked)} projects")
    
    # Run the analyses
    analyze_diversity_issue(df_ranked)
    analyze_confidence_issue(df_ranked)
    analyze_rc_issue(df_ranked)
    
    print("\n" + "=" * 80)
    print("✅ DIAGNOSIS COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    main()
