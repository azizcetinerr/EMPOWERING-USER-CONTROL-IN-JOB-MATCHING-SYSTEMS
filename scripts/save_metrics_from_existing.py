#!/usr/bin/env python3
"""
Script that computes and saves the metrics from existing IP5 results
"""
import sys
from pathlib import Path

# Add the project root
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import json

# Load the config and logger
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.scoring.final_ranking import FinalRankingCalculator

def load_existing_results():
    """Load the existing IP5 results"""
    print("📂 Loading existing results...")
    
    # Load the IP5 CSV
    csv_file = project_root / "outputs/results/IP5_final_fair_ranking.csv"
    if not csv_file.exists():
        print(f"❌ IP5 CSV file not found: {csv_file}")
        return None, None
    
    df_ranked = pd.read_csv(csv_file)
    print(f"✅ IP5 CSV loaded: {len(df_ranked)} projects")
    
    # Load the vectors (needed for diversity)
    vector_file = project_root / "data/processed/IP1_output_listings_vectorized.pkl"
    vectors_dict = None
    if vector_file.exists():
        try:
            with open(vector_file, 'rb') as f:
                vectors_dict = pickle.load(f)
            if isinstance(vectors_dict, pd.DataFrame):
                print(f"✅ Vectors loaded: {len(vectors_dict)} projects (DataFrame)")
            elif isinstance(vectors_dict, (dict, list)):
                print(f"✅ Vectors loaded: {len(vectors_dict)} projects ({type(vectors_dict).__name__})")
            else:
                print(f"✅ Vectors loaded: {type(vectors_dict).__name__}")
            
            # Add the vectors to the DataFrame
            if 'link' in df_ranked.columns:
                if isinstance(vectors_dict, dict):
                    # If it is a dict, map by link
                    df_ranked['skill_vectors'] = df_ranked['link'].map(lambda x: vectors_dict.get(str(x)))
                    missing_count = df_ranked['skill_vectors'].isna().sum()
                    if missing_count > 0:
                        print(f"⚠️  No vector found for {missing_count} projects")
                elif isinstance(vectors_dict, pd.DataFrame):
                    # If it is a DataFrame, merge by link
                    if 'link' in vectors_dict.columns and 'skill_vectors' in vectors_dict.columns:
                        df_ranked = df_ranked.merge(
                            vectors_dict[['link', 'skill_vectors']], 
                            on='link', 
                            how='left'
                        )
                        missing_count = df_ranked['skill_vectors'].isna().sum()
                        if missing_count > 0:
                            print(f"⚠️  No vector found for {missing_count} projects")
                    elif 'skill_vectors' in vectors_dict.columns and len(vectors_dict) == len(df_ranked):
                        # Add by index (if the counts match)
                        df_ranked['skill_vectors'] = vectors_dict['skill_vectors'].values
                    else:
                        print(f"⚠️  DataFrame format does not match the expected one")
                elif isinstance(vectors_dict, list):
                    # If it is a list, add by index (only if the counts match)
                    if len(vectors_dict) == len(df_ranked):
                        df_ranked['skill_vectors'] = vectors_dict[:len(df_ranked)]
                    else:
                        print(f"⚠️  Number of vectors ({len(vectors_dict)}) does not match the number of projects ({len(df_ranked)}) - link-based matching required")
                else:
                    print(f"⚠️  Unrecognized vector format: {type(vectors_dict)}")
                    print(f"    Content sample: {str(vectors_dict)[:200]}")
        except Exception as e:
            print(f"⚠️  Could not load the vectors (diversity may not be computable): {e}")
    else:
        print(f"⚠️  Vector file not found (diversity may not be computable): {vector_file}")
    
    return df_ranked, vectors_dict

def calculate_and_save_metrics():
    """Compute and save the metrics"""
    print("\n" + "="*60)
    print("📊 COMPUTING AND SAVING THE METRICS")
    print("="*60 + "\n")
    
    # Load the results
    df_ranked, vectors_dict = load_existing_results()
    if df_ranked is None:
        print("❌ Could not load the results, the metrics cannot be computed")
        return
    
    # Load the config and logger
    config = load_config()
    log_config = config.get('logging', {})
    logger = setup_logger(
        name=log_config.get('name', 'tubitak_project'),
        log_level=log_config.get('level', 'INFO'),
        log_file=log_config.get('file', None),
        console=log_config.get('console', True),
        log_format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    )
    
    # Create the FinalRankingCalculator
    ranking_calc = FinalRankingCalculator(config, logger)
    
    # Compute the metrics
    print("\n🔢 Computing metrics...")
    try:
        relevance, diversity, confidence, rc, uc = ranking_calc.calculate_performance_metrics(
            df_ranked,
            top_k=5,
            vector_column="skill_vectors",
            m_score_column="M_Score"
        )
        
        print(f"\n✅ Metrics computed:")
        print(f"   Relevance Accuracy: {relevance:.4f}")
        print(f"   Diversity Score: {diversity:.4f}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   RC (Recommendation Coverage): {rc:.4f}")
        print(f"   UC (User Coverage): {uc:.4f}")
        
        # Save the metrics
        print("\n💾 Metrikler kaydediliyor...")
        ranking_calc._save_metrics_report(relevance, diversity, confidence, rc, uc, df_ranked)
        ranking_calc._create_metrics_visualization(relevance, diversity, confidence, rc, uc)
        
        print("\n✅ Metrics saved successfully!")
        print(f"   📁 Directory: {project_root / 'outputs' / 'metrics'}")
        
    except Exception as e:
        print(f"\n❌ Metric computation error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    calculate_and_save_metrics()
