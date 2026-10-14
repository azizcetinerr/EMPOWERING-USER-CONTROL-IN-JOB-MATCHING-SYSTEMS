#!/usr/bin/env python3
"""
Script for fixing critical issues
1. Diversity: vectors are too similar → also use the project descriptions (not only the skills)
2. Confidence: M scores in the top-5 are too close → fix the Z-score normalization
3. RC: only the top-5 are recommended → increase the Top-K value
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config

def analyze_skill_similarity(df):
    """Analyze skill similarity"""
    print("\n📊 SKILL SIMILARITY ANALYSIS:")
    
    if 'extracted_skills' not in df.columns:
        print("   ❌ Skill column not found")
        return
    
    # Get the skills of the top-5 projects
    top5 = df.head(5)
    skill_sets = []
    
    for idx, row in top5.iterrows():
        skills = row['extracted_skills']
        if isinstance(skills, list):
            skill_sets.append(set(s.lower().strip() for s in skills if s))
        else:
            skill_sets.append(set())
    
    # Jaccard similarity
    if len(skill_sets) >= 2:
        similarities = []
        for i in range(len(skill_sets)):
            for j in range(i+1, len(skill_sets)):
                set1, set2 = skill_sets[i], skill_sets[j]
                if len(set1) == 0 and len(set2) == 0:
                    sim = 1.0
                elif len(set1) == 0 or len(set2) == 0:
                    sim = 0.0
                else:
                    intersection = len(set1.intersection(set2))
                    union = len(set1.union(set2))
                    sim = intersection / union if union > 0 else 0.0
                similarities.append(sim)
        
        if similarities:
            print(f"   Jaccard similarity (skill sets):")
            print(f"     Mean: {np.mean(similarities):.4f}")
            print(f"     Min: {np.min(similarities):.4f}")
            print(f"     Max: {np.max(similarities):.4f}")
            
            if np.mean(similarities) > 0.5:
                print(f"     ⚠️  Skill similarity above 50%!")
            
            # Common skills
            all_skills = set()
            for s in skill_sets:
                all_skills.update(s)
            
            common_skills = set.intersection(*skill_sets) if skill_sets else set()
            print(f"     Total unique skills: {len(all_skills)}")
            print(f"     Skills common to all projects: {len(common_skills)}")
            if common_skills:
                print(f"     Common skills: {', '.join(list(common_skills)[:10])}")

def main():
    """Main analysis"""
    print("=" * 80)
    print("🔍 DETAILED ANALYSIS OF CRITICAL ISSUES")
    print("=" * 80)
    
    # Load the CSV
    csv_file = project_root / "outputs" / "results" / "IP5_final_fair_ranking.csv"
    if not csv_file.exists():
        print(f"\n❌ CSV not found: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    print(f"\n✅ CSV loaded: {len(df)} projects")
    
    # Run the analysis
    analyze_skill_similarity(df)

if __name__ == "__main__":
    main()
