#!/usr/bin/env python3
"""
Kritik sorunları düzeltme script'i
1. Diversity: Vektörler çok benzer → Proje açıklamalarını da kullan (sadece beceriler değil)
2. Confidence: Top-5'te M skorları çok yakın → Z-score normalizasyonunu düzelt
3. RC: Sadece top-5 öneriliyor → Top-K değerini artır
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config

def analyze_skill_similarity(df):
    """Beceri benzerliğini analiz et"""
    print("\n📊 BECERİ BENZERLİĞİ ANALİZİ:")
    
    if 'cikarilan_beceriler' not in df.columns:
        print("   ❌ Beceri sütunu bulunamadı")
        return
    
    # Top-5 projelerin becerilerini al
    top5 = df.head(5)
    skill_sets = []
    
    for idx, row in top5.iterrows():
        skills = row['cikarilan_beceriler']
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
            print(f"   Jaccard similarity (beceri setleri):")
            print(f"     Ortalama: {np.mean(similarities):.4f}")
            print(f"     Min: {np.min(similarities):.4f}")
            print(f"     Max: {np.max(similarities):.4f}")
            
            if np.mean(similarities) > 0.5:
                print(f"     ⚠️  %50'den fazla beceri benzerliği var!")
            
            # Ortak beceriler
            all_skills = set()
            for s in skill_sets:
                all_skills.update(s)
            
            common_skills = set.intersection(*skill_sets) if skill_sets else set()
            print(f"     Toplam benzersiz beceri: {len(all_skills)}")
            print(f"     Tüm projelerde ortak beceriler: {len(common_skills)}")
            if common_skills:
                print(f"     Ortak beceriler: {', '.join(list(common_skills)[:10])}")

def main():
    """Ana analiz"""
    print("=" * 80)
    print("🔍 KRİTİK SORUNLAR İÇİN DETAYLI ANALİZ")
    print("=" * 80)
    
    # CSV yükle
    csv_file = project_root / "outputs" / "results" / "IP5_Nihai_Adil_Siralama.csv"
    if not csv_file.exists():
        print(f"\n❌ CSV bulunamadı: {csv_file}")
        return
    
    df = pd.read_csv(csv_file)
    print(f"\n✅ CSV yüklendi: {len(df)} proje")
    
    # Analiz yap
    analyze_skill_similarity(df)

if __name__ == "__main__":
    main()
