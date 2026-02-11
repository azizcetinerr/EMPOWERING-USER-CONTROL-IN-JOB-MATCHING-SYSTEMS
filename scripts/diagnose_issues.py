#!/usr/bin/env python3
"""
Kritik sorunları teşhis etme script'i
1. Diversity neden 0.0028? (duplicate content kontrolü)
2. Confidence neden 0.04? (M skorları neden aynı?)
3. RC neden 0.0053? (sadece 5 proje neden öneriliyor?)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity

# Proje root'unu bul
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger
from src.scoring.final_ranking import FinalRankingCalculator

def analyze_diversity_issue(df_ranked):
    """Diversity neden 0.0028? Duplicate content kontrolü"""
    print("\n" + "=" * 80)
    print("🔍 SORUN 1: DIVERSITY = 0.0028 ANALİZİ")
    print("=" * 80)
    
    # Top-5 projeleri al
    top5 = df_ranked.head(5).copy()
    
    print(f"\nTop-5 Projeler:")
    for i, (idx, row) in enumerate(top5.iterrows(), 1):
        print(f"  {i}. Link: {str(row.get('link', ''))[:60]}...")
        if 'tam_aciklama' in row:
            desc = str(row['tam_aciklama'])[:100] if pd.notna(row['tam_aciklama']) else "N/A"
            print(f"     Açıklama: {desc}...")
        if 'cikarilan_beceriler' in row:
            skills = row['cikarilan_beceriler']
            if isinstance(skills, list):
                print(f"     Beceriler ({len(skills)}): {', '.join(skills[:10])}")
            else:
                print(f"     Beceriler: {skills}")
    
    # 1. Duplicate content kontrolü
    print("\n📊 1. DUPLICATE CONTENT KONTROLÜ:")
    if 'tam_aciklama' in top5.columns:
        descriptions = top5['tam_aciklama'].fillna('').astype(str)
        unique_descriptions = descriptions.nunique()
        print(f"   Benzersiz açıklama sayısı: {unique_descriptions}/5")
        if unique_descriptions < 5:
            print(f"   ⚠️  {5 - unique_descriptions} proje aynı açıklamaya sahip!")
            # Aynı açıklamaları bul
            desc_counts = descriptions.value_counts()
            for desc, count in desc_counts.items():
                if count > 1:
                    print(f"   ⚠️  '{desc[:50]}...' açıklaması {count} kez tekrarlanıyor")
    
    # 2. Jenerik kelimeler kontrolü
    print("\n📊 2. JENERİK KELİMELER KONTROLÜ:")
    if 'cikarilan_beceriler' in top5.columns:
        all_skills = []
        for skills in top5['cikarilan_beceriler']:
            if isinstance(skills, list):
                all_skills.extend(skills)
        
        if all_skills:
            skill_counts = Counter(all_skills)
            print(f"   Toplam beceri sayısı: {len(all_skills)}")
            print(f"   Benzersiz beceri sayısı: {len(skill_counts)}")
            print(f"   En çok geçen beceriler:")
            for skill, count in skill_counts.most_common(15):
                print(f"     - {skill}: {count} kez ({count*100/len(all_skills):.1f}%)")
            
            # Jenerik kelimeler
            generic_skills = ['communication', 'teamwork', 'collaboration', 'leadership', 
                            'problem solving', 'time management', 'organization']
            generic_count = sum(skill_counts.get(skill.lower(), 0) for skill in generic_skills)
            print(f"\n   Jenerik beceri toplamı: {generic_count}/{len(all_skills)} ({generic_count*100/len(all_skills):.1f}%)")
            if generic_count > len(all_skills) * 0.5:
                print(f"   ⚠️  %50'den fazla jenerik beceri var!")
    
    # 3. Vektör benzerliği kontrolü
    print("\n📊 3. VEKTÖR BENZERLİĞİ KONTROLÜ:")
    if 'beceri_vektorleri' in top5.columns:
        vectors_list = []
        valid_indices = []
        
        for idx, row in top5.iterrows():
            vector = row['beceri_vektorleri']
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
            
            print(f"   İşlenebilen vektör sayısı: {len(vectors_list)}/5")
            print(f"   Ortalama benzerlik: {similarities.mean():.6f}")
            print(f"   Min benzerlik: {similarities.min():.6f}")
            print(f"   Max benzerlik: {similarities.max():.6f}")
            print(f"   Std sapma: {similarities.std():.6f}")
            
            if similarities.mean() > 0.95:
                print(f"   ⚠️  %95'ten fazla benzerlik var! Vektörler neredeyse identik!")
            elif similarities.mean() > 0.80:
                print(f"   ⚠️  %80'den fazla benzerlik var! Vektörler çok benzer!")
        else:
            print(f"   ❌ Yeterli vektör yok: {len(vectors_list)}/5")

def analyze_confidence_issue(df_ranked):
    """Confidence neden 0.04? M skorları neden aynı?"""
    print("\n" + "=" * 80)
    print("🔍 SORUN 2: CONFIDENCE = 0.04 ANALİZİ (M SKORLARI)")
    print("=" * 80)
    
    # M skorlarını kontrol et
    m_score_cols = ['M_Skoru', 'M_Skoru_Ham', 'M_Skoru_Normalize']
    available_cols = [col for col in m_score_cols if col in df_ranked.columns]
    
    if not available_cols:
        print("   ❌ M_Skoru sütunu bulunamadı!")
        return
    
    for col in available_cols:
        scores = df_ranked[col].dropna()
        if len(scores) == 0:
            continue
        
        print(f"\n📊 {col} İstatistikleri:")
        print(f"   Toplam proje: {len(scores)}")
        print(f"   Min: {scores.min():.4f}")
        print(f"   Max: {scores.max():.4f}")
        print(f"   Ortalama: {scores.mean():.4f}")
        print(f"   Medyan: {scores.median():.4f}")
        print(f"   Std Sapma: {scores.std():.4f}")
        print(f"   Benzersiz değer sayısı: {scores.nunique()}")
        
        if scores.std() < 0.1:
            print(f"   ⚠️  Standart sapma çok düşük! Skorlar neredeyse aynı!")
        
        # İlk 10 skoru göster
        print(f"   İlk 10 skor: {scores.head(10).tolist()}")
        
        # Aynı skora sahip proje sayısı
        score_counts = scores.value_counts()
        duplicate_scores = score_counts[score_counts > 1]
        if len(duplicate_scores) > 0:
            print(f"   ⚠️  {len(duplicate_scores)} farklı skor {duplicate_scores.sum()} projede tekrarlanıyor")
            print(f"   En çok tekrarlanan skorlar:")
            for score, count in duplicate_scores.head(5).items():
                print(f"     - {score:.4f}: {count} proje")
    
    # Top-5 M skorlarını detaylı göster
    print(f"\n📊 Top-5 M Skorları Detayı:")
    top5 = df_ranked.head(5)
    for i, (idx, row) in enumerate(top5.iterrows(), 1):
        print(f"   {i}. ", end="")
        for col in available_cols:
            val = row.get(col, None)
            if pd.notna(val):
                print(f"{col}={val:.4f} ", end="")
        print()

def analyze_rc_issue(df_ranked):
    """RC neden 0.0053? Neden sadece 5 proje öneriliyor?"""
    print("\n" + "=" * 80)
    print("🔍 SORUN 3: RC = 0.0053 ANALİZİ (RECOMMENDATION COVERAGE)")
    print("=" * 80)
    
    total_projects = len(df_ranked)
    print(f"\n📊 Genel İstatistikler:")
    print(f"   Toplam proje sayısı: {total_projects}")
    
    # Top-K kontrolü
    print(f"\n📊 Top-K Analizi:")
    for k in [5, 10, 20, 50, 100]:
        top_k = df_ranked.head(k)
        rc = k / total_projects
        print(f"   Top-{k}: RC = {rc:.4f} ({k}/{total_projects})")
        
        # Bu top-K içinde kaç tanesi yeterli skora sahip?
        if 'Adil_Skor' in df_ranked.columns:
            threshold = 40.0
            eligible_in_topk = len(top_k[top_k['Adil_Skor'] > threshold])
            eligible_total = len(df_ranked[df_ranked['Adil_Skor'] > threshold])
            print(f"      Yeterli skora sahip (Adil_Skor > {threshold}): {eligible_in_topk}/{eligible_total} proje")
    
    # Adil_Skor dağılımı
    if 'Adil_Skor' in df_ranked.columns:
        print(f"\n📊 Adil_Skor Dağılımı:")
        scores = df_ranked['Adil_Skor'].dropna()
        print(f"   Toplam proje: {len(scores)}")
        print(f"   Min: {scores.min():.2f}")
        print(f"   Max: {scores.max():.2f}")
        print(f"   Ortalama: {scores.mean():.2f}")
        print(f"   Medyan: {scores.median():.2f}")
        print(f"   Std Sapma: {scores.std():.2f}")
        
        # Eşik üzerindeki projeler
        thresholds = [30, 40, 50, 60, 70]
        for thresh in thresholds:
            count = len(scores[scores > thresh])
            pct = count * 100 / len(scores)
            print(f"   Adil_Skor > {thresh}: {count} proje ({pct:.1f}%)")
    
    # Top-5'in sürekli aynı projeler mi?
    print(f"\n📊 Top-5 Proje Stabilitesi:")
    top5_links = df_ranked.head(5)['link'].tolist()
    print(f"   Top-5 link'ler:")
    for i, link in enumerate(top5_links, 1):
        print(f"     {i}. {str(link)[:60]}...")

def main():
    """Ana analiz fonksiyonu"""
    print("=" * 80)
    print("🔍 KRİTİK SORUNLAR TEŞHİS RAPORU")
    print("=" * 80)
    
    # Config yükle
    config_path = project_root / "config" / "config.yaml"
    config = load_config(str(config_path))
    
    # Logger
    logger_config = config.get('logging', {})
    logger = setup_logger(
        name=logger_config.get('name', 'diagnosis'),
        log_level='WARNING',
        console=False
    )
    
    # CSV yükle
    csv_file = project_root / "outputs" / "results" / "IP5_Nihai_Adil_Siralama.csv"
    if not csv_file.exists():
        print(f"\n❌ CSV bulunamadı: {csv_file}")
        return
    
    df_ranked = pd.read_csv(csv_file)
    print(f"\n✅ CSV yüklendi: {len(df_ranked)} proje")
    
    # Vektörleri yükle
    vector_file = project_root / "data/processed/IP1_Ciktisi_Projeler_Vektorlu.pkl"
    if vector_file.exists():
        with open(vector_file, 'rb') as f:
            vectors_data = pickle.load(f)
        
        if isinstance(vectors_data, pd.DataFrame):
            vectors_df = vectors_data[['link', 'beceri_vektorleri']].copy()
            vectors_df['link'] = vectors_df['link'].astype(str)
            df_ranked['link'] = df_ranked['link'].astype(str)
            df_ranked = df_ranked.merge(vectors_df, on='link', how='left')
            print(f"✅ Vektörler yüklendi: {df_ranked['beceri_vektorleri'].notna().sum()}/{len(df_ranked)} proje")
    
    # Analizleri yap
    analyze_diversity_issue(df_ranked)
    analyze_confidence_issue(df_ranked)
    analyze_rc_issue(df_ranked)
    
    print("\n" + "=" * 80)
    print("✅ TEŞHİS TAMAMLANDI")
    print("=" * 80)

if __name__ == "__main__":
    main()
