#!/usr/bin/env python3
"""
Diversity Score debug script'i
Diversity Score'un neden 0.0 kaldığını analiz eder
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Proje root'unu bul
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config
from src.utils.logger import setup_logger

def debug_diversity():
    """Diversity Score'u debug et"""
    print("=" * 80)
    print("🔍 DIVERSITY SCORE DEBUG")
    print("=" * 80)
    
    # CSV yükle
    csv_file = project_root / "outputs" / "results" / "IP5_Nihai_Adil_Siralama.csv"
    if not csv_file.exists():
        print(f"❌ CSV dosyası bulunamadı: {csv_file}")
        return
    
    df_ranked = pd.read_csv(csv_file)
    print(f"✅ CSV yüklendi: {len(df_ranked)} proje")
    
    # Vektörleri yükle
    vector_files = [
        project_root / "data/processed/IP1_Ciktisi_Projeler_Vektorlu.pkl",
        project_root / "outputs" / "processed" / "IP1_Ciktisi_Projeler_Vektorlu.pkl",
    ]
    
    vectors_loaded = False
    for vector_file in vector_files:
        if vector_file.exists():
            print(f"\n📂 Vektörler yükleniyor: {vector_file}")
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
                    
                    df_ranked['beceri_vektorleri'] = df_ranked['link'].apply(get_vector)
                    vectors_loaded_count = df_ranked['beceri_vektorleri'].notna().sum()
                    print(f"✅ Vektörler yüklendi: {vectors_loaded_count}/{len(df_ranked)}")
                    vectors_loaded = True
                    break
                elif isinstance(vectors_data, pd.DataFrame):
                    if 'link' in vectors_data.columns and 'beceri_vektorleri' in vectors_data.columns:
                        df_ranked = df_ranked.merge(
                            vectors_data[['link', 'beceri_vektorleri']],
                            on='link',
                            how='left'
                        )
                        vectors_loaded_count = df_ranked['beceri_vektorleri'].notna().sum()
                        print(f"✅ Vektörler yüklendi: {vectors_loaded_count}/{len(df_ranked)}")
                        vectors_loaded = True
                        break
            except Exception as e:
                print(f"⚠️  Hata: {e}")
                continue
    
    if not vectors_loaded or 'beceri_vektorleri' not in df_ranked.columns:
        print("❌ Vektörler yüklenemedi!")
        return
    
    # Top-5 projeleri analiz et
    top5 = df_ranked.head(5)
    print(f"\n📊 TOP-5 PROJE ANALİZİ")
    print("=" * 80)
    
    vectors_list = []
    valid_links = []
    
    for idx, row in top5.iterrows():
        link = row.get('link', 'N/A')
        vec = row.get('beceri_vektorleri')
        
        print(f"\nProje {idx}:")
        print(f"  Link: {link[:60]}...")
        print(f"  Adil_Skor: {row.get('Adil_Skor', 0):.2f}")
        print(f"  Vektör: {type(vec)}")
        
        if vec is None:
            print("  ⚠️  Vektör YOK!")
            continue
        
        try:
            vec_array = np.array(vec)
            print(f"  Vektör shape: {vec_array.shape}")
            print(f"  Vektör dtype: {vec_array.dtype}")
            print(f"  Vektör min/max: {vec_array.min():.4f} / {vec_array.max():.4f}")
            print(f"  Vektör norm: {np.linalg.norm(vec_array):.4f}")
            
            if vec_array.ndim > 1:
                vec_array = vec_array.flatten()
                print(f"  Flattened shape: {vec_array.shape}")
            
            norm = np.linalg.norm(vec_array)
            if norm > 0:
                vec_normalized = vec_array / norm
                vectors_list.append(vec_normalized)
                valid_links.append(link)
                print(f"  ✅ Normalize edildi (norm={norm:.4f})")
            else:
                print(f"  ⚠️  Sıfır norm!")
        except Exception as e:
            print(f"  ❌ Hata: {e}")
            import traceback
            print(traceback.format_exc()[:200])
    
    print(f"\n📊 VEKTÖR BENZERLİK ANALİZİ")
    print("=" * 80)
    print(f"Geçerli vektör sayısı: {len(vectors_list)}")
    
    if len(vectors_list) < 2:
        print("❌ Yeterli vektör yok (en az 2 gerekli)")
        return
    
    # Cosine similarity hesapla
    try:
        vectors_matrix = np.vstack(vectors_list)
        print(f"Vektör matrisi shape: {vectors_matrix.shape}")
        
        similarity_matrix = cosine_similarity(vectors_matrix)
        print(f"Benzerlik matrisi shape: {similarity_matrix.shape}")
        
        # Diagonal'i 0 yap
        np.fill_diagonal(similarity_matrix, 0.0)
        
        # Üst üçgen matris
        mask = np.triu(np.ones_like(similarity_matrix, dtype=bool), k=1)
        similarities = similarity_matrix[mask]
        
        if len(similarities) > 0:
            avg_sim = similarities.mean()
            min_sim = similarities.min()
            max_sim = similarities.max()
            std_sim = similarities.std()
            
            print(f"\nBenzerlik İstatistikleri:")
            print(f"  Ortalama: {avg_sim:.6f}")
            print(f"  Min: {min_sim:.6f}")
            print(f"  Max: {max_sim:.6f}")
            print(f"  Std: {std_sim:.6f}")
            
            diversity = 1.0 - avg_sim
            print(f"\nDiversity Score: {diversity:.6f}")
            
            if avg_sim > 0.999:
                print("\n⚠️  ÇOK YÜKSEK BENZERLİK (>0.999)!")
                print("   Projeler neredeyse aynı vektörlere sahip")
                print("   Bu, projelerin gerçekten çok benzer olduğunu gösterir")
            elif avg_sim > 0.99:
                print("\n⚠️  Yüksek benzerlik (>0.99)")
            elif avg_sim > 0.95:
                print("\n⚠️  Benzerlik yüksek (>0.95)")
            else:
                print("\n✅ Normal benzerlik seviyesi")
            
            # Pairwise benzerlikleri göster
            print(f"\nPairwise Benzerlikler:")
            for i in range(len(vectors_list)):
                for j in range(i+1, len(vectors_list)):
                    sim = similarity_matrix[i, j]
                    print(f"  {i} <-> {j}: {sim:.6f} (Link {i}: {valid_links[i][:30]}... <-> Link {j}: {valid_links[j][:30]}...)")
        else:
            print("❌ Benzerlik hesaplanamadı")
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        print(traceback.format_exc())

if __name__ == "__main__":
    debug_diversity()
