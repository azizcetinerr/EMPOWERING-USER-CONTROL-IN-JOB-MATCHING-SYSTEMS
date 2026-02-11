"""
Veri temizleme ve ön işleme fonksiyonları
"""

import pandas as pd
from typing import List, Set, Dict, Any
import re


def remove_duplicate_links(links: List[str]) -> List[str]:
    """
    Tekrar eden linkleri kaldırır ve sıralar.
    
    Args:
        links: Link listesi
    
    Returns:
        Tekrarsız link listesi
    """
    unique_links = list(dict.fromkeys(links))  # Sırayı koruyarak tekrarları kaldır
    return unique_links


def filter_skills(
    skills: List[str],
    min_length: int = 2,
    stopwords: Set[str] = None
) -> List[str]:
    """
    Beceri listesini filtreler.
    
    Args:
        skills: Ham beceri listesi
        min_length: Minimum karakter uzunluğu
        stopwords: Filtrelenecek kelimeler
    
    Returns:
        Filtrelenmiş beceri listesi
    """
    if stopwords is None:
        stopwords = {"com", "lo", "e", "m", "km", "a", "i", "o", "u"}
    
    filtered = []
    for skill in skills:
        skill_lower = skill.lower().strip()
        
        # Minimum uzunluk kontrolü
        if len(skill_lower) < min_length:
            continue
        
        # Stopword kontrolü
        if skill_lower in stopwords:
            continue
        
        # Sadece harf içeren beceriler (opsiyonel - sayısal beceriler için kaldırılabilir)
        # if not re.search(r'[a-zA-Z]', skill_lower):
        #     continue
        
        filtered.append(skill)
    
    return list(set(filtered))  # Tekrarları kaldır


def clean_project_descriptions(df: pd.DataFrame, description_col: str = "tam_aciklama") -> pd.DataFrame:
    """
    Proje açıklamalarını temizler.
    
    Args:
        df: Proje verisi DataFrame
        description_col: Açıklama sütunu adı
    
    Returns:
        Temizlenmiş DataFrame
    """
    import re
    
    df_clean = df.copy()
    
    # Boş açıklamaları kaldır
    df_clean = df_clean[df_clean[description_col].notna()]
    df_clean = df_clean[df_clean[description_col].str.strip() != ""]
    
    # Dil kodlarını ve gereksiz karakterleri temizle
    def clean_text(text):
        if pd.isna(text) or not isinstance(text, str):
            return text
        
        # BGбългарскиCSčeština gibi dil kodlarını temizle
        # İki harfli kodlar (BG, CS, DA, DE...) ve hemen ardından gelen karakterleri temizle
        text = re.sub(r'[A-Z]{2}[а-яА-ЯёЁa-záčďéěíňóřšťúůýžàáâãäåæçèéêëìíîïðñòóôõöøùúûüýþÿığşüöçąćęłńóśźż]+', '', text)
        
        # Tekrarlayan büyük harfli kodlar (BGCSDADE...)
        text = re.sub(r'\b[A-Z]{2,10}\b(?=[а-яА-ЯёЁa-záčďéěíňóřšťúůýž])', '', text)
    
    # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    if description_col in df_clean.columns:
        df_clean[description_col] = df_clean[description_col].apply(clean_text)
    
    return df_clean.reset_index(drop=True)


def normalize_skill_name(skill: str) -> str:
    """
    Beceri adını normalize eder (küçük harf, boşluk temizleme).
    
    Args:
        skill: Ham beceri adı
    
    Returns:
        Normalize edilmiş beceri adı
    """
    normalized = skill.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)  # Fazla boşlukları temizle
    return normalized


def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Veri kalitesi kontrolü yapar.
    
    Args:
        df: Kontrol edilecek DataFrame
        required_columns: Gerekli sütunlar
    
    Returns:
        Kalite raporu sözlüğü
    """
    report = {
        "total_rows": len(df),
        "missing_columns": [],
        "empty_rows": {},
        "duplicates": 0
    }
    
    # Eksik sütun kontrolü
    for col in required_columns:
        if col not in df.columns:
            report["missing_columns"].append(col)
        else:
            empty_count = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
            report["empty_rows"][col] = empty_count
    
    # Duplicate kontrolü
    if "link" in df.columns:
        report["duplicates"] = df["link"].duplicated().sum()
    
    return report

