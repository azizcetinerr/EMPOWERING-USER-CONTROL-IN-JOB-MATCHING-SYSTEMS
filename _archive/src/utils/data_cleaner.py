"""
Data cleaning and preprocessing functions
"""

import pandas as pd
from typing import List, Set, Dict, Any
import re


def remove_duplicate_links(links: List[str]) -> List[str]:
    """
    Removes duplicate links and sorts them.
    
    Args:
        links: List of links
    
    Returns:
        List of unique links
    """
    unique_links = list(dict.fromkeys(links))  # Remove duplicates while preserving order
    return unique_links


def filter_skills(
    skills: List[str],
    min_length: int = 2,
    stopwords: Set[str] = None
) -> List[str]:
    """
    Filters a skill list.
    
    Args:
        skills: Raw skill list
        min_length: Minimum length in characters
        stopwords: Words to filter out
    
    Returns:
        Filtered skill list
    """
    if stopwords is None:
        stopwords = {"com", "lo", "e", "m", "km", "a", "i", "o", "u"}
    
    filtered = []
    for skill in skills:
        skill_lower = skill.lower().strip()
        
        # Minimum length check
        if len(skill_lower) < min_length:
            continue
        
        # Stopword check
        if skill_lower in stopwords:
            continue
        
        # Keep only skills that contain letters (optional - can be removed for numeric skills)
        # if not re.search(r'[a-zA-Z]', skill_lower):
        #     continue
        
        filtered.append(skill)
    
    return list(set(filtered))  # Remove duplicates


def clean_project_descriptions(df: pd.DataFrame, description_col: str = "full_description") -> pd.DataFrame:
    """
    Cleans project descriptions.
    
    Args:
        df: Project data DataFrame
        description_col: Name of the description column
    
    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove empty descriptions
    df_clean = df_clean[df_clean[description_col].notna()]
    df_clean = df_clean[df_clean[description_col].str.strip() != ""]
    
    # Remove extra whitespace
    df_clean[description_col] = df_clean[description_col].str.replace(r'\s+', ' ', regex=True)
    
    return df_clean.reset_index(drop=True)


def normalize_skill_name(skill: str) -> str:
    """
    Normalizes a skill name (lower case, whitespace cleanup).
    
    Args:
        skill: Raw skill name
    
    Returns:
        Normalized skill name
    """
    normalized = skill.lower().strip()
    normalized = re.sub(r'\s+', ' ', normalized)  # Remove extra whitespace
    return normalized


def validate_data_quality(df: pd.DataFrame, required_columns: List[str]) -> Dict[str, Any]:
    """
    Runs a data quality check.
    
    Args:
        df: DataFrame to check
        required_columns: Required columns
    
    Returns:
        Quality report dictionary
    """
    report = {
        "total_rows": len(df),
        "missing_columns": [],
        "empty_rows": {},
        "duplicates": 0
    }
    
    # Missing column check
    for col in required_columns:
        if col not in df.columns:
            report["missing_columns"].append(col)
        else:
            empty_count = df[col].isna().sum() + (df[col].astype(str).str.strip() == "").sum()
            report["empty_rows"][col] = empty_count
    
    # Duplicate check
    if "link" in df.columns:
        report["duplicates"] = df["link"].duplicated().sum()
    
    return report

