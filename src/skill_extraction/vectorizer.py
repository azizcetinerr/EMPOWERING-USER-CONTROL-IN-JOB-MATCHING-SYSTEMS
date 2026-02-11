"""
Becerileri vektörlere dönüştüren modül (XLM-RoBERTa)
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from typing import List, Union
import logging


class SkillVectorizer:
    """Beceri vektörizasyon sınıfı"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        model_config = config.get('models', {}).get('embedding', {})
        self.model_name = model_config.get('model_name', 'xlm-roberta-base')
        self.batch_size = model_config.get('batch_size', 32)
        
        # Cihaz belirleme
        device_str = model_config.get('device', 'auto')
        if device_str == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)
        
        self.logger.info(f"Kullanılan cihaz: {self.device}")
        
        # Modeli yükle
        try:
            self.logger.info(f"Model yükleniyor: {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            
            # Torch compile optimizasyonu (PyTorch 2.0+)
            try:
                if hasattr(torch, 'compile') and self.config.get('performance', {}).get('use_torch_compile', False):
                    self.logger.info("Torch compile optimizasyonu uygulanıyor...")
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    self.logger.info("✅ Torch compile aktif")
            except Exception as e:
                self.logger.debug(f"Torch compile uygulanamadı (normal): {e}")
            
            self.logger.info("Model başarıyla yüklendi")
        except Exception as e:
            self.logger.error(f"Model yüklenirken hata: {e}")
            raise
    
    def vectorize_skills(self, skills: Union[List[str], List[List[str]]]) -> np.ndarray:
        """
        Beceri listesini vektörlere dönüştürür.
        
        Args:
            skills: Beceri listesi veya beceri listelerinin listesi
        
        Returns:
            Vektör matrisi (numpy array)
        """
        if not skills or len(skills) == 0:
            return np.zeros((1, 768))  # XLM-R-base boyutu
        
        # Eğer iç içe liste ise (her proje için beceri listesi)
        if isinstance(skills[0], list):
            # Her proje için becerileri birleştir
            skill_texts = [' '.join(skill_list) if skill_list else '' for skill_list in skills]
        else:
            skill_texts = skills
        
        # Batch işleme
        all_embeddings = []
        
        for i in range(0, len(skill_texts), self.batch_size):
            batch = skill_texts[i:i + self.batch_size]
            
            with torch.no_grad():
                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512
                ).to(self.device)
                
                outputs = self.model(**inputs)
                # Ortalama pooling
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
                all_embeddings.append(embeddings)
        
        result = np.vstack(all_embeddings)
        return result
    
    def vectorize_dataframe(self, df: pd.DataFrame, skill_column: str = "cikarilan_beceriler") -> pd.DataFrame:
        """
        DataFrame'deki beceri listelerini vektörlere dönüştürür.
        
        Args:
            df: İşlenecek DataFrame
            skill_column: Beceri listesi sütunu adı
        
        Returns:
            Vektörler eklenmiş DataFrame
        """
        self.logger.info(f"{len(df)} adet proje vektörize ediliyor...")
        
        skill_lists = df[skill_column].tolist()
        vectors = []
        
        for skill_list in skill_lists:
            if not skill_list or len(skill_list) == 0:
                vectors.append(np.zeros((1, 768)))
            else:
                vec = self.vectorize_skills(skill_list)
                vectors.append(vec)
        
        df_result = df.copy()
        df_result["beceri_vektorleri"] = vectors
        
        self.logger.info("Tüm projeler vektörize edildi")
        return df_result

