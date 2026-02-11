"""
Samimiyet skoru (S-Skoru) hesaplayan modül
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import logging


class SincerityScoreCalculator:
    """Samimiyet skoru hesaplayıcı sınıfı"""
    
    def __init__(self, config: dict, logger: logging.Logger = None):
        """
        Args:
            config: Konfigürasyon sözlüğü
            logger: Logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        
        # AI Detection modeli
        ai_config = config.get('models', {}).get('ai_detection', {})
        ai_model_name = ai_config.get('model_name', 'roberta-base-openai-detector')
        
        self.logger.info(f"AI Detection modeli yükleniyor: {ai_model_name}...")
        self.ai_tokenizer = AutoTokenizer.from_pretrained(ai_model_name)
        self.ai_model = AutoModelForSequenceClassification.from_pretrained(ai_model_name)
        self.ai_model.eval()
        
        # Embedding modeli
        embed_config = config.get('models', {}).get('embedding', {})
        embed_model_name = embed_config.get('model_name', 'xlm-roberta-base')
        device_str = embed_config.get('device', 'auto')
        
        if device_str == 'auto':
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device_str)
        
        self.logger.info(f"Embedding modeli yükleniyor: {embed_model_name}...")
        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model_name)
        self.embed_model = AutoModel.from_pretrained(embed_model_name).to(self.device)
        self.embed_model.eval()
        
        # Skorlama ayarları
        sincerity_config = config.get('sincerity_scoring', {})
        self.human_weight = sincerity_config.get('human_score_weight', 0.4)
        self.context_weight = sincerity_config.get('context_score_weight', 0.6)
        self.top_n = sincerity_config.get('top_n_projects', 20)
        
        # Motivasyon mektubu
        self.motivation_letter = config.get('data_sources', {}).get('motivation_letter', {}).get('text', '')
        if not self.motivation_letter:
            self.logger.warning("Motivasyon mektubu bulunamadı!")
        
        # Mektubun vektörünü önceden hesapla
        if self.motivation_letter:
            self.letter_vector = self._get_embedding(self.motivation_letter)
            self.human_score = self._detect_ai_generated(self.motivation_letter)
            self.logger.info(f"Motivasyon mektubu AI skoru: {self.human_score:.4f}")
    
    def _detect_ai_generated(self, text: str) -> float:
        """Metnin AI tarafından yazılma olasılığını döndürür"""
        inputs = self.ai_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = self.ai_model(**inputs).logits
        probs = F.softmax(logits, dim=1)
        ai_probability = probs[0][1].item()
        return ai_probability
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Metnin embedding'ini döndürür"""
        inputs = self.embed_tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.embed_model(**inputs)
        
        return outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    
    def calculate_sincerity_score(self, project_text: str) -> tuple:
        """
        Proje için samimiyet skorunu hesaplar.
        
        Args:
            project_text: Proje açıklama metni
        
        Returns:
            (S_Skoru, Insan_Puani, Baglam_Puani) tuple'ı
        """
        # Proje vektörünü al
        project_vector = self._get_embedding(str(project_text))
        
        # Kosinüs benzerliği
        similarity = cosine_similarity(self.letter_vector, project_vector)[0][0]
        context_score = similarity * 100
        
        # İnsan yazımı skoru (tersi)
        human_score = (1 - self.human_score) * 100
        
        # Nihai S skoru
        final_s = (human_score * self.human_weight) + (context_score * self.context_weight)
        
        return final_s, human_score, context_score
    
    def calculate_all_scores(
        self,
        df_projeler: pd.DataFrame,
        text_column: str = "tam_aciklama",
        calculate_all: bool = False
    ) -> pd.DataFrame:
        """
        Tüm projeler için samimiyet skorlarını hesaplar (sadece top N veya tümü).
        
        Args:
            df_projeler: Proje DataFrame'i (M_Skoru ile sıralanmış olmalı)
            text_column: Metin sütunu adı
            calculate_all: True ise tüm projeler için, False ise sadece top N için hesapla
        
        Returns:
            S_Skoru eklenmiş DataFrame
        """
        # Hızlı mod: Sadece en iyi N projeyi işle (varsayılan)
        if calculate_all:
            df_top = df_projeler.copy()
            self.logger.info(f"{len(df_top)} adet proje için samimiyet skorları hesaplanıyor (tüm projeler)...")
        else:
            df_top = df_projeler.head(self.top_n).copy()
            self.logger.info(f"{len(df_top)} adet proje için samimiyet skorları hesaplanıyor (top {self.top_n} proje)...")
        
        scores = []
        human_scores = []
        context_scores = []
        
        for idx, row in df_top.iterrows():
            s_score, h_score, c_score = self.calculate_sincerity_score(row[text_column])
            scores.append(s_score)
            human_scores.append(h_score)
            context_scores.append(c_score)
        
        df_result = df_top.copy()
        df_result["S_Skoru"] = scores
        df_result["Insan_Puani"] = human_scores
        df_result["Baglam_Puani"] = context_scores
        
        # Eğer sadece top N hesaplandıysa, diğer projeler için NaN değerleri ekle
        if not calculate_all and len(df_projeler) > len(df_result):
            # Orijinal DataFrame'den eksik projeleri al
            remaining_df = df_projeler.iloc[len(df_result):].copy()
            remaining_df["S_Skoru"] = None
            remaining_df["Insan_Puani"] = None
            remaining_df["Baglam_Puani"] = None
            
            # Sonuçları birleştir
            df_result = pd.concat([df_result, remaining_df], ignore_index=True)
        
        self.logger.info("Samimiyet skorları hesaplandı")
        return df_result

