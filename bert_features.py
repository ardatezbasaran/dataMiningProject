import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm

class BERTFeatureExtractor:
    def __init__(self, model_name='dbmdz/bert-base-turkish-cased'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
    
    def extract_features(self, texts, batch_size=4):
        """Metinlerden BERT özelliklerini çıkarır."""
        features = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,  # Daha kısa maksimum uzunluk
                return_tensors='pt'
            )
            
            # GPU'ya taşı
            input_ids = encoded['input_ids'].to(self.device)
            attention_mask = encoded['attention_mask'].to(self.device)
            
            # Özellikleri çıkar
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
                # [CLS] token'ının embedding'ini al
                batch_features = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                features.extend(batch_features)
            
            # Belleği temizle
            del input_ids
            del attention_mask
            del outputs
            torch.cuda.empty_cache()
        
        return np.array(features) 