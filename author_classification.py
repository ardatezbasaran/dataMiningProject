import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import seaborn as sns
from bert_features import BERTFeatureExtractor

class AuthorClassifier:
    def __init__(self, data_path):
        self.data_path = data_path
        self.texts = []
        self.authors = []
        self.load_data()
        
    def load_data(self):
        """Veri setini yükler ve metinleri ve yazarları listeler."""
        for author_dir in os.listdir(self.data_path):
            author_path = os.path.join(self.data_path, author_dir)
            if os.path.isdir(author_path):
                for file in os.listdir(author_path):
                    if file.endswith('.txt'):
                        file_path = os.path.join(author_path, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                                if len(text.strip()) > 0:  # Boş dosyaları atla
                                    self.texts.append(text)
                                    self.authors.append(author_dir)
                        except Exception as e:
                            print(f"Hata: {file_path} dosyası okunamadı - {str(e)}")
    
    def extract_features(self, method='tfidf', ngram_range=(1,1)):
        """Farklı özellik çıkarma yöntemlerini uygular."""
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(ngram_range=ngram_range)
            features = vectorizer.fit_transform(self.texts)
            return features
        elif method == 'bert':
            bert_extractor = BERTFeatureExtractor()
            features = bert_extractor.extract_features(self.texts)
            return features
    
    def train_and_evaluate(self, X, y):
        """Farklı sınıflandırıcıları eğitir ve değerlendirir."""
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Eğer X sparse matrix ise ve SVM/MLP kullanılacaksa yoğun matrise dönüştür
        if hasattr(X, 'toarray'):
            X_train_dense = X_train.toarray()
            X_test_dense = X_test.toarray()
        else:
            X_train_dense = X_train
            X_test_dense = X_test
        
        classifiers = {
            'Random Forest': (RandomForestClassifier(), False),
            'SVM': (SVC(), True),
            'XGBoost': (XGBClassifier(), True),
            'Naive Bayes': (MultinomialNB(), False),
            'MLP': (MLPClassifier(), True),
            'Decision Tree': (DecisionTreeClassifier(), False)
        }
        
        results = []
        for name, (clf, needs_dense) in classifiers.items():
            try:
                # Yoğun matris gerekiyorsa onu kullan
                if needs_dense:
                    clf.fit(X_train_dense, y_train)
                    y_pred = clf.predict(X_test_dense)
                else:
                    clf.fit(X_train, y_train)
                    y_pred = clf.predict(X_test)
                
                results.append({
                    'Classifier': name,
                    'Accuracy': accuracy_score(y_test, y_pred),
                    'Precision': precision_score(y_test, y_pred, average='weighted'),
                    'Recall': recall_score(y_test, y_pred, average='weighted'),
                    'F1-Score': f1_score(y_test, y_pred, average='weighted')
                })
            except Exception as e:
                print(f"Hata: {name} sınıflandırıcısı çalıştırılamadı - {str(e)}")
        
        return pd.DataFrame(results)

def main():
    # Veri seti yolu
    data_path = 'dataset_authorship'
    
    # Sınıflandırıcıyı oluştur
    classifier = AuthorClassifier(data_path)
    
    # Farklı özellik çıkarma yöntemlerini dene
    methods = [
        ('TF-IDF (1-gram)', 'tfidf', (1,1)),
        ('TF-IDF (2-gram)', 'tfidf', (2,2)),
        ('TF-IDF (3-gram)', 'tfidf', (3,3)),
        ('BERT', 'bert', None)
    ]
    
    all_results = []
    for name, method, ngram_range in methods:
        print(f"\n{name} özellik çıkarma yöntemi kullanılıyor...")
        X = classifier.extract_features(method=method, ngram_range=ngram_range)
        results = classifier.train_and_evaluate(X, classifier.authors)
        results['Method'] = name
        all_results.append(results)
        print(f"{name} sonuçları:")
        print(results)
    
    # Sonuçları birleştir
    final_results = pd.concat(all_results)
    
    # Sonuçları kaydet
    final_results.to_csv('classification_results.csv', index=False)
    print("\nTüm sonuçlar:")
    print(final_results)
    
    # Sonuçları görselleştir
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=final_results, x='Classifier', y=metric, hue='Method')
        plt.xticks(rotation=45)
        plt.title(f'Sınıflandırıcı {metric} Karşılaştırması')
        plt.tight_layout()
        plt.savefig(f'results_{metric.lower()}.png')
        plt.close()

if __name__ == "__main__":
    main() 