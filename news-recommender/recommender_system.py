# recommender_system.py
import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class NewsRecommender:
    def __init__(self):
        self.df = None
        self.vectorizer = None
        self.X = None
    
    def preprocess_text(self, text):
        """Your text preprocessing function"""
        # Add your text cleaning steps here
        return text.lower()  # Example - replace with your actual preprocessing
    
    def load_data(self, filepath):
        """Load and prepare data"""
        self.df = pd.read_csv(filepath)
        
        # Ensure article_id exists or create it
        if 'article_id' not in self.df.columns:
            self.df['article_id'] = range(1, len(self.df)+1)
        
        # Preprocess text
        self.df['processed_content'] = self.df['content'].apply(self.preprocess_text)
    
    def train_model(self):
        """Train the vectorizer and create similarity matrix"""
        self.vectorizer = TfidfVectorizer()
        self.X = self.vectorizer.fit_transform(self.df['processed_content'])
    
    def find_similar_articles(self, target_article_id, top_n=5):
        """Find similar articles"""
        try:
            target_idx = self.df[self.df['article_id'] == target_article_id].index[0]
            cosine_sim = cosine_similarity(self.X[target_idx], self.X).flatten()
            
            # Get most similar articles (excluding itself)
            similar_indices = np.argsort(cosine_sim)[-top_n-1:-1][::-1]
            
            results = self.df.iloc[similar_indices].copy()
            results['similarity_score'] = cosine_sim[similar_indices]
            return results
        
        except Exception as e:
            print(f"Error finding similar articles: {str(e)}")
            return pd.DataFrame()
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'df': self.df,
                'vectorizer': self.vectorizer,
                'X': self.X
            }, f)
    
    @classmethod
    def load_model(cls, filepath):
        """Load a trained model from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        recommender = cls()
        recommender.df = data['df']
        recommender.vectorizer = data['vectorizer']
        recommender.X = data['X']
        return recommender

if __name__ == "__main__":
    # Example usage
    recommender = NewsRecommender()
    recommender.load_data('news_data.csv')  # Replace with your data path
    recommender.train_model()
    recommender.save_model('news_recommender.pkl')