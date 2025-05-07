# app.py
import streamlit as st
import pandas as pd
import pickle
from recommender_system import NewsRecommender

def main():
    st.set_page_config(page_title="News Recommender", layout="wide")
    
    # Load the recommender system
    try:
        recommender = NewsRecommender.load_model('news_recommender.pkl')
    except FileNotFoundError:
        st.error("Model file not found. Please train the model first.")
        return
    
    # Sidebar with filters
    st.sidebar.header("Filters")
    min_score = st.sidebar.slider("Minimum similarity score", 0.0, 1.0, 0.5)
    top_n = st.sidebar.slider("Number of recommendations", 1, 10, 5)
    
    # Main interface
    st.title("News Article Recommender System")
    
    # Display article selection
    article_titles = recommender.df.set_index('article_id')['title'].to_dict()
    selected_id = st.selectbox(
        "Select an article to get recommendations:",
        options=list(article_titles.keys()),
        format_func=lambda x: f"{x}: {article_titles[x]}"
    )
    
    if st.button("Get Recommendations"):
        recommendations = recommender.find_similar_articles(selected_id, top_n)
        recommendations = recommendations[recommendations['similarity_score'] >= min_score]
        
        if not recommendations.empty:
            st.subheader(f"Recommended Articles (similar to {article_titles[selected_id]}):")
            
            for _, row in recommendations.iterrows():
                with st.expander(f"{row['title']} (Score: {row['similarity_score']:.2f})"):
                    st.write(f"**Source:** {row.get('newspaper', 'N/A')}")
                    st.write(f"**Date:** {row.get('date', 'N/A')}")
                    st.write(f"**Similarity Score:** {row['similarity_score']:.2f}")
                    st.write("**Content Preview:**")
                    st.write(row['content'][:500] + "...")
        else:
            st.warning("No recommendations found matching your criteria.")

if __name__ == "__main__":
    main()