# How to Run:
# Save as social_media_dashboard.py
# Install requirements: pip install streamlit pandas matplotlib seaborn plotly wordcloud spacy pillow
# Run with: streamlit run social_media_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud
import spacy
from spacy import displacy
import pickle
import absa_utils  # Your custom ABSA utilities
from PIL import Image

# Set up page configuration
st.set_page_config(
    page_title="Airline Sentiment Analysis Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data and models
@st.cache_resource
def load_data():
    # Load cleaned data
    df = pd.read_csv('data/cleaned_data.csv')
    
    # Load ABSA results
    absa_df = pd.read_csv('data/absa_results.csv')
    
    # Load model results
    with open('models/traditional_ml/results_summary.pkl', 'rb') as f:
        ml_results = pickle.load(f)
    with open('models/deep_learning/deep_learning_results.pkl', 'rb') as f:
        dl_results = pickle.load(f)
    
    return df, absa_df, ml_results, dl_results

df, absa_df, ml_results, dl_results = load_data()

# Load spaCy model
nlp = spacy.load('en_core_web_lg')

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    .metric-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stPlotlyChart {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar controls
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio(
    "Select Analysis Type",
    ["Overview", "Sentiment Analysis", "Opinion Mining", "Aspect-Based Analysis", "Model Comparison"]
)

# Main content
st.title("✈️ Airline Sentiment Analysis Dashboard")

if analysis_type == "Overview":
    st.header("Dataset Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Basic Statistics")
        st.write(f"Total Tweets: {len(df)}")
        st.write(f"Unique Airlines: {df['airline'].nunique()}")
        
        # Sentiment distribution
        sentiment_dist = df['airline_sentiment'].value_counts()
        fig1 = px.pie(sentiment_dist, 
                     values=sentiment_dist.values, 
                     names=sentiment_dist.index,
                     title="Overall Sentiment Distribution")
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("Airlines Distribution")
        airline_dist = df['airline'].value_counts()
        fig2 = px.bar(airline_dist, 
                     x=airline_dist.index, 
                     y=airline_dist.values,
                     title="Tweets by Airline")
        st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

elif analysis_type == "Sentiment Analysis":
    st.header("Sentiment Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Distribution", "Word Clouds", "Time Trends"])
    
    with tab1:
        st.subheader("Sentiment Distribution by Airline")
        
        airline = st.selectbox("Select Airline", df['airline'].unique())
        filtered = df[df['airline'] == airline]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(filtered, x='airline_sentiment', 
                              color='airline_sentiment',
                              title=f"Sentiment for {airline}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            avg_length = filtered.groupby('airline_sentiment')['cleaned_text'].apply(
                lambda x: x.str.split().str.len().mean()
            ).reset_index()
            
            fig = px.bar(avg_length, x='airline_sentiment', y='cleaned_text',
                        title="Average Text Length by Sentiment")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Word Clouds by Sentiment")
        
        sentiment = st.selectbox("Select Sentiment", ['positive', 'neutral', 'negative'])
        text = ' '.join(df[df['airline_sentiment'] == sentiment]['cleaned_text'])
        
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        st.pyplot(plt)
    
    with tab3:
        st.subheader("Sentiment Trends Over Time")
        
        # Convert date if available (assuming your data has a date column)
        if 'tweet_created' in df.columns:
            df['date'] = pd.to_datetime(df['tweet_created']).dt.date
            daily_sentiment = df.groupby(['date', 'airline_sentiment']).size().unstack().fillna(0)
            
            fig = px.line(daily_sentiment, x=daily_sentiment.index, 
                         y=daily_sentiment.columns,
                         title="Daily Sentiment Trends")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No date information available in the dataset")

elif analysis_type == "Opinion Mining":
    st.header("Opinion Mining")
    
    st.subheader("Opinion Target Extraction")
    
    tweet_text = st.selectbox("Select a tweet to analyze", df['original_text'].sample(10))
    doc = nlp(tweet_text)
    
    # Extract opinion targets (noun phrases)
    targets = [chunk.text for chunk in doc.noun_chunks if chunk.root.pos_ in ['NOUN', 'PROPN']]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Original Tweet:**")
        st.write(tweet_text)
        
        st.markdown("**Identified Opinion Targets:**")
        for target in targets:
            st.write(f"- {target}")
    
    with col2:
        st.markdown("**Dependency Parse Visualization**")
        dep_svg = displacy.render(doc, style="dep", jupyter=False)
        st.image(dep_svg, width=600)
    
    st.subheader("Opinion Lexicon Analysis")
    
    # Load opinion lexicon (example - you should have your own lexicon)
    positive_words = ['good', 'great', 'excellent', 'awesome']
    negative_words = ['bad', 'poor', 'terrible', 'awful']
    
    pos_count = sum(1 for word in tweet_text.lower().split() if word in positive_words)
    neg_count = sum(1 for word in tweet_text.lower().split() if word in negative_words)
    
    st.metric("Positive Words Found", pos_count)
    st.metric("Negative Words Found", neg_count)

elif analysis_type == "Aspect-Based Analysis":
    st.header("Aspect-Based Sentiment Analysis")
    
    tab1, tab2 = st.tabs(["Aspect Explorer", "Interactive Analysis"])
    
    with tab1:
        st.subheader("Aspect Sentiment Distribution")
        
        aspect = st.selectbox("Select Aspect", absa_df['aspect'].value_counts().head(20).index)
        aspect_data = absa_df[absa_df['aspect'] == aspect]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(aspect_data, names='sentiment_label',
                        title=f"Sentiment for {aspect}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            airline_aspect = aspect_data.groupby(['overall_sentiment', 'sentiment_label']).size().unstack()
            fig = px.bar(airline_aspect, barmode='group',
                        title="Aspect Sentiment by Overall Tweet Sentiment")
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Example Tweets")
        for sentiment in ['Positive', 'Neutral', 'Negative']:
            st.markdown(f"**{sentiment} Examples:**")
            examples = aspect_data[aspect_data['sentiment_label'] == sentiment]['text'].head(3)
            for ex in examples:
                st.write(f"- {ex}")
    
    with tab2:
        st.subheader("Interactive ABSA Analysis")
        
        user_input = st.text_area("Enter your own text to analyze aspects:")
        
        if user_input:
            aspects = absa_utils.extract_aspects_enhanced(user_input)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Identified Aspects & Sentiment**")
                for aspect in aspects:
                    sentiment = absa_utils.analyze_aspect_sentiment(user_input, aspect)
                    st.write(f"- {aspect}: {['Negative', 'Neutral', 'Positive'][sentiment]}")
            
            with col2:
                st.markdown("**Visualization**")
                html = displacy.render(nlp(user_input), style="ent", options={"ents": ["ASPECT"]})
                st.components.v1.html(html, height=300, scrolling=True)

elif analysis_type == "Model Comparison":
    st.header("Model Performance Comparison")
    
    st.subheader("Traditional ML vs Deep Learning")
    
    # Create comparison dataframe
    comparison_data = []
    for model in ml_results['all_results']:
        comparison_data.append({
            'Model': model['Model'],
            'Type': 'Traditional ML',
            'Accuracy': model['Accuracy'],
            'F1-Score': model['F1-Score']
        })
    for model in dl_results['complete_comparison']:
        if model['Model'] in ['LSTM', 'Bidirectional LSTM']:
            comparison_data.append({
                'Model': model['Model'],
                'Type': 'Deep Learning',
                'Accuracy': model['Accuracy'],
                'F1-Score': model['F1-Score']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.bar(comparison_df, x='Model', y='Accuracy', color='Type',
                    title="Model Accuracy Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.bar(comparison_df, x='Model', y='F1-Score', color='Type',
                    title="Model F1-Score Comparison")
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Best Performing Model")
    best_model = max(comparison_data, key=lambda x: x['F1-Score'])
    
    st.metric("Best Model", best_model['Model'])
    st.metric("F1-Score", f"{best_model['F1-Score']:.3f}")
    st.metric("Accuracy", f"{best_model['Accuracy']:.3f}")
    
    st.subheader("Confusion Matrices")
    
    # Placeholder for confusion matrices - you would need to load these from your model results
    st.image("https://via.placeholder.com/600x200?text=Confusion+Matrix+Placeholder", 
             caption="Example Confusion Matrix")

# Footer
st.markdown("---")
st.markdown("""
**Social Media Computing Project**  
Developed by [Your Names]  
[University Name] - [Course Name]
""")