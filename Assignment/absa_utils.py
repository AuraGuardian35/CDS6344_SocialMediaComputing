# ABSA utilities for dashboard

import spacy
from spacy import displacy
from collections import defaultdict

nlp = spacy.load('en_core_web_lg')

AIRLINE_ASPECTS = {
    'service': ['service', 'staff', 'crew', 'employee', 'attendant', 'agent'],
    'flight': ['flight', 'trip', 'journey', 'travel'],
    'delay': ['delay', 'late', 'cancellation', 'cancel', 'on time', 'early'],
    'baggage': ['baggage', 'luggage', 'bag', 'suitcase'],
    'food': ['food', 'meal', 'snack', 'beverage', 'drink'],
    'comfort': ['seat', 'comfort', 'legroom', 'space', 'recline'],
    'entertainment': ['entertainment', 'movie', 'tv', 'screen', 'music'],
    'price': ['price', 'cost', 'fare', 'expensive', 'cheap'],
    'booking': ['booking', 'reservation', 'checkin', 'check-in', 'website', 'app'],
    'safety': ['safety', 'clean', 'hygiene', 'mask', 'covid']
}

def extract_aspects(text, nlp_model=nlp):
    """Extract aspects from text using noun phrases and predefined aspects"""
    doc = nlp_model(text)
    
    aspects = set()
    
    # Extract noun phrases
    for chunk in doc.noun_chunks:
        if chunk.root.pos_ in ['NOUN', 'PROPN'] and len(chunk.text) > 2:
            aspects.add(chunk.text.lower())
    
    # Extract predefined aspects
    for aspect, keywords in AIRLINE_ASPECTS.items():
        for keyword in keywords:
            if keyword in text.lower():
                aspects.add(aspect)
    
    return list(aspects)

def analyze_aspect_sentiment(text, aspect):
    """Simple sentiment analysis for aspects"""
    positive_words = ['good', 'great', 'excellent', 'awesome', 'fantastic', 
                     'wonderful', 'amazing', 'love', 'like', 'happy']
    negative_words = ['bad', 'poor', 'terrible', 'horrible', 'awful',
                     'disappointing', 'hate', 'worst', 'unhappy']
    
    aspect_pos = text.lower().find(aspect.lower())
    if aspect_pos == -1:
        return 1  # Neutral
    
    window = text[max(0, aspect_pos-30):min(len(text), aspect_pos+30)]
    pos_count = sum(1 for word in positive_words if word in window.lower())
    neg_count = sum(1 for word in negative_words if word in window.lower())
    
    if pos_count > neg_count:
        return 2  # Positive
    elif neg_count > pos_count:
        return 0  # Negative
    else:
        return 1  # Neutral

def visualize_aspect_sentiment(text):
    """Visualize aspect sentiment for a given text"""
    doc = nlp(text)
    aspects = extract_aspects(text)
    
    print(f"Text: {text}")
    print("\nIdentified Aspects:")
    for aspect in aspects:
        sentiment = analyze_aspect_sentiment(text, aspect)
        print(f"- {aspect}: {['Negative', 'Neutral', 'Positive'][sentiment]}")
    
    colors = {"ASPECT": "linear-gradient(90deg, #aa9cfc, #fc9ce7)"}
    options = {"ents": ["ASPECT"], "colors": colors}
    
    ents = []
    for aspect in aspects:
        start = text.lower().find(aspect.lower())
        if start != -1:
            end = start + len(aspect)
            ents.append({"start": start, "end": end, "label": "ASPECT"})
    
    doc.ents = [doc.char_span(e["start"], e["end"], label=e["label"]) for e in ents]
    displacy.render(doc, style="ent", options=options, jupyter=True)
