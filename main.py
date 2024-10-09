import pandas as pd
import gradio as gr
from transformers import pipeline

# Initialize Sentiment Analysis Model Using Hugging Face's Pipeline
sentiment_analysis = pipeline('sentiment-analysis')

# Positive and Negative Review Counters
pos_count = 0
neg_count = 0

# Process and Analyze the Review Sentiment
def analyze_review(review):
    """
    Analyze the sentiment of a review and update the counters for positive and negative reviews.

    Args:
        review (str): The text of the review to analyze.

    Returns:
        int: Updated count of positive reviews.
        int: Updated count of negative reviews.
    """
    
    global pos_count, neg_count
    sentiment = sentiment_analysis(review)[0]['label']

    if sentiment == "POSITIVE":
        pos_count +=1
    else:
        neg_count +=1
    return pos_count, neg_count

# Define the Gradio Interface
interface = gr.Interface(
    fn=analyze_review,
    inputs=[gr.Textbox(label="Review")],
    outputs=[gr.Textbox(label="Positive Reviews"), gr.Textbox(label="Negative Reviews")],
    live=False,
    title="Sentiment Analysis App",
    description="Analyze the sentiment of reviews and track positive and negative review counts."
)

# Launch the Gradio App
interface.launch()

