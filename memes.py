import cv2
import pytesseract
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from transformers import pipeline

model = models.resnet50(weights="IMAGENET1K_V1") 
model.eval()

nlp = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

known_memes = {
    "Distracted Boyfriend": "A man looks at another woman while his girlfriend looks on disapprovingly.",
    "Drake Hotline Bling": "A two-panel meme where the first panel shows disapproval and the second panel shows approval.",
    "Expanding Brain": "A multi-panel meme that shows the ironic progression of ideas from supposedly primitive to more advanced"
}

def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray_img)
    return text.strip()

def classify_meme(image_path):
    img = Image.open(image_path)
    img = img.convert('RGB')  
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
    ])

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
    
    _, predicted_class = torch.max(outputs, 1)
    meme_format = str(predicted_class.item())
    return meme_format

def infer_topic_from_text(text):
    if not text.strip():  
        return "Unknown Topic"  
    
    possible_labels = ["politics", "work", "relationships", "life", "technology", "education"]
    result = nlp(text, candidate_labels=possible_labels)
    return result['labels'][0]

def analyze_meme(image_path):
    """Analyze meme and return its topic, sentiment, and meme format."""
    text = extract_text_from_image(image_path)
    meme_format = classify_meme(image_path)
    meme_name = known_memes.get(meme_format, "Unknown Meme Format")
    topic = infer_topic_from_text(text)
    sentiment_result = sentiment_analyzer(text)  
    sentiment = sentiment_result[0]['label'] 
    
    return {
        "meme_name": meme_name,
        "meme_format": meme_format,
        "topic": topic,
        "text": text,
        "sentiment": sentiment 
    }

def analyze_all_memes_in_folder(folder_path):
    """Analyze all memes in the specified folder and return results."""
    meme_results = []
    sentiment_counts = {}  
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".webp") or filename.endswith(".jpeg"): 
            image_path = os.path.join(folder_path, filename)
            print(f"Analyzing: {filename}")
            
            analysis_result = analyze_meme(image_path)  
            meme_results.append({
                "filename": filename,
                "analysis": analysis_result  
            })
            
            meme_format = analysis_result['meme_format']
            topic = analysis_result['topic']
            sentiment = analysis_result['sentiment']
            
            if meme_format not in sentiment_counts:
                sentiment_counts[meme_format] = {}
            if topic not in sentiment_counts[meme_format]:
                sentiment_counts[meme_format][topic] = {'positive': 0, 'negative': 0}
                
            sentiment_counts[meme_format][topic][sentiment.lower()] += 1
    
    return meme_results, sentiment_counts

def calculate_sentiment_percentages(sentiment_counts):
    """Calculate and return the sentiment percentages for each meme format and topic."""
    summary = {}
    
    for meme_format, topics in sentiment_counts.items():
        summary[meme_format] = {}
        for topic, sentiments in topics.items():
            total_sentiments = sentiments['positive'] + sentiments['negative']
            if total_sentiments > 0:
                positive_percentage = (sentiments['positive'] / total_sentiments) * 100
                negative_percentage = (sentiments['negative'] / total_sentiments) * 100
                summary[meme_format][topic] = {
                    'positive_percentage': positive_percentage,
                    'negative_percentage': negative_percentage
                }
            else:
                summary[meme_format][topic] = {
                    'positive_percentage': 0,
                    'negative_percentage': 0
                }
    
    return summary

def calculate_total_sentiment_percentages(sentiment_counts):
    """Calculate and return the total sentiment percentages for each topic regardless of meme format."""
    total_sentiment_counts = {}
    
    # Accumulate sentiment counts by topic across all formats
    for meme_format, topics in sentiment_counts.items():
        for topic, sentiments in topics.items():
            if topic not in total_sentiment_counts:
                total_sentiment_counts[topic] = {'positive': 0, 'negative': 0}
            total_sentiment_counts[topic]['positive'] += sentiments['positive']
            total_sentiment_counts[topic]['negative'] += sentiments['negative']
    
    total_summary = {}
    for topic, sentiments in total_sentiment_counts.items():
        total_sentiments = sentiments['positive'] + sentiments['negative']
        if total_sentiments > 0:
            positive_percentage = (sentiments['positive'] / total_sentiments) * 100
            negative_percentage = (sentiments['negative'] / total_sentiments) * 100
            total_summary[topic] = {
                'positive_percentage': positive_percentage,
                'negative_percentage': negative_percentage
            }
        else:
            total_summary[topic] = {
                'positive_percentage': 0,
                'negative_percentage': 0
            }
    
    return total_summary


memes_folder = input("Please enter the path to the memes folder: ")

memes_analysis, sentiment_counts = analyze_all_memes_in_folder(memes_folder)

for meme in memes_analysis:
    print(f"File: {meme['filename']}")
    print(f"  Meme Name: {meme['analysis']['meme_name']}")
    print(f"  Meme Format: {meme['analysis']['meme_format']}")
    print(f"  Topic: {meme['analysis']['topic']}")
    print(f"  Text: {meme['analysis']['text']}")
    print(f"  Sentiment: {meme['analysis']['sentiment']}") 
    print("-" * 50)

summary = calculate_sentiment_percentages(sentiment_counts)
print("Sentiment Summary:")
for meme_format, topics in summary.items():
    print(f"\nMeme Format: {meme_format}")
    for topic, sentiment_data in topics.items():
        print(f"  Topic: {topic}")
        print(f"    Positive Sentiment: {sentiment_data['positive_percentage']:.2f}%")
        print(f"    Negative Sentiment: {sentiment_data['negative_percentage']:.2f}%")

total_summary = calculate_total_sentiment_percentages(sentiment_counts)
print("\nTotal Sentiment Summary by Topic (Across All Formats):")
for topic, sentiment_data in total_summary.items():
    print(f"  Topic: {topic}")
    print(f"    Positive Sentiment: {sentiment_data['positive_percentage']:.2f}%")
    print(f"    Negative Sentiment: {sentiment_data['negative_percentage']:.2f}%")