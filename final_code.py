import openai
import pandas as pd
import re
import matplotlib.pyplot as plt

OPENAI_API_KEY = "your api key"
openai.api_key = OPENAI_API_KEY

def perform_sentiment_analysis(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            temperature=0.5, 
            max_tokens=100,
        )
        response_text = response.choices[0].text.strip()
        return response_text
    except Exception as e:
        print(f"Error occurred during sentiment analysis: {e}")
        return ""

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

def process_csv(input_file):
    aspects = ['Food', 'Hygiene', 'Staff', 'Ambience', 'Others']
    aspect_sentiment_counts = {aspect: {'Positive': 0, 'Negative': 0, 'Neutral': 0} for aspect in aspects}
    overall_sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
    
    try:
        df = pd.read_csv(input_file, usecols=[0]) 
        for feedback_text in df.iloc[:, 0]:
            feedback_text = str(feedback_text) 
            print(f"Processing line: {feedback_text}")
    
            try:
                cleaned_line = preprocess_text(feedback_text)
                
                prompt = f"For a restaurant review that reads '{cleaned_line}', what is the main aspect being discussed? Choose from Food, Hygiene, Staff, Ambience, Others. Additionally, mention whether the sentiment is Positive, Negative, or Neutral.\n"
                response = perform_sentiment_analysis(prompt)
                aspect_response = response.strip().lower()

                aspect = 'Others'
                if 'food' in aspect_response:
                    aspect = 'Food'
                elif 'staff' in aspect_response:
                    aspect = 'Staff'
                elif 'hygiene' in aspect_response:
                    aspect = 'Hygiene'
                elif 'ambience' in aspect_response:
                    aspect = 'Ambience'

                sentiment = 'Neutral'
                if 'positive' in aspect_response:
                    sentiment = 'Positive'
                elif 'negative' in aspect_response:
                    sentiment = 'Negative'

                aspect_sentiment_counts[aspect][sentiment] += 1

                overall_sentiment_counts[sentiment] += 1

                print(f"Sentiment: {sentiment}")
                print(f"Aspect: {aspect}\n")
        
            except Exception as e:
                print(f"Error occurred while processing line: {feedback_text}, Error: {e}")

       
        for aspect, sentiment_counts in aspect_sentiment_counts.items():
            plt.figure(figsize=(8, 6))
            plt.bar(sentiment_counts.keys(), sentiment_counts.values(), color='lightgreen')
            plt.title(f'{aspect} Sentiment Distribution')
            plt.xlabel('Sentiment')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.show()

     
        plt.figure(figsize=(8, 6))
        plt.bar(overall_sentiment_counts.keys(), overall_sentiment_counts.values(), color='lightblue')
        plt.title('Overall Sentiment Distribution')
        plt.xlabel('Sentiment')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()

    except FileNotFoundError:
        print(f"File not found: {input_file}")

if __name__ == "__main__":
    input_file = "/Users/rahulkanwar/Desktop/IOCL/final project/Restaurant_Reviews.csv"
    process_csv(input_file)
