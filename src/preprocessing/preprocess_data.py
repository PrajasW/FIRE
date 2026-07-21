import os
import json
import glob
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import string
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from tqdm import tqdm
import re

# Download NLTK data
# nltk.download('punkt')
# nltk.download('stopwords')

def load_news_data(data_path, num_samples=1000):
    """
    Load news data from the specified path. Each file is a single JSON object.
    """
    articles = []
    count = 0
    json_files = glob.glob(os.path.join(data_path, '**', '*.json'), recursive=True)
    
    for filepath in tqdm(json_files, desc="Loading news data"):
        if count >= num_samples:
            break
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                article = json.load(f)
                if article.get('language', '').lower() == 'english':
                    # Return a dictionary to keep title and date
                    articles.append({
                        'title': article.get('title', ''),
                        'text': article.get('text', ''),
                        'published': article.get('published', '')
                    })
                    count += 1
        except Exception as e:
            # print(f"Error reading or processing file {filepath}: {e}")
            continue
    return articles

def load_wikipedia_data():
    """
    Load Wikipedia data.
    """
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train[:1%]")
    # Return a dictionary to keep title and date (using url as title, and no date)
    articles = [{'title': item['url'], 'text': item['text'], 'published': None} for item in ds]
    return articles

def preprocess_text(text):
    """
    Preprocess a single text document.
    """
    # Remove URLs
    text = re.sub(r'http\S+|https\S+', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Lowercasing
    tokens = [word.lower() for word in tokens]
    # Punctuation and symbols removal
    tokens = [word for word in tokens if word.isalpha()]
    # Remove single-character tokens
    tokens = [word for word in tokens if len(word) > 1]
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

def get_preprocessed_data():
    """
    Loads and preprocesses the data, returning it for other scripts.
    """
    news_articles = load_news_data('data/data/News_Datasets', num_samples=1000)
    wikipedia_articles = load_wikipedia_data()
    
    articles = news_articles + wikipedia_articles[:1000]

    processed_data = []
    for article in tqdm(articles, desc="Preprocessing articles"):
        processed_data.append({
            'title': article['title'],
            'body': preprocess_text(article['text']),
            'date': article['published']
        })
    return processed_data


def plot_word_frequency(words, filename):
    """
    Plot and save word frequency distribution.
    """
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(20)
    words, counts = zip(*most_common_words)

    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 20 Word Frequencies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    # This main function is now for generating the plots and report
    # The actual data loading and preprocessing can be imported by other scripts
    
    news_articles = load_news_data('data/data/News_Datasets', num_samples=1000)
    wikipedia_articles = load_wikipedia_data()
    
    articles = news_articles + wikipedia_articles[:1000]

    # Raw word frequency
    raw_words = []
    for article in tqdm(articles, desc="Generating raw word frequency"):
        raw_words.extend(word_tokenize(article['text'].lower()))
    
    # Filter out punctuation for a cleaner raw word plot
    raw_words_filtered = [word for word in raw_words if word.isalpha()]
    plot_word_frequency(raw_words_filtered, 'raw_word_freq.png')

    # Preprocess data for word frequency plot
    processed_articles_text = [preprocess_text(article['text']) for article in tqdm(articles, desc="Preprocessing for plot")]
    
    # Processed word frequency
    processed_words = [word for text in processed_articles_text for word in text.split()]
    plot_word_frequency(processed_words, 'processed_word_freq.png')

    # Create a markdown report
    with open('data_preprocessing_report.md', 'w') as f:
        f.write("# Data Preprocessing Report\n\n")
        f.write("This report details the preprocessing of news and Wikipedia data.\n\n")
        f.write("## Data Sources\n")
        f.write("- **News Data**: 1000 English articles from the webz.io dataset.\n")
        f.write("- **Wikipedia Data**: 1000 articles from the 'wikimedia/wikipedia' dataset ('20231101.en' split).\n\n")
        f.write("## Preprocessing Steps\n")
        f.write("1. **URL Removal**: Removing http/https links.\n")
        f.write("2. **Tokenization**: Splitting text into individual words.\n")
        f.write("3. **Lowercasing**: Converting all text to lowercase.\n")
        f.write("4. **Punctuation & Symbol Removal**: Removing non-alphabetic characters.\n")
        f.write("5. **Single-Character Removal**: Removing tokens of length 1.\n")
        f.write("6. **Stopword Removal**: Removing common English stopwords.\n")
        f.write("7. **Stemming**: Reducing words to their root form using PorterStemmer.\n\n")
        f.write("## Word Frequency Plots\n")
        f.write("### Before Preprocessing\n")
        f.write("![Raw Word Frequency](raw_word_freq.png)\n\n")
        f.write("### After Preprocessing\n")
        f.write("![Processed Word Frequency](processed_word_freq.png)\n")


if __name__ == '__main__':
    main()
