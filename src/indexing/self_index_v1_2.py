import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import time
import psutil
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing.preprocess_data import load_news_data, load_wikipedia_data, preprocess_text

class SelfIndex_v1_2:   
    """
    A ranked retrieval system using term frequency (word counts).
    The index structure is the same as v1.1, but the query logic is different.
    """
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.documents = {}
        self.doc_lengths = defaultdict(int)

    def build_index(self, articles):
        """
        Builds the inverted index from a list of articles.
        """
        for i, article in enumerate(tqdm(articles, desc="Building Index v1.2")):
            doc_id = i
            self.documents[doc_id] = {'title': article['title'], 'date': article['published']}
            
            tokens = preprocess_text(article['text']).split()
            self.doc_lengths[doc_id] = len(tokens)
            
            term_positions = defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)
            
            for term, positions in term_positions.items():
                # Store doc_id and term frequency (count) for that doc
                self.inverted_index[term].append((doc_id, len(positions)))

    def save_index(self, filepath="self_index_v1_2.pkl"):
        """Saves the index and documents to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump((self.inverted_index, self.documents, self.doc_lengths), f)
        print(f"Index v1.2 saved to {filepath}")

    def load_index(self, filepath="self_index_v1_2.pkl"):
        """Loads the index and documents from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.inverted_index, self.documents, self.doc_lengths = pickle.load(f)
            print(f"Index v1.2 loaded from {filepath}")
            return True
        return False

    def query(self, query_str, top_k=10):
        """
        Performs a ranked query based on term frequency score.
        Score(doc) = sum over query terms t (TF_t,doc)
        """
        query_terms = preprocess_text(query_str).split()
        if not query_terms:
            return []

        doc_scores = defaultdict(float)
        
        # Find all documents that contain at least one of the query terms
        candidate_docs = set()
        for term in query_terms:
            postings = self.inverted_index.get(term, [])
            for doc_id, _ in postings:
                candidate_docs.add(doc_id)

        # Calculate scores for candidate documents
        for doc_id in candidate_docs:
            score = 0
            for term in query_terms:
                # Find the posting for this term and doc_id
                postings = self.inverted_index.get(term, [])
                for post_doc_id, tf in postings:
                    if post_doc_id == doc_id:
                        score += tf
                        break # Move to the next term
            doc_scores[doc_id] = score
            
        # Sort documents by score in descending order
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        
        return sorted_docs[:top_k]


def plot_latency_comparison(v1_1_metrics_file, v1_2_metrics_file):
    """Generates a plot comparing v1.1 and v1.2 latency."""
    try:
        v1_1_df = pd.read_csv(v1_1_metrics_file)
        v1_2_df = pd.read_csv(v1_2_metrics_file)

        v1_1_latency = v1_1_df[v1_1_df['Metric'] == 'p95_latency_ms']['Value'].iloc[0]
        v1_2_latency = v1_2_df[v1_2_df['Metric'] == 'p95_latency_ms']['Value'].iloc[0]

        # Plotting
        plt.figure(figsize=(8, 6))
        
        labels = ['v1.1 (Positional Boolean)', 'v1.2 (Term-count Ranking)']
        latencies = [v1_1_latency, v1_2_latency]
        
        plt.bar(labels, latencies, color=['#FFC300', '#C70039'])
        plt.title('p95 Query Latency: Boolean vs. Term-count Ranking')
        plt.ylabel('Latency (ms)')
        
        plt.tight_layout()
        plt.savefig('plot_C_x2.png')
        print("Comparison plot 'plot_C_x2.png' generated.")
        plt.close()

    except (FileNotFoundError, IndexError) as e:
        print(f"Could not generate plot: {e}. Make sure both metric files exist.")


def main():
    index = SelfIndex_v1_2()
    
    start_build_time = time.time()
    if not index.load_index():
        print("Building new index (v1.2)...")
        news_articles = load_news_data('data/data/News_Datasets', num_samples=1000)
        wikipedia_articles = load_wikipedia_data()
        articles = news_articles + wikipedia_articles[:1000]
        
        index.build_index(articles)
        index.save_index()
    build_time = time.time() - start_build_time
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # For ranking, we use simpler queries as complex boolean logic isn't the focus
    queries_to_run = {
        "Simple Rank": "world news",
        "Tech Rank": "apple computer technology",
        "Health Rank": "health pandemic virus",
        "Finance Rank": "economy stock market",
        "Politics Rank": "government election policy"
    }

    all_latencies = []
    num_runs = 10
    for i in range(num_runs):
        for q in queries_to_run.values():
            start_time = time.time()
            index.query(q)
            end_time = time.time()
            all_latencies.append((end_time - start_time) * 1000)

    p95 = np.percentile(all_latencies, 95)
    p99 = np.percentile(all_latencies, 99)
    
    print("\n--- SelfIndex-v1.2 Performance Metrics ---")
    print(f"Index Build Time: {build_time:.2f} seconds")
    print(f"Memory Footprint: {memory_mb:.2f} MB")
    print(f"p95 Latency: {p95:.2f} ms")
    print(f"p99 Latency: {p99:.2f} ms")

    metrics_file = 'self_index_v1_2_metrics.csv'
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['build_time_s', build_time])
        writer.writerow(['memory_mb', memory_mb])
        writer.writerow(['p95_latency_ms', p95])
        writer.writerow(['p99_latency_ms', p99])
    print(f"SelfIndex v1.2 metrics saved to {metrics_file}")

    # Generate comparison plot against v1.1
    plot_latency_comparison('self_index_v1_1_metrics.csv', metrics_file)


if __name__ == "__main__":
    main()
