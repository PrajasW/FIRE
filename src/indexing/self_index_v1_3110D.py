import os
import pickle
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
import time
import psutil
import csv
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing.preprocess_data import load_news_data, load_wikipedia_data, preprocess_text
from src.querying.query_utils import load_queries

class SelfIndex_v1_3:
    """
    A ranked retrieval system using TF-IDF and Cosine Similarity.
    """
    def __init__(self):
        self.inverted_index = defaultdict(list)  # {term: [(doc_id, raw_tf), ...]}
        self.documents = {}
        self.doc_freq = defaultdict(int)  # DF for each term
        self.num_docs = 0
        self.doc_vectors = {} # Pre-calculated document vectors {doc_id: {term: tf-idf}}

    def build_index(self, articles):
        """
        Builds the inverted index and pre-calculates TF-IDF vectors for all documents.
        """
        self.num_docs = len(articles)
        
        # First pass: Build inverted index and get document frequencies
        for i, article in enumerate(tqdm(articles, desc="Building Index (Pass 1/2)")):
            doc_id = i
            self.documents[doc_id] = {'title': article['title'], 'date': article['published']}
            
            tokens = preprocess_text(article['text']).split()
            term_counts = Counter(tokens)
            
            for term, raw_tf in term_counts.items():
                self.inverted_index[term].append((doc_id, raw_tf))
                self.doc_freq[term] += 1

        # Second pass: Calculate TF-IDF vectors for each document
        for doc_id in tqdm(range(self.num_docs), desc="Building Index (Pass 2/2)"):
            doc_vector = {}
            # Find all terms in the document using the inverted index
            for term, postings in self.inverted_index.items():
                for post_doc_id, raw_tf in postings:
                    if post_doc_id == doc_id:
                        # Calculate weighted TF: 1 + log(TF)
                        tf_w = 1 + np.log(raw_tf)
                        # Calculate IDF: log(N / DF)
                        idf = np.log(self.num_docs / self.doc_freq[term])
                        # Calculate TF-IDF
                        tf_idf = tf_w * idf
                        doc_vector[term] = tf_idf
                        break # Found the term for this doc, move to next term
            self.doc_vectors[doc_id] = doc_vector

    def save_index(self, filepath="self_index_v1_3.pkl"):
        """Saves the index and pre-calculated vectors to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump((self.inverted_index, self.documents, self.doc_freq, self.num_docs, self.doc_vectors), f)
        print(f"Index v1.3 saved to {filepath}")

    def load_index(self, filepath="self_index_v1_3.pkl"):
        """Loads the index and vectors from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.inverted_index, self.documents, self.doc_freq, self.num_docs, self.doc_vectors = pickle.load(f)
            print(f"Index v1.3 loaded from {filepath}")
            return True
        return False

    def query(self, query_str, top_k=10):
        """
        Performs a ranked query using TF-IDF and cosine similarity.
        """
        query_terms = preprocess_text(query_str).split()
        if not query_terms:
            return []

        # 1. Calculate query TF-IDF vector
        query_counts = Counter(query_terms)
        query_vector = {}
        for term, raw_tf in query_counts.items():
            if term in self.doc_freq: # Only consider terms present in the corpus
                tf_w = 1 + np.log(raw_tf)
                idf = np.log(self.num_docs / self.doc_freq[term])
                query_vector[term] = tf_w * idf
        
        if not query_vector:
            return []

        # 2. Calculate Cosine Similarity with all documents
        doc_scores = {}
        query_norm = np.linalg.norm(list(query_vector.values()))

        for doc_id, doc_vector in self.doc_vectors.items():
            # Find common terms to calculate dot product
            common_terms = set(query_vector.keys()).intersection(doc_vector.keys())
            
            if not common_terms:
                continue

            dot_product = sum(query_vector[term] * doc_vector[term] for term in common_terms)
            
            doc_norm = np.linalg.norm(list(doc_vector.values()))

            if doc_norm > 0 and query_norm > 0:
                cosine_sim = dot_product / (query_norm * doc_norm)
                doc_scores[doc_id] = cosine_sim
        
        # 3. Sort documents by score
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        
        return sorted_docs[:top_k]


def plot_latency_comparison(v1_2_metrics_file, v1_3_metrics_file):
    """Generates a plot comparing v1.2 and v1.3 latency."""
    try:
        v1_2_df = pd.read_csv(v1_2_metrics_file)
        v1_3_df = pd.read_csv(v1_3_metrics_file)

        v1_2_latency = v1_2_df[v1_2_df['Metric'] == 'p95_latency_ms']['Value'].iloc[0]
        v1_3_latency = v1_3_df[v1_3_df['Metric'] == 'p95_latency_ms']['Value'].iloc[0]

        plt.figure(figsize=(8, 6))
        
        labels = ['v1.2 (Term-count)', 'v1.3 (TF-IDF)']
        latencies = [v1_2_latency, v1_3_latency]
        
        plt.bar(labels, latencies, color=['#C70039', '#581845'])
        plt.title('p95 Query Latency: Term-count vs. TF-IDF Ranking')
        plt.ylabel('Latency (ms)')
        
        plt.tight_layout()
        plt.savefig('plot_C_x3.png')
        print("Comparison plot 'plot_C_x3.png' generated.")
        plt.close()

    except (FileNotFoundError, IndexError) as e:
        print(f"Could not generate plot: {e}. Make sure all required metric files exist.")


def main():
    index = SelfIndex_v1_3()
    
    start_build_time = time.time()
    if not index.load_index():
        print("Building new index (v1.3)...")
        news_articles = load_news_data('data/data/News_Datasets', num_samples=1000)
        wikipedia_articles = load_wikipedia_data()
        articles = news_articles + wikipedia_articles[:1000]
        
        index.build_index(articles)
        index.save_index()
    build_time = time.time() - start_build_time
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    queries_to_run = load_queries('ranked')

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
    
    print("\n--- SelfIndex-v1.3 Performance Metrics ---")
    print(f"Index Build Time: {build_time:.2f} seconds")
    print(f"Memory Footprint: {memory_mb:.2f} MB")
    print(f"p95 Latency: {p95:.2f} ms")
    print(f"p99 Latency: {p99:.2f} ms")

    metrics_file = 'self_index_v1_3_metrics.csv'
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['build_time_s', build_time])
        writer.writerow(['memory_mb', memory_mb])
        writer.writerow(['p95_latency_ms', p95])
        writer.writerow(['p99_latency_ms', p99])
    print(f"SelfIndex v1.3 metrics saved to {metrics_file}")

    plot_latency_comparison('self_index_v1_2_metrics.csv', metrics_file)


if __name__ == "__main__":
    main()
