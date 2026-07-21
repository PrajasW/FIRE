import os
import pickle
import json
from collections import defaultdict
from tqdm import tqdm
import time
import psutil
import numpy as np
import csv
import matplotlib.pyplot as plt
import pandas as pd

from src.preprocessing.preprocess_data import load_news_data, load_wikipedia_data, preprocess_text

class SelfIndex_v1_y1:
    """
    A boolean index to test persistence formats. Based on v1.1.
    Data structure: { term: [(doc_id, [positions]), ...] }
    """
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.documents = {}

    def build_index(self, articles):
        """
        Builds the inverted index from a list of articles.
        """
        for i, article in enumerate(tqdm(articles, desc="Building Index")):
            doc_id = i
            self.documents[doc_id] = {'title': article['title'], 'date': article['published']}
            
            tokens = preprocess_text(article['text']).split()
            
            term_positions = defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)
            
            for term, positions in term_positions.items():
                self.inverted_index[term].append((doc_id, positions))

    # --- Pickle Persistence ---
    def save_index_pickle(self, filepath="self_index_v1_y1.pkl"):
        with open(filepath, 'wb') as f:
            pickle.dump((self.inverted_index, self.documents), f)

    def load_index_pickle(self, filepath="self_index_v1_y1.pkl"):
        with open(filepath, 'rb') as f:
            self.inverted_index, self.documents = pickle.load(f)

    # --- JSON Persistence ---
    def save_index_json(self, filepath="self_index_v1_y1.json"):
        # Convert defaultdict to dict for JSON serialization
        serializable_index = dict(self.inverted_index)
        with open(filepath, 'w') as f:
            json.dump((serializable_index, self.documents), f)

    def load_index_json(self, filepath="self_index_v1_y1.json"):
        with open(filepath, 'r') as f:
            serializable_index, self.documents = json.load(f)
            # Convert back to defaultdict
            self.inverted_index = defaultdict(list, serializable_index)

    # --- Query Logic (from v1.1) ---
    def _intersect(self, list1, list2):
        p1, p2 = 0, 0
        result = []
        while p1 < len(list1) and p2 < len(list2):
            if list1[p1][0] == list2[p2][0]:
                result.append(list1[p1]); p1 += 1; p2 += 1
            elif list1[p1][0] < list2[p2][0]:
                p1 += 1
            else:
                p2 += 1
        return result

    def query(self, query_str):
        # Simplified query logic for performance testing
        processed_query = preprocess_text(query_str).split()
        if not processed_query: return []
        
        # For simplicity, we'll just test intersection (AND)
        terms = [t for t in processed_query if t != 'and']
        if not terms: return []

        res = self.inverted_index.get(terms[0], [])
        for term in terms[1:]:
            res = self._intersect(res, self.inverted_index.get(term, []))
        return [d[0] for d in res]


def run_performance_test(index, format_type):
    """
    Runs a full save, load, and query test for a given persistence format.
    """
    metrics = {'format': format_type}
    
    # --- Save Test ---
    save_path = f"self_index_v1_y1.{'pkl' if format_type == 'pickle' else 'json'}"
    start_save = time.time()
    if format_type == 'pickle':
        index.save_index_pickle(save_path)
    else:
        index.save_index_json(save_path)
    metrics['save_time_s'] = time.time() - start_save
    metrics['index_size_mb'] = os.path.getsize(save_path) / (1024 * 1024)

    # --- Load Test ---
    # Create a new empty index to load into
    load_index = SelfIndex_v1_y1()
    start_load = time.time()
    if format_type == 'pickle':
        load_index.load_index_pickle(save_path)
    else:
        load_index.load_index_json(save_path)
    metrics['load_time_s'] = time.time() - start_load

    # --- Memory and Query Test ---
    process = psutil.Process(os.getpid())
    metrics['memory_mb'] = process.memory_info().rss / (1024 * 1024)
    
    queries_to_run = {
        "AND Query 1": "war and peace",
        "AND Query 2": "apple and computer",
        "AND Query 3": "global and warming",
        "AND Query 4": "election and results",
        "AND Query 5": "health and pandemic"
    }

    all_latencies = []
    num_runs = 20 # More runs for stable latency measurement
    total_query_time = 0
    start_query_session = time.time()

    for _ in range(num_runs):
        for q in queries_to_run.values():
            start_q_time = time.time()
            load_index.query(q)
            end_q_time = time.time()
            all_latencies.append((end_q_time - start_q_time) * 1000)
    
    total_query_time = time.time() - start_query_session
    total_queries = len(queries_to_run) * num_runs

    metrics['p95_latency_ms'] = np.percentile(all_latencies, 95)
    metrics['throughput_qps'] = total_queries / total_query_time if total_query_time > 0 else 0
    
    print(f"\n--- Results for {format_type.upper()} ---")
    print(f"Save Time: {metrics['save_time_s']:.4f} s")
    print(f"Load Time: {metrics['load_time_s']:.4f} s")
    print(f"Index Size: {metrics['index_size_mb']:.2f} MB")
    print(f"Memory Footprint: {metrics['memory_mb']:.2f} MB")
    print(f"p95 Latency: {metrics['p95_latency_ms']:.4f} ms")
    print(f"Throughput: {metrics['throughput_qps']:.2f} qps")

    # Clean up the created index file
    # os.remove(save_path)
    
    return metrics

def plot_comparison(results_df):
    """Generates bar plots comparing metrics for different persistence formats."""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Persistence Format Performance Comparison (v1.y1)', fontsize=16)
    
    formats = results_df['format']
    
    # Plot 1: Load and Save Time
    ax = axes[0, 0]
    width = 0.35
    x = np.arange(len(formats))
    rects1 = ax.bar(x - width/2, results_df['save_time_s'], width, label='Save Time')
    rects2 = ax.bar(x + width/2, results_df['load_time_s'], width, label='Load Time')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Index Load and Save Time')
    ax.set_xticks(x)
    ax.set_xticklabels(formats)
    ax.legend()

    # Plot 2: Index Size on Disk
    ax = axes[0, 1]
    ax.bar(formats, results_df['index_size_mb'], color=['#008080', '#FFA07A'])
    ax.set_ylabel('Size (MB)')
    ax.set_title('Index Size on Disk')

    # Plot 3: p95 Query Latency
    ax = axes[1, 0]
    ax.bar(formats, results_df['p95_latency_ms'], color=['#4682B4', '#D2691E'])
    ax.set_ylabel('Latency (ms)')
    ax.set_title('p95 Query Latency')

    # Plot 4: Memory Footprint
    ax = axes[1, 1]
    ax.bar(formats, results_df['memory_mb'], color=['#5F9EA0', '#CD5C5C'])
    ax.set_ylabel('Memory (MB)')
    ax.set_title('Memory Footprint After Load')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('plot_A_y1.png')
    print("\nComparison plot 'plot_A_y1.png' generated.")
    plt.close()


def main():
    # Build the index once in memory
    base_index = SelfIndex_v1_y1()
    news_articles = load_news_data('data/data/News_Datasets', num_samples=1000)
    wikipedia_articles = load_wikipedia_data()
    articles = news_articles + wikipedia_articles[:1000]
    base_index.build_index(articles)
    
    # Run tests for both formats
    pickle_metrics = run_performance_test(base_index, 'pickle')
    json_metrics = run_performance_test(base_index, 'json')
    
    # Save results to CSV
    results_df = pd.DataFrame([pickle_metrics, json_metrics])
    metrics_file = 'self_index_v1_y1_metrics.csv'
    results_df.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to {metrics_file}")

    # Generate comparison plot
    plot_comparison(results_df)


if __name__ == "__main__":
    main()
