import os
import re
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
from src.querying.query_utils import load_queries

class SelfIndex_v1_1:
    """
    An inverted index storing positional information and document/term frequencies.
    Data structure: { term: [(doc_id, [positions]), ...] }
    """
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.documents = {}
        self.doc_freq = defaultdict(int)
        self.term_freq = defaultdict(int)

    def build_index(self, articles):
        """
        Builds the inverted index and calculates frequencies.
        """
        for i, article in enumerate(tqdm(articles, desc="Building Index v1.1")):
            doc_id = i
            self.documents[doc_id] = {'title': article['title'], 'date': article['published']}
            
            tokens = preprocess_text(article['text']).split()
            
            term_positions = defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)
            
            for term, positions in term_positions.items():
                self.inverted_index[term].append((doc_id, positions))
                self.doc_freq[term] += 1
                self.term_freq[term] += len(positions)

    def save_index(self, filepath="self_index_v1_1.pkl"):
        """Saves the index and related stats to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump((self.inverted_index, self.documents, self.doc_freq, self.term_freq), f)
        print(f"Index v1.1 saved to {filepath}")

    def load_index(self, filepath="self_index_v1_1.pkl"):
        """Loads the index and stats from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.inverted_index, self.documents, self.doc_freq, self.term_freq = pickle.load(f)
            print(f"Index v1.1 loaded from {filepath}")
            return True
        return False

    def _intersect(self, list1, list2):
        p1, p2 = 0, 0
        result = []
        while p1 < len(list1) and p2 < len(list2):
            if list1[p1][0] == list2[p2][0]:
                result.append(list1[p1])
                p1 += 1
                p2 += 1
            elif list1[p1][0] < list2[p2][0]:
                p1 += 1
            else:
                p2 += 1
        return result

    def _union(self, list1, list2):
        p1, p2 = 0, 0
        result = []
        while p1 < len(list1) and p2 < len(list2):
            if list1[p1][0] == list2[p2][0]:
                result.append(list1[p1]); p1 += 1; p2 += 1
            elif list1[p1][0] < list2[p2][0]:
                result.append(list1[p1]); p1 += 1
            else:
                result.append(list2[p2]); p2 += 1
        result.extend(list1[p1:])
        result.extend(list2[p2:])
        return result

    def _query_phrase(self, phrase):
        terms = preprocess_text(phrase).split()
        if not terms: return []

        result_postings = self.inverted_index.get(terms[0], [])
        
        for i in range(1, len(terms)):
            current_term_postings = self.inverted_index.get(terms[i], [])
            temp_result = []
            p1, p2 = 0, 0
            while p1 < len(result_postings) and p2 < len(current_term_postings):
                doc_id1, pos1_list = result_postings[p1]
                doc_id2, pos2_list = current_term_postings[p2]

                if doc_id1 == doc_id2:
                    for p1 in pos1_list:
                        if (p1 + 1) in pos2_list:
                            temp_result.append((doc_id1, pos2_list))
                            break
                    p1 += 1; p2 += 1
                elif doc_id1 < doc_id2:
                    p1 += 1
                else:
                    p2 += 1
            result_postings = temp_result
        
        return [doc_id for doc_id, _ in result_postings]

    def query(self, query_str):
        if '"' in query_str:
            return self._query_phrase(query_str.replace('"', ''))

        processed_query = preprocess_text(query_str).split()
        
        if 'and' in processed_query:
            terms = [t for t in processed_query if t != 'and']
            res = self.inverted_index.get(terms[0], [])
            for term in terms[1:]:
                res = self._intersect(res, self.inverted_index.get(term, []))
            return [d[0] for d in res]

        if 'or' in processed_query:
            terms = [t for t in processed_query if t != 'or']
            res = self.inverted_index.get(terms[0], [])
            for term in terms[1:]:
                res = self._union(res, self.inverted_index.get(term, []))
            return [d[0] for d in res]

        if 'not' in processed_query:
            term1, term2 = processed_query[0], processed_query[2]
            list1 = self.inverted_index.get(term1, [])
            list2_docs = {d[0] for d in self.inverted_index.get(term2, [])}
            return [d[0] for d in list1 if d[0] not in list2_docs]

        if not processed_query: return []
        return [d[0] for d in self.inverted_index.get(processed_query[0], [])]

def plot_comparison(es_metrics_file, self_index_metrics_file):
    """Generates a plot comparing ES and SelfIndex metrics."""
    try:
        es_df = pd.read_csv(es_metrics_file)
        si_df = pd.read_csv(self_index_metrics_file)

        es_latency = es_df[es_df['Metric'] == 'p95_latency_ms']['Value'].iloc[0]
        es_mem = es_df[es_df['Metric'] == 'memory_mb']['Value'].iloc[0]

        si_latency = si_df[si_df['Metric'] == 'p95_latency_ms']['Value'].iloc[0]
        si_mem = si_df[si_df['Metric'] == 'memory_mb']['Value'].iloc[0]

        # Plotting
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Latency Plot
        ax1.bar(['Elasticsearch', 'SelfIndex-v1.1'], [es_latency, si_latency], color=['#007ACC', '#FFC300'])
        ax1.set_title('p95 Query Latency Comparison')
        ax1.set_ylabel('Latency (ms)')
        
        # Memory Plot
        ax2.bar(['Elasticsearch', 'SelfIndex-v1.1'], [es_mem, si_mem], color=['#007ACC', '#FFC300'])
        ax2.set_title('Memory Footprint Comparison')
        ax2.set_ylabel('Memory (MB)')
        
        plt.tight_layout()
        plt.savefig('plot_C_x1.png')
        print("Comparison plot 'plot_C_x1.png' generated.")
        plt.close()

    except (FileNotFoundError, IndexError) as e:
        print(f"Could not generate plot: {e}. Make sure both metric files exist.")


def main():
    index = SelfIndex_v1_1()
    
    start_build_time = time.time()
    if not index.load_index():
        print("Building new index (v1.1)...")
        news_articles = load_news_data('data/data/News_Datasets', num_samples=1000)
        wikipedia_articles = load_wikipedia_data()
        articles = news_articles + wikipedia_articles[:1000]
        
        index.build_index(articles)
        index.save_index()
    build_time = time.time() - start_build_time
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    queries_to_run = load_queries('boolean')

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
    
    print("\n--- SelfIndex-v1.1 Performance Metrics ---")
    print(f"Index Build Time: {build_time:.2f} seconds")
    print(f"Memory Footprint: {memory_mb:.2f} MB")
    print(f"p95 Latency: {p95:.2f} ms")
    print(f"p99 Latency: {p99:.2f} ms")

    metrics_file = 'self_index_v1_1_metrics.csv'
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['build_time_s', build_time])
        writer.writerow(['memory_mb', memory_mb])
        writer.writerow(['p95_latency_ms', p95])
        writer.writerow(['p99_latency_ms', p99])
    print(f"SelfIndex v1.1 metrics saved to {metrics_file}")

    # Generate comparison plot
    plot_comparison('baseline_metrics.csv', metrics_file)


if __name__ == "__main__":
    main()
