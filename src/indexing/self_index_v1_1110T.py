import os
import pickle
from collections import defaultdict
from tqdm import tqdm
import time
import psutil
import numpy as np
import csv

from src.preprocessing.preprocess_data import load_news_data, load_wikipedia_data, preprocess_text
from src.querying.query_utils import load_queries

class SelfIndex_v1_1110T:
    """
    A simple inverted index with a boolean query model.
    Data structure: { "term": { "doc_id1": [pos1, pos2, ...], ... } }
    """
    def __init__(self):
        self.inverted_index = defaultdict(dict)
        self.documents = {}  # To store doc info like title

    def build_index(self, articles):
        """
        Builds the inverted index from a list of articles.
        """
        for i, article in enumerate(tqdm(articles, desc="Building Index v1.0")):
            doc_id = i
            self.documents[doc_id] = {'title': article['title'], 'date': article['published']}
            
            tokens = preprocess_text(article['text']).split()
            
            term_positions = defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)
            
            for term, positions in term_positions.items():
                self.inverted_index[term][doc_id] = positions

    def save_index(self, filepath="self_index_v1_1110T.pkl"):
        """Saves the index and documents to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump((self.inverted_index, self.documents), f)
        print(f"Index v1.1110T saved to {filepath}")

    def load_index(self, filepath="self_index_v1_1110T.pkl"):
        """Loads the index and documents from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.inverted_index, self.documents = pickle.load(f)
            print(f"Index v1.1110T loaded from {filepath}")
            return True
        return False

    def _intersect(self, docs1, docs2):
        return sorted(list(docs1.intersection(docs2)))

    def _union(self, docs1, docs2):
        return sorted(list(docs1.union(docs2)))

    def _query_phrase(self, phrase):
        terms = preprocess_text(phrase).split()
        if not terms:
            return []

        # Get doc IDs for the first term
        result_docs = set(self.inverted_index.get(terms[0], {}).keys())
        if not result_docs:
            return []

        for i in range(1, len(terms)):
            current_term_docs = self.inverted_index.get(terms[i], {})
            docs_to_check = result_docs.intersection(current_term_docs.keys())
            
            final_docs = set()
            for doc_id in docs_to_check:
                pos1_list = self.inverted_index[terms[i-1]][doc_id]
                pos2_list = current_term_docs[doc_id]
                
                for p1 in pos1_list:
                    if (p1 + 1) in pos2_list:
                        final_docs.add(doc_id)
                        break
            result_docs = final_docs
        
        return sorted(list(result_docs))

    def query(self, query_str):
        if '"' in query_str:
            return self._query_phrase(query_str.replace('"', ''))

        # Preprocess the whole query string
        processed_query = preprocess_text(query_str).split()
        
        # Simple term query if no operators
        if 'and' not in processed_query and 'or' not in processed_query and 'not' not in processed_query:
            if not processed_query: return []
            term = processed_query[0]
            return sorted(list(self.inverted_index.get(term, {}).keys()))

        # Handle boolean logic
        # This is a simplified parser, assuming one operator type per query
        if 'and' in processed_query:
            terms = [t for t in processed_query if t != 'and']
            if not terms: return []
            result_docs = set(self.inverted_index.get(terms[0], {}).keys())
            for term in terms[1:]:
                result_docs.intersection_update(self.inverted_index.get(term, {}).keys())
            return sorted(list(result_docs))

        if 'or' in processed_query:
            terms = [t for t in processed_query if t != 'or']
            if not terms: return []
            result_docs = set(self.inverted_index.get(terms[0], {}).keys())
            for term in terms[1:]:
                result_docs.union_update(self.inverted_index.get(term, {}).keys())
            return sorted(list(result_docs))

        if 'not' in processed_query:
            # Assumes "term1 NOT term2"
            term1, term2 = processed_query[0], processed_query[2]
            docs1 = set(self.inverted_index.get(term1, {}).keys())
            docs2 = set(self.inverted_index.get(term2, {}).keys())
            return sorted(list(docs1 - docs2))

        return []


def main():
    index = SelfIndex_v1_1110T()
    
    start_build_time = time.time()
    if not index.load_index():
        print("Building new index (v1.1110T)...")
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
    total_query_time = 0
    start_query_session = time.time()

    for i in range(num_runs):
        for name, q in queries_to_run.items():
            start_time = time.time()
            results = index.query(q)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            all_latencies.append(latency)
            
            if i == 0:
                print(f"\nQuery: '{q}' ({name}) - Found {len(results)} docs in {latency:.2f} ms.")

    total_query_time = time.time() - start_query_session

    p95 = np.percentile(all_latencies, 95)
    p99 = np.percentile(all_latencies, 99)
    total_queries = len(queries_to_run) * num_runs
    throughput = total_queries / total_query_time if total_query_time > 0 else 0

    print("\n--- SelfIndex-v1.1110T Performance Metrics ---")
    print(f"Index Build Time: {build_time:.2f} seconds")
    print(f"Memory Footprint: {memory_mb:.2f} MB")
    print(f"p95 Latency: {p95:.2f} ms")
    print(f"p99 Latency: {p99:.2f} ms")
    print(f"Throughput: {throughput:.2f} queries/sec")

    metrics_file = 'self_index_v1_1110T_metrics.csv'
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['build_time_s', build_time])
        writer.writerow(['memory_mb', memory_mb])
        writer.writerow(['p95_latency_ms', p95])
        writer.writerow(['p99_latency_ms', p99])
        writer.writerow(['throughput_qps', throughput])
    print(f"SelfIndex v1.0 metrics saved to {metrics_file}")


if __name__ == "__main__":
    main()