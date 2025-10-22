import os
import re
import pickle
from collections import defaultdict
from tqdm import tqdm
import time
import psutil
import numpy as np
import csv

# We need a way to get the raw data, let's modify preprocess_data to expose that
# For now, let's assume a function get_raw_data() exists that gives us title, text, published
from preprocess_data import load_news_data, load_wikipedia_data
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

def preprocess_for_indexing(text):
    """
    Preprocessor that matches the one used for Elasticsearch.
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
    return tokens


class SelfIndex:
    def __init__(self):
        self.inverted_index = defaultdict(list)
        self.documents = {} # To store doc info like title

    def build_index(self, articles):
        """
        Builds the inverted index from a list of articles.
        Each article is a dictionary with 'title', 'text', 'published'.
        """
        for i, article in enumerate(tqdm(articles, desc="Building Index")):
            doc_id = i
            self.documents[doc_id] = {'title': article['title'], 'date': article['published']}
            
            tokens = preprocess_for_indexing(article['text'])
            
            term_positions = defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)
            
            for term, positions in term_positions.items():
                self.inverted_index[term].append((doc_id, positions))

    def save_index(self, filepath="self_index.pkl"):
        """Saves the index and documents to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump((self.inverted_index, self.documents), f)
        print(f"Index saved to {filepath}")

    def load_index(self, filepath="self_index.pkl"):
        """Loads the index and documents from a file."""
        if os.path.exists(filepath):
            with open(filepath, 'rb') as f:
                self.inverted_index, self.documents = pickle.load(f)
            print(f"Index loaded from {filepath}")
            return True
        return False

    def _intersect(self, list1, list2):
        """Helper to intersect two postings lists."""
        # This is a simple merge-based intersection
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
        """Helper to union two postings lists."""
        p1, p2 = 0, 0
        result = []
        while p1 < len(list1) and p2 < len(list2):
            if list1[p1][0] == list2[p2][0]:
                result.append(list1[p1])
                p1 += 1
                p2 += 1
            elif list1[p1][0] < list2[p2][0]:
                result.append(list1[p1])
                p1 += 1
            else:
                result.append(list2[p2])
                p2 += 1
        result.extend(list1[p1:])
        result.extend(list2[p2:])
        return result

    def _query_phrase(self, phrase):
        """Handles a phrase query."""
        # Simple phrase query: "term1 term2"
        terms = preprocess_for_indexing(phrase)
        if not terms:
            return []

        # Get postings for the first term
        result_postings = self.inverted_index.get(terms[0], [])
        
        for i in range(1, len(terms)):
            current_term_postings = self.inverted_index.get(terms[i], [])
            
            temp_result = []
            
            # Intersect the current results with the postings for the next term
            # and check for positional adjacency.
            p1, p2 = 0, 0
            while p1 < len(result_postings) and p2 < len(current_term_postings):
                doc_id1, pos1 = result_postings[p1]
                doc_id2, pos2 = current_term_postings[p2]

                if doc_id1 == doc_id2:
                    # Check if any position in pos2 is one after any position in pos1
                    for p in pos1:
                        if (p + 1) in pos2:
                            temp_result.append((doc_id1, pos2)) # Keep positions of the last term
                            break
                    p1 += 1
                    p2 += 1
                elif doc_id1 < doc_id2:
                    p1 += 1
                else:
                    p2 += 1
            result_postings = temp_result
        
        return [doc_id for doc_id, _ in result_postings]


    def query(self, query_str):
        """
        Performs a boolean or phrase query.
        Supports AND, OR, NOT and phrase queries in quotes.
        NOTE: This is a simplified parser. It doesn't handle complex precedence.
        It processes NOT, then phrases, then AND, then OR.
        """
        
        if '"' in query_str:
            # Assume it's a phrase query
            phrase = query_str.replace('"', '')
            return self._query_phrase(phrase)

        # Handle boolean logic
        if ' and ' in query_str:
            terms = query_str.split(' and ')
            term1, term2 = terms[0], terms[1]
            list1 = self.inverted_index.get(term1.lower(), [])
            list2 = self.inverted_index.get(term2.lower(), [])
            result_postings = self._intersect(list1, list2)
            return [doc_id for doc_id, _ in result_postings]

        if ' or ' in query_str:
            terms = query_str.split(' or ')
            term1, term2 = terms[0], terms[1]
            list1 = self.inverted_index.get(term1.lower(), [])
            list2 = self.inverted_index.get(term2.lower(), [])
            result_postings = self._union(list1, list2)
            return [doc_id for doc_id, _ in result_postings]

        if ' not ' in query_str:
            terms = query_str.split(' not ')
            term1, term2 = terms[0], terms[1]
            list1 = self.inverted_index.get(term1.lower(), [])
            list2_docs = {doc_id for doc_id, _ in self.inverted_index.get(term2.lower(), [])}
            result_postings = [posting for posting in list1 if posting[0] not in list2_docs]
            return [doc_id for doc_id, _ in result_postings]

        # Simple term query
        term = query_str.lower()
        postings = self.inverted_index.get(term, [])
        return [doc_id for doc_id, _ in postings]


def main():
    index = SelfIndex()
    
    # --- Index building and loading ---
    start_build_time = time.time()
    if not index.load_index():
        print("Building new index...")
        news_articles = load_news_data('data/data/News_Datasets', num_samples=1000)
        wikipedia_articles = load_wikipedia_data()
        articles = news_articles + wikipedia_articles[:1000]
        
        index.build_index(articles)
        index.save_index()
    build_time = time.time() - start_build_time 
    
    # --- Memory Footprint ---
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    
    # --- Demo Queries ---
    print("\n--- Running Demo Queries ---")
    queries_to_run = {
        "Simple Term": "news",
        "AND Query": "war AND peace",
        "OR Query": "apple OR google",
        "NOT Query": "world NOT peace",
        "Phrase Query": '"new york"'
    }

    all_latencies = []
    num_runs = 10
    total_query_time = 0

    for i in range(num_runs):
        for name, q in queries_to_run.items():
            start_time = time.time()
            results = index.query(q)
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000
            all_latencies.append(latency)
            
            if i == 0: # Only print results on the first run
                print(f"\nQuery: '{q}' ({name})")
                print(f"Found {len(results)} documents in {latency:.2f} ms.")
                if results:
                    print("Top 5 results:")
                    for doc_id in results[:5]:
                        print(f"  - Doc {doc_id}: {index.documents[doc_id]['title']}")
        total_query_time += (time.time() - start_time)


    # --- Metrics Calculation ---
    p95 = np.percentile(all_latencies, 95)
    p99 = np.percentile(all_latencies, 99)
    total_queries = len(queries_to_run) * num_runs
    throughput = total_queries / total_query_time if total_query_time > 0 else 0

    print("\n--- SelfIndex Performance Metrics ---")
    print(f"Index Build Time: {build_time:.2f} seconds")
    print(f"Memory Footprint: {memory_mb:.2f} MB")
    print(f"p95 Latency: {p95:.2f} ms")
    print(f"p99 Latency: {p99:.2f} ms")
    print(f"Throughput: {throughput:.2f} queries/sec")

    # --- Save to CSV ---
    with open('self_index_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['build_time_s', build_time])
        writer.writerow(['memory_mb', memory_mb])
        writer.writerow(['p95_latency_ms', p95])
        writer.writerow(['p99_latency_ms', p99])
        writer.writerow(['throughput_qps', throughput])
    print("SelfIndex metrics saved to self_index_metrics.csv")


if __name__ == "__main__":
    main()
