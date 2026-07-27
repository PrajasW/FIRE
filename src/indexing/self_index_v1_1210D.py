# Prerequisites:
# pip install psycopg2-binary redis pandas matplotlib
#
# Assumes a running PostgreSQL instance with a database named 'ir_project'.
# You may need to adjust the DB_PARAMS dictionary to match your setup.
# CREATE DATABASE ir_project;
#
# Assumes a running Redis instance on localhost:6379.

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
import psycopg2
import redis
import shutil

from src.preprocessing.preprocess_data import load_news_data, load_wikipedia_data, preprocess_text
from src.querying.query_utils import load_queries

# --- PostgreSQL Connection Parameters ---
# !!! IMPORTANT: Adjust these to your local PostgreSQL setup !!!
DB_PARAMS = {
    "dbname": "ire_1",
    "user": "postgres",
    "password": "root@1234", # Or your password
    "host": "localhost",
    "port": 5432
}
POSTGRES_TABLE_NAME = "inverted_index_y2"

# --- Redis Connection Parameters ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379

class PostgresIndex:
    """Handles indexing and querying using PostgreSQL."""
    def __init__(self, db_params):
        self.db_params = db_params
        self._create_table()

    def _get_conn(self):
        return psycopg2.connect(**self.db_params)

    def _create_table(self):
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {POSTGRES_TABLE_NAME} (
                    term TEXT PRIMARY KEY,
                    postings JSONB
                );
                """)
                
                # Clear the table for a fresh build
                cur.execute(f"TRUNCATE TABLE {POSTGRES_TABLE_NAME};")

    def build_and_save(self, articles):
        """Builds the index and inserts it into PostgreSQL."""
        inverted_index = defaultdict(list)
        for i, article in enumerate(tqdm(articles, desc="Building Temp Index for PG")):
            doc_id = i
            tokens = preprocess_text(article['text']).split()
            term_positions = defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)
            for term, positions in term_positions.items():
                inverted_index[term].append((doc_id, positions))
        
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                for term, postings in tqdm(inverted_index.items(), desc="Inserting into PostgreSQL"):
                    # Convert postings to JSON string
                    postings_json = json.dumps(postings)
                    cur.execute(
                        f"INSERT INTO {POSTGRES_TABLE_NAME} (term, postings) VALUES (%s, %s)",
                        (term, postings_json)
                    )

    def query(self, query_str):
        """Queries PostgreSQL to retrieve postings and then processes them."""
        processed_query = preprocess_text(query_str).split()
        if not processed_query: return []
        
        terms = [t for t in processed_query if t != 'and']
        if not terms: return []

        postings_lists = []
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                for term in terms:
                    cur.execute(f"SELECT postings FROM {POSTGRES_TABLE_NAME} WHERE term = %s", (term,))
                    result = cur.fetchone()
                    if result:
                        postings_lists.append(result[0]) # result[0] is the JSONB content
                    else:
                        return [] # If any term is missing, intersection is empty
        
        # Perform intersection in Python
        res = postings_lists[0]
        for i in range(1, len(postings_lists)):
            res = self._intersect(res, postings_lists[i])
        return [d[0] for d in res]

    def _intersect(self, list1, list2):
        # Standard intersection logic from previous versions
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

    def get_disk_size(self):
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(f"SELECT pg_total_relation_size('{POSTGRES_TABLE_NAME}');")
                size_bytes = cur.fetchone()[0]
                return size_bytes / (1024 * 1024)

class RedisIndex:
    """Handles indexing and querying using Redis."""
    def __init__(self, redis_host, redis_port):
        self.r = redis.Redis(host=redis_host, port=redis_port, db=0)
        self.r.flushdb() # Clear the database for a fresh build

    def build_and_save(self, articles):
        """Builds the index and inserts it into Redis."""
        inverted_index = defaultdict(list)
        for i, article in enumerate(tqdm(articles, desc="Building Temp Index for Redis")):
            doc_id = i
            tokens = preprocess_text(article['text']).split()
            term_positions = defaultdict(list)
            for pos, term in enumerate(tokens):
                term_positions[term].append(pos)
            for term, positions in term_positions.items():
                inverted_index[term].append((doc_id, positions))

        for term, postings in tqdm(inverted_index.items(), desc="Inserting into Redis"):
            self.r.set(term.encode('utf-8'), pickle.dumps(postings))

    def query(self, query_str):
        """Queries Redis to retrieve postings and then processes them."""
        processed_query = preprocess_text(query_str).split()
        if not processed_query: return []
        
        terms = [t for t in processed_query if t != 'and']
        if not terms: return []

        postings_lists = []
        for term in terms:
            result = self.r.get(term.encode('utf-8'))
            if result:
                postings_lists.append(pickle.loads(result))
            else:
                return []

        if not postings_lists:
            return []
            
        res = postings_lists[0]
        for i in range(1, len(postings_lists)):
            res = self._intersect(res, postings_lists[i])
        return [d[0] for d in res]

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

    def get_disk_size(self):
        # Redis stores data in memory, so this reflects memory usage.
        # The on-disk size (from RDB/AOF) can be different.
        return self.r.info('memory')['used_memory'] / (1024 * 1024)


def run_performance_test(datastore_type, articles):
    """Runs a full build, save, and query test for a given datastore."""
    metrics = {'datastore': datastore_type}
    index = None

    # --- Build and Save Test ---
    start_build = time.time()
    if datastore_type == 'postgres':
        index = PostgresIndex(DB_PARAMS)
        index.build_and_save(articles)
    elif datastore_type == 'redis':
        index = RedisIndex(REDIS_HOST, REDIS_PORT)
        index.build_and_save(articles)
    else: # fallback or error
        print(f"Unknown datastore type: {datastore_type}")
        return {}
        
    metrics['build_time_s'] = time.time() - start_build
    metrics['index_size_mb'] = index.get_disk_size()

    # --- Memory and Query Test ---
    process = psutil.Process(os.getpid())
    metrics['memory_mb'] = process.memory_info().rss / (1024 * 1024)
    
    queries_to_run = load_queries('and_only')

    all_latencies = []
    num_runs = 20
    total_query_time = 0
    start_query_session = time.time()

    for _ in range(num_runs):
        for q in queries_to_run.values():
            start_q_time = time.time()
            index.query(q)
            end_q_time = time.time()
            all_latencies.append((end_q_time - start_q_time) * 1000)
    
    total_query_time = time.time() - start_query_session
    total_queries = len(queries_to_run) * num_runs

    metrics['p95_latency_ms'] = np.percentile(all_latencies, 95)
    metrics['throughput_qps'] = total_queries / total_query_time if total_query_time > 0 else 0
    
    print(f"\n--- Results for {datastore_type.upper()} ---")
    print(f"Build Time: {metrics['build_time_s']:.2f} s")
    print(f"Index Size: {metrics['index_size_mb']:.2f} MB")
    print(f"p95 Latency: {metrics['p95_latency_ms']:.4f} ms")
    print(f"Throughput: {metrics['throughput_qps']:.2f} qps")
    
    return metrics

def plot_comparison(results_df):
    """Generates bar plots comparing metrics for different datastores."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('External Datastore Performance Comparison (v1.y2)', fontsize=16)
    
    datastores = results_df['datastore']
    
    # Plot 1: Build Time
    ax = axes[0, 0]
    ax.bar(datastores, results_df['build_time_s'], color=['#007ACC', '#FFC300'])
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Index Build Time')

    # Plot 2: Index Size on Disk
    ax = axes[0, 1]
    ax.bar(datastores, results_df['index_size_mb'], color=['#008080', '#FFA07A'])
    ax.set_ylabel('Size (MB)')
    ax.set_title('Index Size on Disk')

    # Plot 3: p95 Query Latency
    ax = axes[1, 0]
    ax.bar(datastores, results_df['p95_latency_ms'], color=['#4682B4', '#D2691E'])
    ax.set_ylabel('Latency (ms)')
    ax.set_title('p95 Query Latency')

    # Plot 4: Throughput
    ax = axes[1, 1]
    ax.bar(datastores, results_df['throughput_qps'], color=['#5F9EA0', '#CD5C5C'])
    ax.set_ylabel('Queries/sec')
    ax.set_title('Throughput')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('plot_A_y2.png')
    print("\nComparison plot 'plot_A_y2.png' generated.")
    plt.close()

def main():
    print("Loading articles into memory...")
    news_articles = load_news_data('data/data/News_Datasets', num_samples=1000)
    wikipedia_articles = load_wikipedia_data()
    articles = news_articles + wikipedia_articles[:1000]
    
    all_metrics = []
    
    # Run tests for PostgreSQL
    try:
        pg_metrics = run_performance_test('postgres', articles)
        all_metrics.append(pg_metrics)
    except Exception as e:
        print("\n---!!!---")
        print(f"Could not run PostgreSQL test. Error: {e}")
        print("Please ensure PostgreSQL is running and the DB_PARAMS are correct.")
        print("---!!!---\n")

    # Run tests for Redis
    try:
        redis_metrics = run_performance_test('redis', articles)
        all_metrics.append(redis_metrics)
    except Exception as e:
        print(f"Could not run Redis test. Error: {e}")

    if not all_metrics:
        print("No tests were successfully run. Exiting.")
        return

    # Save results to CSV and plot
    results_df = pd.DataFrame(all_metrics)
    metrics_file = 'self_index_v1_y2_metrics.csv'
    results_df.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to {metrics_file}")

    if len(all_metrics) > 1:
        plot_comparison(results_df)
    else:
        print("Only one test completed. Skipping plot generation.")


if __name__ == "__main__":
    main()
