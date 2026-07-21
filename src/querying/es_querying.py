from elasticsearch import Elasticsearch
import time
import numpy as np
import matplotlib.pyplot as plt
import csv

def run_queries(es_client, index_name):
    """
    Runs a set of queries and measures performance.
    """
    queries = {
        "simple_term": {"query": {"term": {"body": "news"}}},
        "phrase_query": {"query": {"match_phrase": {"body": "new york"}}},
        "boolean_and": {"query": {"bool": {"must": [{"term": {"body": "war"}}, {"term": {"body": "peace"}}]}}},
        "boolean_or": {"query": {"bool": {"should": [{"term": {"body": "apple"}}, {"term": {"body": "google"}}]}}},
        "boolean_not": {"query": {"bool": {"must_not": [{"term": {"body": "politics"}}]}}}
    }

    latencies = []
    for name, query in queries.items():
        start_time = time.time()
        es_client.search(index=index_name, body=query)
        end_time = time.time()
        latency = (end_time - start_time) * 1000  # in ms
        latencies.append(latency)
        print(f"Query '{name}' took {latency:.2f} ms")

    return latencies

def calculate_metrics(latencies, num_queries, duration):
    """
    Calculates and prints performance metrics.
    """
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    throughput = num_queries / duration if duration > 0 else 0

    print(f"\n--- Performance Metrics ---")
    print(f"p95 Latency: {p95:.2f} ms")
    print(f"p99 Latency: {p99:.2f} ms")
    print(f"Throughput: {throughput:.2f} queries/sec")
    
    return p95, p99, throughput

def get_memory_footprint(es_client):
    """
    Gets the memory footprint of the Elasticsearch cluster.
    """
    stats = es_client.nodes.stats(metric='jvm')
    total_heap_used = 0
    for node in stats['nodes'].values():
        total_heap_used += node['jvm']['mem']['heap_used_in_bytes']
    memory_mb = total_heap_used / (1024*1024)
    print(f"Memory Footprint (Heap Used): {memory_mb:.2f} MB")
    return memory_mb

def plot_metrics(p95, p99, throughput, memory_mb):
    """
    Generates and saves plots for the metrics.
    """
    # Latency plot
    plt.figure(figsize=(8, 5))
    plt.bar(['p95', 'p99'], [p95, p99], color=['blue', 'orange'])
    plt.ylabel('Latency (ms)')
    plt.title('Query Latency')
    plt.savefig('latency_baseline.png')
    plt.close()

    # Throughput plot
    plt.figure(figsize=(8, 5))
    plt.bar(['Throughput'], [throughput], color='green')
    plt.ylabel('Queries/sec')
    plt.title('Query Throughput')
    plt.savefig('throughput_baseline.png')
    plt.close()

    # Memory plot
    plt.figure(figsize=(8, 5))
    plt.bar(['Memory Usage'], [memory_mb], color='purple')
    plt.ylabel('Memory (MB)')
    plt.title('Elasticsearch Heap Memory Usage')
    plt.savefig('memory_baseline.png')
    plt.close()

def save_metrics_to_csv(p95, p99, throughput, memory_mb):
    """
    Saves the performance metrics to a CSV file.
    """
    with open('baseline_metrics.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Metric', 'Value'])
        writer.writerow(['p95_latency_ms', p95])
        writer.writerow(['p99_latency_ms', p99])
        writer.writerow(['throughput_qps', throughput])
        writer.writerow(['memory_mb', memory_mb])
    print("Metrics saved to baseline_metrics.csv")

def main():
    es_client = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    if not es_client.ping():
        raise ConnectionError("Could not connect to Elasticsearch")

    index_name = "articles_index"
    
    total_duration = 0
    all_latencies = []
    
    # Run queries multiple times to get a better average
    num_runs = 10
    for i in range(num_runs):
        print(f"\n--- Running query set {i+1}/{num_runs} ---")
        start_run = time.time()
        latencies = run_queries(es_client, index_name)
        end_run = time.time()
        
        all_latencies.extend(latencies)
        total_duration += (end_run - start_run)

    p95, p99, throughput = calculate_metrics(all_latencies, len(all_latencies), total_duration)
    memory_mb = get_memory_footprint(es_client)
    
    plot_metrics(p95, p99, throughput, memory_mb)
    save_metrics_to_csv(p95, p99, throughput, memory_mb)

    # Create a markdown report
    with open('ESIndex-v1.0-report.md', 'w') as f:
        f.write("# Elasticsearch Baseline Performance Report (v1.0)\n\n")
        f.write("This report details the baseline performance of our Elasticsearch index.\n\n")
        f.write("## Index Schema\n")
        f.write("The index uses a simple schema with `title`, `body`, and `date` fields.\n\n")
        f.write("## Query Performance\n")
        f.write(f"* **p95 Latency**: {p95:.2f} ms\n")
        f.write(f"* **p99 Latency**: {p99:.2f} ms\n")
        f.write(f"* **Throughput**: {throughput:.2f} queries/sec\n")
        f.write(f"* **Memory Footprint**: {memory_mb:.2f} MB\n\n")
        f.write("Metrics have also been saved to `baseline_metrics.csv`.\n\n")
        f.write("## Performance Plots\n")
        f.write("### Latency\n")
        f.write("![Latency Baseline](latency_baseline.png)\n\n")
        f.write("### Throughput\n")
        f.write("![Throughput Baseline](throughput_baseline.png)\n\n")
        f.write("### Memory\n")
        f.write("![Memory Baseline](memory_baseline.png)\n\n")
        f.write("## Precision/Recall\n")
        f.write("Manual evaluation of precision and recall was not automated for this baseline.\n")


if __name__ == "__main__":
    main()
