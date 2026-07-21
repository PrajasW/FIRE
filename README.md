# FIRE: Fast Information Retrieval Engine 

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Elasticsearch](https://img.shields.io/badge/Elasticsearch-Integration-005571?logo=elasticsearch)
![System](https://img.shields.io/badge/System-Distributed-brightgreen)

## Overview

I built this custom information retrieval (IR) engine to experiment with inverted index construction, storage backend trade-offs, and query processing optimization. The primary goal was to construct a full-text search index from scratch and benchmark its performance against production-ready systems like Elasticsearch. We use large-scale Wikipedia and news datasets to test latency, throughput, and memory constraints under realistic loads.

## Core Features

- **Custom Inverted Index:** Handles text preprocessing through built-in tokenization, stop-word filtering, and stemming.
- **Query Execution Models:** Implements both Term-At-A-Time (TAAT) and Document-At-A-Time (DAAT) execution strategies to evaluate query planning trade-offs.
- **Boolean Logic Evaluation:** Parses and executes complex queries using `AND`, `OR`, `NOT`, and `PHRASE` operators with strict nested precedence.
- **Relevance Scoring:** Computes document relevance using Term-Frequency (TF) and TF-IDF metrics.
- **Postings Optimization:** Compresses postings lists to save memory and leverages skip pointers to speed up query intersection.
- **Storage Backends:** Benchmarks custom disk serialization against databases like PostgreSQL GIN, Redis, and RocksDB.
- **Performance Profiling:** Tracks 95th and 99th percentile query latency, read/write QPS, and memory footprint, using a standard Elasticsearch deployment as a baseline.

## Architecture

```text
src/
├── core/             # Base classes for indexers, documents, and storage interfaces
├── indexing/         # Inverted index implementations and database connections
├── preprocessing/    # NLP pipelines: Tokenization, Stemming, Stop-word filtering
└── querying/         # Query parsers, TAAT/DAAT execution engines, Boolean logic evaluators
```

## Benchmarks and Evaluation

*(Placeholder for latency, throughput, and memory plots)*

- **Latency:** Comparing $p_{95}$ and $p_{99}$ response times of our custom index versus Elasticsearch.
- **Throughput:** Measuring read and write QPS under varying query complexity.
- **Memory Footprint:** Tracking storage constraints across different datastores and compression algorithms.

## Getting Started

You will need Python 3.8+ and a running Elasticsearch instance (local or via Docker) to execute the benchmarking suite.

### Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/your-username/Custom-Search-Engine.git
cd Custom-Search-Engine
pip install -r requirements.txt
```

### Usage

The following commands represent the standard workflow for ingesting data and running queries.

**1. Indexing Data**

To build a new index from a dataset, pass the file path to the indexing module:

```bash
python -m src.indexing.self_index_v1_0 --data ./data/wiki.json
```

**2. Running Queries**

Execute a boolean search against the built index:

```bash
python -m src.querying --query '("Apple" AND "Banana") OR ("Orange" AND NOT "Grape")'
```
