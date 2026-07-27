import json
import random
import os

def load_queries(query_type, num_queries=5, filepath='test_queries.json'):
    if not os.path.exists(filepath):
        # fallback for different working directories
        fallback = os.path.join(os.path.dirname(__file__), '..', '..', 'test_queries.json')
        if os.path.exists(fallback):
            filepath = fallback
        else:
            print(f"Warning: {filepath} not found. Using fallback hardcoded queries.")
            if query_type == 'ranked':
                return {"Ranked 1": "world news", "Ranked 2": "apple technology"}
            elif query_type == 'and_only':
                return {"AND 1": "war and peace", "AND 2": "apple and computer"}
            else:
                return {"Simple": "news", "AND": "war and peace", "OR": "apple or google"}

    with open(filepath, 'r') as f:
        queries_data = json.load(f)
        
    selected_queries = {}
    if query_type == 'boolean':
        for qt, qlist in queries_data['boolean'].items():
            q = random.choice(qlist)
            selected_queries[f"Random {qt}"] = q
    elif query_type == 'ranked':
        qlist = queries_data['ranked']
        sampled = random.sample(qlist, min(num_queries, len(qlist)))
        for i, q in enumerate(sampled):
            selected_queries[f"Ranked {i+1}"] = q
    elif query_type == 'and_only':
        qlist = queries_data['boolean']['AND']
        sampled = random.sample(qlist, min(num_queries, len(qlist)))
        for i, q in enumerate(sampled):
            selected_queries[f"AND {i+1}"] = q
            
    return selected_queries
