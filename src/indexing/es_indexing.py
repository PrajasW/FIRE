from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from src.preprocessing.preprocess_data import get_preprocessed_data

def create_es_index(es_client, index_name):
    """
    Creates an Elasticsearch index with a specific mapping.
    """
    if es_client.indices.exists(index=index_name):
        print(f"Index {index_name} already exists. Deleting.")
        es_client.indices.delete(index=index_name)

    mapping = {
        "properties": {
            "title": {"type": "text"},
            "body": {"type": "text"},
            "date": {"type": "date", "format": "iso8601||strict_date_optional_time||epoch_millis"}
        }
    }
    es_client.indices.create(index=index_name, mappings=mapping)
    print(f"Index {index_name} created.")

def index_data(es_client, index_name, data):
    """
    Indexes preprocessed data into Elasticsearch using the bulk API.
    """
    actions = [
        {
            "_index": index_name,
            "_source": {
                "title": doc["title"],
                "body": doc["body"],
                "date": doc["date"]
            }
        }
        for doc in data if doc['date'] is not None # Filter out docs with no date for now
    ]
    
    success, failed = bulk(es_client, actions)
    print(f"Successfully indexed {success} documents.")
    if failed:
        print(f"Failed to index {len(failed)} documents.")

def main():
    # Initialize Elasticsearch client
    # Assumes Elasticsearch is running on localhost:9200
    es_client = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])
    
    if not es_client.ping():
        raise ConnectionError("Could not connect to Elasticsearch")

    index_name = "articles_index"
    
    # Get preprocessed data
    preprocessed_data = get_preprocessed_data()
    
    # Create index and index data
    create_es_index(es_client, index_name)
    index_data(es_client, index_name, preprocessed_data)

if __name__ == "__main__":
    main()
