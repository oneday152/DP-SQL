import pickle
import chromadb
import os

def manage_vector_store(client, collection_name):
    try:
        existing_collections = client.list_collections()
        if collection_name in existing_collections:
            client.delete_collection(name=collection_name)
            print(f"Collection '{collection_name}' deleted.")
    except Exception as e:
        print(f"Error deleting collection '{collection_name}': {e}")

    try:
        vector_store = client.get_or_create_collection(name=collection_name)
        print(f"Collection '{collection_name}' created.")
        return vector_store
    except Exception as e:
        print(f"Error creating collection '{collection_name}': {e}")
        return None
    
def save_vector_data_to_disk(embeddings, metadata, ids, docs, filename, db_name, table_name):
    base_path = "Data\\dev_databases\\"
    db_path = base_path + db_name
    table_path = db_path + "\\" + table_name
    os.makedirs(table_path, exist_ok=True)
    filename = table_path + "\\" + filename
    try:
        with open(filename, 'wb') as f:
            pickle.dump((embeddings, metadata, ids, docs), f)
        print(f"Vector data saved to {filename}.")
    except Exception as e:
        print(f"Error saving vector data to disk: {e}")

def load_vector_data_from_disk(filename, db_name, table_name):
    base_path = "Data\\dev_databases\\"
    db_path = base_path + db_name
    table_path = db_path + "\\" + table_name

    os.makedirs(table_path, exist_ok=True)
    filename = table_path + "\\" + filename
    try:
        with open(filename, 'rb') as f:
            embeddings, metadata, ids, docs = pickle.load(f)
        print(f"Vector data loaded from {filename}.")
        return embeddings, metadata, ids, docs
    except Exception as e:
        print(f"Error loading vector data from disk: {e}")
        return None, None, None, None