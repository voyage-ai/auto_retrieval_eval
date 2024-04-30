import pickle
import json
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Configure paths and settings for the project.")
    parser.add_argument('--task-name', type=str, default='example_task', help='Task name for the dataset.')
    parser.add_argument('--embedding-models', type=str, default='voyage-large-2,text-embedding-3-large', help='Comma-separated list of embedding models.')
    parser.add_argument('--topk', type=int, default=20, help='Top K retrieved results.')
    parser.add_argument('--generative-model', type=str, default='gpt-4-0125-preview', help='Generative model name.')
    
    args = parser.parse_args()
    return args

def create_directories(config):
    """Create necessary directories for the project."""
    try:
        os.makedirs(config.embedding_path, exist_ok=True)
        os.makedirs(config.retrieval_path, exist_ok=True)
        os.makedirs(config.merged_retrieval_path, exist_ok=True)
    except OSError as e:
        print(f"Error creating directories: {e}")

def read_json(data_path):
    with open(data_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def read_json_lines(data_path):
    data = []
    with open(data_path, 'r') as json_file:
        for idx, line in enumerate(json_file):
            try:
                data_line = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
                print('Line:', idx)
            data.append(data_line)
    return data

def save_json(result_path, data):
    with open(result_path, 'w') as jsonl_file:
        jsonl_file.write(json.dumps(data) + '\n')

def save_json_lines(result_path, data):
    with open(result_path, 'w') as f:
        for data_line in data:
            json.dump(data_line, f)
            f.write('\n')

def read_txt(data_path):
    with open(data_path, 'r') as file:
        data = file.read()
    return data

def read_txt_lines(data_path):
    data = []
    with open(data_path, 'r') as file:    
        for line in file:
            data.append(line.strip())
    return data

def save_pickle(result_path, data):
    with open(result_path, 'wb') as file:
        pickle.dump(data, file)

def read_pickle(result_path):
    with open(result_path, 'rb') as file:
        data = pickle.load(file)
    return data