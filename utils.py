import pickle
import json

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