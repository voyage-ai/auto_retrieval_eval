import re
import os
from utils import parse_arguments, read_json_lines, save_json_lines
from config import Config
from generation.openai_gen_fn import get_gpt4_results
from tqdm import tqdm
import json

def read_json_if_exists(path):
    if os.path.exists(path):
        return read_json_lines(path)
    return []

def generate_label(generative_model, pairs, merged_pair_labels_path):
    existing_pairs = read_json_if_exists(merged_pair_labels_path)
    new_pairs = identify_new_pairs(pairs, existing_pairs)
    print(f'{len(new_pairs)} new pairs to be labelled')

    if new_pairs:
        labeled_pairs = label_pairs(new_pairs, generative_model)
        all_pairs = labeled_pairs + existing_pairs
        save_json_lines(merged_pair_labels_path, all_pairs)

def identify_new_pairs(pairs, existing_pairs):
    existing_ids = {'{}_{}'.format(p['query_id'], p['doc_id']) for p in existing_pairs}
    return [p for p in pairs if '{}_{}'.format(p['query_id'], p['doc_id']) not in existing_ids]

def label_pairs(pairs, generative_model):
    prompt = build_criterion_prompt()
    labeled_pairs = []
    for pair in tqdm(pairs, desc="Labeling pairs"):
        labeled_pair = label_pair(pair, prompt, generative_model)
        labeled_pairs.append(labeled_pair)
    return labeled_pairs

def build_criterion_prompt():
    prompt = (
        "You are an excellent AI researcher and you can do the data labelling well. Please decide whether the document "
        "and query are pairs based on the following criterion.\n"
        "Labeling criterion: There are four levels: reject, borderline reject, borderline accept, and accept.\n"
        "An 'Accept' classification, with a score of 4, is awarded when the query can be directly and precisely answered "
        "by utilizing only the information provided in the document, without necessitating any additional prior knowledge "
        "of the broader context or related topics.\n"
        "A 'Borderline Accept' with a score of 3 is assigned when the document contains information that partially answers "
        "at least approximately 50% of the query, without necessitating any additional prior knowledge of the broader "
        "context or related topics.\n"
        "A 'Borderline Reject' with a score of 2 is assigned when the document contains information that partially answers "
        "at most approximately 50% of the query, without necessitating any additional prior knowledge of the broader "
        "context or related topics.\n"
        "A 'Reject' rating, accompanied by a score of 1, is given when the document offers scant information relevant to "
        "answering the query, without necessitating any additional prior knowledge of the broader context or related topics.\n"
        "Your response must be one of the 4 labels {reject, borderline reject, borderline accept, accept}, representing the "
        "likelihood that the query is relevant to the document provided.\n"
        "Please use 4 to represent accept, 3 to represent borderline accept, 2 to represent borderline reject, and 1 to represent "
        "reject.\n"
        "Please put a single numeric value after your explanation and inside the \\boxed{generated answer} at the end of your response.\n"
    )
    return prompt

def label_pair(pair, prompt, generative_model):
    query, doc = pair['query'], pair['doc']
    input_text = f'Given a document and a query:\nQuery:\n{query}\nDocument:\n{doc}\n{prompt}'
    response_text = get_gpt4_results(input_text, generative_model)
    pair['gpt4_response'] = response_text
    return pair

def parse_score(merged_pair_labels_path):
    doc_query_scores = {}
    pairs = read_json_if_exists(merged_pair_labels_path)
    for pair in pairs:
        try:
            gpt4_score = int(parse_digit(pair['gpt4_response']))
            if gpt4_score > 2:
                if not doc_query_scores.get(pair['query_id']):
                    doc_query_scores[pair['query_id']] = {}
                doc_query_scores[pair['query_id']][pair['doc_id']] = int(gpt4_score)
        except ValueError:
            print(f"Error parsing score for {pair['query_id']} {pair['doc_id']}")
    
    # Remove empty doc_query_scores entries
    doc_query_scores = {query_id: scores for query_id, scores in doc_query_scores.items() if scores}
    
    return doc_query_scores

def parse_digit(response_text):
    pattern = r'\\boxed\{([^}]*)\}'
    match = re.search(pattern, response_text)
    if match:
        numbers = re.findall(r'\d+', match.group(1))
        if numbers:
            return numbers[0]
    return None

def main():
    args = parse_arguments()
    config = Config(args)
    pairs = read_json_lines(config.merged_retrieval_data_path)
    print(f'there are {len(pairs)} pairs in top {config.topk}')
    generate_label(config.generative_model, pairs, config.merged_pair_labels_path)
    doc_query_scores = parse_score(config.merged_pair_labels_path)
    with open(os.path.join(config.data_path, "relevance.json"), "w") as f:
        json.dump(doc_query_scores, f)

if __name__ == "__main__":
    main()
