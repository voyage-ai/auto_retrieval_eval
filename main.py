import argparse
import re
from utils import *
from paths import *
from generation.openai_gen_fn import get_gpt4_results

parser = argparse.ArgumentParser(description="build evaluation datasets")
parser.add_argument("--task_type", type=str, default="", choices=['', 'generate_score', 'parse_score', 'build_dataset'])
args = parser.parse_args()


def generate_label(generative_model, pairs, merged_pair_labels_path):
    if os.path.exists(merged_pair_labels_path):
        print('exist %s' % merged_pair_labels_path)
        pairs_previous = read_json_lines(merged_pair_labels_path)
        pairs_previous_ids = ['%s_%s'%(pair['query_id'], pair['doc_id']) for pair in pairs_previous]
        print('there are %d pairs in the previous labels %s' % (len(pairs_previous), merged_pair_labels_path))

        pairs_new = []
        for pair in pairs:
            pair_id = '%s_%s'%(pair['query_id'], pair['doc_id'])
            if pair_id not in pairs_previous_ids:
                pairs_new.append(pair)

        print('there are %d new pairs to be labelled' % (len(pairs_new)))
        pairs = pairs_new
        pairs_new = pairs_previous
    else:
        pairs_new = []
    
    if len(pairs)>0:
        criterion_prompt = f'You are an excellent AI researcher and you can do the data labelling well. Please decide whether the document and query are pairs based on the following criterion.\n'
        criterion_prompt += f'Labeling criterion: There are four levels: reject, borderline reject, borderline accept, and accept.\n'
        criterion_prompt += f"An 'Accept' classification, with a score of 4, is awarded when the query can be directly and precisely answered by utilizing only the information provided in the document, without necessitating any additional prior knowledge of the broader context or related topics.\n"
        criterion_prompt += f"A 'Borderline Accept' with a score of 3 is assigned when the document contains information that partially answers at least approximately 50% of the query, without necessitating any additional prior knowledge of the broader context or related topics.\n"
        criterion_prompt += f"A 'Borderline Reject' with a score of 2 is assigned when the document contains information that partially answers at most approximately 50% of the query, without necessitating any additional prior knowledge of the broader context or related topics.\n"
        criterion_prompt += f"A 'Reject' rating, accompanied by a score of 1, is given when the document offers scant information relevant to answering the query, without necessitating any additional prior knowledge of the broader context or related topics.\n"
        
        for index, pair in enumerate(pairs):
            print('-----------------------------------------------------------------------------------')
            print(merged_pair_labels_path)
            print('pair %d/%d' % (index, len(pairs)))
            query = pair['query']
            doc = pair['doc']
            
            input_text = f'''Given a document and a query:\nQuery:\n{query}\nDocument:\n{doc}\n'''
            input_text += criterion_prompt
            input_text += 'Your response must be one of the 4 labels {reject, borderline reject, borderline accept, accept}, representing the likelihood that the query is relevant to the document provided.\n'
            input_text += 'Please use 4 to represent accept, 3 to represent borderline accept, 2 to represent borderline reject, and 1 to represent reject.\n'
            input_text += 'Please put a single numeric value after your explanation and inside the \\boxed{generated answer} at the end of your response.'

            response_text = get_gpt4_results(input_text, generative_model)
            pair['gpt4_response'] = response_text
            
            pairs_new.append(pair)
            save_json_lines(merged_pair_labels_path, pairs_new)

def extract_numbers(strings):
    numbers = []
    for s in strings:
        match = re.findall(r'\d+', s)
        if match:
            numbers.extend(match)
    assert len(numbers) == 1
    return numbers[0]

def parse_generated_label(response_text):
    pattern = r'\\boxed\{([^}]*)\}'
    if response_text is None:
        generated_label = None
    
    match = re.search(pattern, response_text)
    
    if match:
        generated_label = match.group(1)
    else:
        generated_label = response_text

    generated_label = extract_numbers(generated_label)
    return generated_label


def parse_score(merged_pair_labels_path):
    print('-----------------------------------------------------')
    print('loading %s' % merged_pair_labels_path)
    pairs = read_json_lines(merged_pair_labels_path)
    print('there are %d labelled pairs' % (len(pairs)))
    
    for pair in pairs:
        try:
            score = int(parse_generated_label(pair['gpt4_response']))
            pair['gpt4_score'] = score            
        except:
            print(pair['query_id'], pair['doc_id'])
    
    print('parsing correct, done!')
    save_json_lines(merged_pair_labels_path, pairs)

def main():
    if args.task_type == 'generate_score':
        pairs = read_json_lines(merged_retrieval_data_path)
        print('there are %d pairs in top %d' % (len(pairs), topk))
        generate_label(generative_model, pairs, merged_pair_labels_path)
    
    elif args.task_type == 'parse_score':
        parse_score(merged_pair_labels_path)
    
    elif args.task_type == 'build_dataset':
        valid_labels = [4]
        pairs = read_json_lines(merged_pair_labels_path)

        valid_pairs = []
        for pair in pairs:
            if pair['gpt4_score'] in valid_labels:
                valid_pairs.append({'query': pair['query'], 'doc': pair['doc'], 'corpus_id': pair['doc_id'], 'query_id': pair['query_id']})

        outname = f'{filtered_pair_path}/pairs.jsonl'
        print('save %d pairs to %s' % (len(valid_pairs), outname))
        save_json_lines(outname, valid_pairs)
    
if __name__ == "__main__":
    main()
