class Config:
    """
    Configuration class for managing paths and settings in the project.
    Uses argparse to allow external configuration through command line.
    """
    def __init__(self, args):
        self.task_name = args.task_name
        self.embedding_models = args.embedding_models.split(',')
        self.topk = args.topk
        self.generative_model = args.generative_model
        self.data_path = f'data/{self.task_name}'
        self.meta_data_path = f'{self.data_path}/meta_data'
        self.embedding_path = f'{self.meta_data_path}/embeddings'
        self.retrieval_path = f'{self.meta_data_path}/retrieval'
        self.merged_retrieval_path = f'{self.meta_data_path}/retrieval_merge'
        self.merged_retrieval_data_path = (
            f'{self.merged_retrieval_path}/pairs_'
            f'{len(self.embedding_models)}embedding_models_top{self.topk}.jsonl'
        )
        self.merged_pair_labels_path = (
            f'{self.merged_retrieval_path}/pairs_'
            f'{len(self.embedding_models)}embedding_models_top{self.topk}_gpt4.jsonl'
        )