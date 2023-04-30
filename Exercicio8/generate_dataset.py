from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from datasets import load_dataset
from pyserini.search.lucene import LuceneSearcher
from tqdm.auto import tqdm
import random
import json
import os
import pickle
import time

random.seed(12)
base_path = "."


class GPTCall:
    def __init__(self, model_name="gpt-3.5-turbo"):
        self.__model_name = model_name
        self.__template = """{instruction}

    Give me the good question for the document above
    """
        self.__prompt = PromptTemplate(template=self.__template, input_variables=["instruction"])
        self.__llm = ChatOpenAI(model_name=self.__model_name)
        self.__llm_chain = LLMChain(prompt=self.__prompt, llm=self.__llm)

    def get_answer(self, instruction):
        return self.__llm_chain.run(instruction).lstrip()


def generate_ramdom_indexes(corpus):
    pickle_file = f"{base_path}/data/random_list.pickle"
    if not os.path.isfile(pickle_file):
        random_list = []
        max = 1000
        with tqdm(total=max) as pbar:
            while len(random_list) < max:
                n = random.randint(0, len(corpus) - 1)

                # Prevent duplicated index
                if n not in random_list and corpus[n - 1:n]["text"][0] != "":
                    random_list.append(n)
                    pbar.update(1)

        with open(pickle_file, "wb") as f:
            pickle.dump(random_list, f)
    else:
        with open(pickle_file, "rb") as f:
            random_list = pickle.load(f)

    return random_list


def generate_ramdom_numbers(max=5, k=1000):
    random_list = []
    while len(random_list) < max:
        n = random.randint(0, k - 1)

        # Prevent duplicated index
        if n not in random_list:
            random_list.append(n)

    return random_list


def generate_dataset():
    pickle_dataset = f"{base_path}/data/pickle_dataset.pickle"
    pickle_indexes_processed = f"{base_path}/data/indexes_processed.pickle"

    gpt_call = GPTCall()
    passages_dataset = load_dataset("BeIR/trec-covid", "corpus")
    corpus = passages_dataset["corpus"]
    random_list = generate_ramdom_indexes(corpus)
    dataset_list = get_pickle_list(pickle_dataset)
    indexes_processed = get_pickle_list(pickle_indexes_processed)

    for index in tqdm(random_list):
        if index not in indexes_processed:
            instruction = corpus[index - 1:index]["text"][0]
            doc_id = corpus[index - 1:index]["_id"][0]
            positive_query = gpt_call.get_answer(instruction)
            random_doc_ids = search_with_bm25(positive_query)

            dataset_dict = {
                "positive_doc_id": doc_id,
                "query": positive_query,
                "negative_doc_ids": random_doc_ids
            }
            dataset_list.append(dataset_dict)
            indexes_processed.append(index)

            with open(pickle_dataset, "wb") as f:
                pickle.dump(dataset_list, f)

            with open(pickle_indexes_processed, "wb") as f:
                pickle.dump(indexes_processed, f)


def get_pickle_list(pickle_file):
    if not os.path.isfile(pickle_file):
        list_data = []
    else:
        with open(pickle_file, "rb") as f:
            list_data = pickle.load(f)

    return list_data


def search_with_bm25(query, max=5, k=1000):
    searcher = LuceneSearcher.from_prebuilt_index('beir-v1.0.0-trec-covid-flat')
    hits = searcher.search(query, k)
    random_list = generate_ramdom_numbers(max=max, k=k)
    random_ids = []

    for index in random_list:
        jsondoc = json.loads(hits[index].raw)
        random_ids.append(jsondoc["_id"])

    return random_ids


def export_jsonl():
    pickle_dataset = f"{base_path}/data/pickle_dataset.pickle"

    with open(pickle_dataset, "rb") as f:
        list_data = pickle.load(f)

    with open(f"{base_path}/data/manoel_1k_generated_queries_20230430.jsonl", "w") as file:
        for item in list_data:
            file.write(f"{json.dumps(item)}\n")


if __name__ == '__main__':
    generate_dataset()
    export_jsonl()
