import datasets
from huggingface_hub import login
from datasets import load_dataset
from tqdm.auto import tqdm
import random
import pandas as pd
import pickle
import os

base_path = "."


# login()

def get_passages():
    pickle_file = f"{base_path}/data/passages.pickle"
    passages_dataset = load_dataset("BeIR/trec-covid", "corpus")

    if not os.path.isfile(pickle_file):
        passages = {}
        for item in tqdm(passages_dataset["corpus"]):
            passages[item["_id"]] = {
                "fulltext": item["title"] + " " + item["text"]
            }

        with open(pickle_file, "wb") as f:
            pickle.dump(passages, f)
    else:
        with open(pickle_file, "rb") as f:
            passages = pickle.load(f)

    return passages


def get_samples():
    trec_ds = datasets.load_dataset('unicamp-dl/trec-covid-experiment')
    samples = []
    for ds in trec_ds:
        if "example" not in ds:
            for item in tqdm(trec_ds[ds]):
                positive_doc_id = item["positive_doc_id"]
                if len(item["negative_doc_ids"]) > 0:
                    negative_doc_id = item["negative_doc_ids"][random.randint(0, len(item["negative_doc_ids"]) - 1)]
                    sample = {
                        "id": positive_doc_id,
                        "query": item["query"],
                        "relevant": passages[positive_doc_id]["fulltext"],
                        "non_relevant": passages[negative_doc_id]["fulltext"],
                    }
                    samples.append(sample)

    df_data = pd.DataFrame(samples)

    df_data_pos = pd.DataFrame()
    df_data_neg = pd.DataFrame()

    df_data_pos["query"] = df_data["query"].values
    df_data_pos["passage"] = df_data["relevant"].values
    df_data_pos["score"] = 1

    df_data_neg["query"] = df_data["query"].values
    df_data_neg["passage"] = df_data["non_relevant"].values
    df_data_neg["score"] = 0

    df_data_merge = pd.concat([df_data_pos, df_data_neg], axis=0, ignore_index=True)

    return df_data_merge


if __name__ == '__main__':
    passages = get_passages()

    df_data = get_samples()
