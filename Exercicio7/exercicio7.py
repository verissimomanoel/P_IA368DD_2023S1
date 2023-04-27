"""Exercicio7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1789O1BFuivzUG1pMPfFxoYR7Xmwv0HFa

# Aula 7 - Solução dos exercícios

Implementar a fase de indexação e buscas de um modelo esparso

- Usar este modelo SPLADE já treinado naver/splade_v2_distil (do distilbert) ou splade-cocondenser-selfdistil (do BERT-base 110M params). Mais informações sobre os modelos estão neste artigo: https://arxiv.org/pdf/2205.04733.pdf
- Não é necessário treinar o modelo
- Avaliar nDCG@10 no TREC-COVID e comparar resultados com o BM25 e buscador denso da semana passada
- A dificuldade do exercício está em implementar a função de busca e ranqueamento usada pelo SPLADE. A implementação do índice invertido é apenas um "dicionário python".
- Comparar seus resultados com a busca "original" do SPLADE.
Medir latencia (s/query)
"""

# Change de path to your drive
base_path = "gdrive/MyDrive/Colab_Notebooks/P_IA368DD_2023S1/Exercicio7"

from datasets import load_dataset
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorWithPadding
from collections import defaultdict
from typing import List
from evaluate import load
import torch
from tqdm.auto import tqdm
import pickle

"""### SPLADE no TREC-COVID"""

model_id = "naver/splade-cocondenser-ensembledistil"
max_length = 256
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trec_eval = load("trec_eval")


def preprocess(sample):
    full_text = sample['title'] + ' ' + sample['text']
    return {'complete_text': full_text, 'len': len(full_text)}


def load_datasets():
    passages_dataset = load_dataset("BeIR/trec-covid", "corpus")
    queries_dataset = load_dataset("BeIR/trec-covid", "queries")
    qrels_dataset = load_dataset("BeIR/trec-covid-qrels")

    passages_dataset = passages_dataset.map(lambda x: preprocess(x))

    return passages_dataset, queries_dataset, qrels_dataset


@torch.no_grad()
def get_embeddings(model, tokenizer, texts: List[str]):
    tokens = tokenizer(texts,
                       return_tensors='pt',
                       padding=True,
                       truncation=True)
    tokens = collator(tokens)
    output = model(**tokens.to(device))

    sparse_vecs = torch.max(
        torch.log(
            1 + torch.relu(output.logits)
        ) * tokens.attention_mask.unsqueeze(-1),
        dim=1
    )[0].squeeze()

    return sparse_vecs


def compress_sparse_embeddings(sparse_embeddings):
    output = []

    non_zeros = sparse_embeddings.nonzero()

    for i in range(0, sparse_embeddings.shape[0]):
        rows = non_zeros[:, 0] == i
        idxs = non_zeros[rows, 1]
        scores = sparse_embeddings[i, idxs]

        output.append(dict(zip(idxs.cpu().tolist(), scores.cpu().tolist())))

    return output


def generate_sparse_passage_embeddings():
    batch_size = 32
    inverted_index = {}

    # Kudos to Marcos Piau/Gustavo
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        for i in tqdm(range(0, len(passages_dataset['corpus']), batch_size)):
            i_end = i + batch_size
            i_end = len(passages_dataset['corpus']) if i_end > len(passages_dataset['corpus']) else i_end

            batch = passages_dataset['corpus'][i:i_end]
            ids = batch['_id']
            texts = batch['complete_text']

            sparse_embeddings = get_embeddings(model, tokenizer, texts)
            compressed_embeddings = compress_sparse_embeddings(sparse_embeddings)

            for compressed_embedding, id in zip(compressed_embeddings, ids):

                sparse_dict_tokens = {
                    idx2token[idx]: weight for idx, weight in compressed_embedding.items()
                }

                for token, weight in sparse_dict_tokens.items():
                    if token not in inverted_index:
                        inverted_index[token] = []
                    inverted_index[token].append((id, weight))

        with open(f'{base_path}/data/inverted_index.pickle', 'wb') as handle:
            pickle.dump(inverted_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return inverted_index


def generate_sparse_query_embeddings():
    queries_compressed = {}
    batch_size = 32

    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
        for i in tqdm(range(0, len(queries_dataset['queries']), batch_size)):
            i_end = i + batch_size
            i_end = len(queries_dataset['queries']) if i_end > len(queries_dataset['queries']) else i_end

            batch = queries_dataset['queries'][i:i_end]
            ids = batch['_id']
            texts = batch['text']

            query_sparse_embeddings = get_embeddings(model, tokenizer, texts)
            compressed_embeddings = compress_sparse_embeddings(query_sparse_embeddings)

            for compressed_embedding, id in zip(compressed_embeddings, ids):
                sparse_dict_tokens = {
                    idx2token[idx]: weight for idx, weight in compressed_embedding.items()
                }

                queries_compressed[id] = sparse_dict_tokens

    return queries_compressed


def generate_scores():
    scores = {}

    for query_id in tqdm(queries_compressed):
        query_compressed = queries_compressed[query_id]
        for token, q_score in query_compressed.items():
            if token in inverted_index:
                if query_id not in scores:
                    scores[query_id] = {}

                docs_ids = inverted_index[token]

                for doc_id, d_score in docs_ids:
                    if doc_id not in scores[query_id]:
                        scores[query_id][doc_id] = 0

                    scores[query_id][doc_id] += q_score * d_score

    for query_id in scores:
        scores[query_id] = dict(sorted(scores[query_id].items(), key=lambda h: h[1], reverse=True))

    return scores


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForMaskedLM.from_pretrained(model_id).to(device)

    collator = DataCollatorWithPadding(tokenizer)

    idx2token = {
        idx: token for token, idx in tokenizer.get_vocab().items()
    }

    passages_dataset, queries_dataset, qrels_dataset = load_datasets()

    sparse_embeddings = get_embeddings(model, tokenizer, queries_dataset['queries']['text'][:2])
    sparse_embeddings.shape

    # inverted_index = generate_sparse_passage_embeddings()
    #
    # queries_compressed = generate_sparse_query_embeddings()
    #
    # scores = generate_scores()
    #
    # qrels_format = defaultdict(list)
    #
    # for query in qrels_dataset['test']:
    #   qrels_format['query'].append(query["query-id"])
    #   qrels_format['q0'].append("q0")
    #   qrels_format['docid'].append(str(query["corpus-id"]))
    #   qrels_format['rel'].append(query["score"])
    #
    # run_format = defaultdict(list)
    #
    # for query_id in tqdm(scores, desc="Query"):
    #   rank = 1
    #   docs = scores[query_id]
    #
    #   for doc_id in docs:
    #     score = scores[query_id][doc_id]
    #     run_format['query'].append(query_id)
    #     run_format['q0'].append("q0")
    #     run_format['docid'].append(str(doc_id))
    #     run_format['rank'].append(rank)
    #     run_format['score'].append(score)
    #     run_format['system'].append("SPLADE")
    #
    #     if rank == 1000:
    #       break
    #
    #     rank += 1
    #
    # results = trec_eval.compute(predictions=[run_format], references=[qrels_format])
    #
    # results['NDCG@10']
