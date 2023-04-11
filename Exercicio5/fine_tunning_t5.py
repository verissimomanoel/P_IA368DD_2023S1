import pandas as pd
from tqdm.auto import tqdm
import json
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig
)
import torch
import pickle

base_path = "."
topics = "trec-covid"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("model")
model = AutoModelForSeq2SeqLM.from_pretrained("model").to(device)

NUMBER_OF_SEQUENCES = 20
generation_params = GenerationConfig(
    max_new_tokens=200,
    do_sample=True,
    temperature=1.2,
    top_k=50,
    top_p=0.8,
    num_beams=1,
    num_return_sequences=NUMBER_OF_SEQUENCES
)
SAVE_AFTER = 1000
START_POSITION=0
END_POSITION=129192
BATCH_SIZE=2
TREC_COVID_EXP=f"{base_path}/data/generated_expansion.pkl"
base_path = "."

# Mudar o formato de jsonl para tsv
# with open(f'{base_path}/data/corpus.tsv','w') as output:
#     with open(f'{base_path}/data/corpus.jsonl', 'r') as file:
#         for line in tqdm(file):
#             data = json.loads(line)
#             id = data['_id']
#             text = data['text']
#             output.write(f'{id}\t{text}\n')
            
trec_covid_corpus_df = pd.read_csv(f"{base_path}/data/corpus.tsv", sep='\t', names=["id", "text"])
trec_covid_corpus_df = trec_covid_corpus_df.dropna()

document_expansion = []

doc_texts = trec_covid_corpus_df.iloc[START_POSITION:END_POSITION]['text'].tolist()
doc_indexes = trec_covid_corpus_df.iloc[START_POSITION:END_POSITION].index.tolist()

for i in (pbar := tqdm(range(0, END_POSITION - START_POSITION, BATCH_SIZE))):
    pbar.set_description("Generating topics for sequence={}~{}/{}".format(doc_indexes[i], doc_indexes[i + BATCH_SIZE - 1], trec_covid_corpus_df.shape[0]))

    input_ids = tokenizer(doc_texts[i:(i + BATCH_SIZE)], padding=True, return_tensors='pt', truncation=True).input_ids.to(device)

    generated_text = model.generate(inputs=input_ids, generation_config=generation_params)

    decoded_text = tokenizer.batch_decode(generated_text, skip_special_tokens=True)

    document_expansion += [" ".join(decoded_text[i:i + NUMBER_OF_SEQUENCES]) for i in list(range(0, len(decoded_text), NUMBER_OF_SEQUENCES))]

with open(TREC_COVID_EXP, "wb") as outputFile:
    pickle.dump({"doc_indexes": doc_indexes,
                 "doc_expansion": document_expansion}, outputFile, pickle.HIGHEST_PROTOCOL)
