import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from torch.utils.data import Dataset
import pandas as pd

model_name = "facebook/opt-125m"
output_dir = "./model"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print(model.num_parameters())


class PortugueseDataset(Dataset):
    def __init__(self, tokenizer, file_path: str):
        self.tokenizer = tokenizer
        self.file_path = file_path
        self.dataset = pd.read_csv(self.file_path, header=None, sep="\\n")
        self.length = len(self.dataset)

    def __len__(self):
        return self.length

    def preprocess(self, text):
        max_length = 512
        outputs = self.tokenizer(
            str(text).strip(),
            add_special_tokens=True,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            return_length=True,
        )

        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == max_length:
                input_batch.append(input_ids)

        return torch.tensor(input_batch)

    def __getitem__(self, i):
        """
        try:
            phrase = self.dataset.get_chunk(i).to_numpy()[0][0]
        except StopIteration:
            self.dataset = pd.read_csv(self.file_path, header=None, sep="\\n", iterator=True)
            phrase = self.dataset.get_chunk(i).to_numpy()[0][0]
        """
        try:
            if i <= self.length:
                phrase = self.dataset.values[i][0]
            else:
                phrase = ""
        except IndexError:
            print('IndexError:', i)
            phrase = ""

        return self.preprocess(phrase)


dataset = PortugueseDataset(
    tokenizer=tokenizer,
    file_path="data/sample-1gb.txt"
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False, mlm_probability=0.15
)

training_args = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=True,
    num_train_epochs=4,
    per_device_train_batch_size=22,
    do_train=True,
    save_steps=10_000,
    save_total_limit=2,
    logging_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=dataset
)

trainer.train()

eval_results = trainer.evaluate()
perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
print(f"Perplexity: {perplexity.item()}")

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
