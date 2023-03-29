import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import pandas as pd

model_name = "facebook/opt-125m"
output_dir = "./model"
max_length = 512

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def tokenize_with_context(data):
    outputs = tokenizer(
        data["text"],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == max_length:
            input_batch.append(input_ids)

    return {"input_ids": input_batch}


def main():
    dataset = load_dataset("text", data_files="data/sample-1gb.txt")
    val_size = 1000 / len(dataset['train'])
    dataset = dataset['train'].train_test_split(test_size=val_size, seed=42)
    dataset = DatasetDict(
        {
            "train": dataset["train"],
            "valid": dataset["test"]
        }
    )

    tokenized_datasets = dataset.map(
      tokenize_with_context, batched=True, remove_columns=dataset["train"].column_names
    )

    tokenized_datasets.save_to_disk("data/tokenized_dataset")

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
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["valid"]
    )

    trainer.train()

    eval_results = trainer.evaluate()
    perplexity = torch.exp(torch.tensor(eval_results["eval_loss"]))
    print(f"Perplexity: {perplexity.item()}")

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == '__main__':
    main()

