import torch
import numpy as np
import random
import pandas as pd
from category_encoders import OrdinalEncoder
from sklearn.model_selection import train_test_split
import re


import evaluate
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel


train = pd.read_csv('./train.csv')

# Include spaces between amino acids
train.SEQUENCE = train.SEQUENCE.apply(lambda row: " ".join(row))
train.SEQUENCE = [re.sub(r"[UZB]", "X", sequence) for sequence in train.SEQUENCE]

# Encode labels
le = OrdinalEncoder(cols = ["LABEL"], return_df = False, mapping = 
                    [{"col": "LABEL", "mapping": {
                        "class0": 0,
                        "class1": 1,
                        "class2": 2,
                        "class3": 3,
                        "class4": 4,
                        "class5": 5,
                        "class6": 6,
                        "class7": 7,
                        "class8": 8,
                        "class9": 9,
                        "class10": 10,
                        "class11": 11,
                        "class12": 12,
                        "class13": 13,
                        "class14": 14,
                        "class15": 15,
                        "class16": 16,
                        "class17": 17,
                        "class18": 18,
                        "class19": 19,
                    }
                     }]
                   )
train.LABEL = le.fit_transform(train.LABEL)[:,0]


train, valid = train_test_split(train, test_size = 0.2, shuffle = True, stratify = train.LABEL, random_state = 42)


tds = Dataset.from_pandas(train[["SEQUENCE", "LABEL"]])
vds = Dataset.from_pandas(valid[["SEQUENCE", "LABEL"]])


ds = DatasetDict()

ds['train'] = tds
ds['valid'] = vds


print(ds)


model_name = "Rostlab/prot_bert_bfd"
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_dataset = ds["train"].shuffle(seed=42)
eval_dataset = ds["valid"].shuffle(seed=42)


def tokenize_function(examples):
    tokenized = tokenizer(
        examples["SEQUENCE"],
        padding="max_length",
        max_length=384,
        truncation=True,
        return_tensors="pt"
    )

    # Convert tensors to lists or numpy arrays
    tokenized_dict = {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in tokenized.items()}
    return tokenized_dict

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_eval_dataset = eval_dataset.map(tokenize_function, batched=True)

tokenized_train_dataset = tokenized_train_dataset.remove_columns(["SEQUENCE"])
tokenized_train_dataset = tokenized_train_dataset.rename_column("LABEL", "labels")

tokenized_eval_dataset = tokenized_eval_dataset.remove_columns(["SEQUENCE","__index_level_0__"])
tokenized_eval_dataset = tokenized_eval_dataset.rename_column("LABEL", "labels")

metric = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=20)
print(model)


# Define LoRA Config
lora_config = LoraConfig(
 r=16,
 lora_alpha=32,
 target_modules=["query", "value"],
 lora_dropout=0.05,
 bias="none",
 task_type=TaskType.SEQ_CLS, # this is necessary
 inference_mode=True
)

# add LoRA adaptor
model = get_peft_model(model, lora_config)
model.print_trainable_parameters() # see % trainable parameters




 # Train the model
training_args = TrainingArguments(output_dir="bert-peft-t", num_train_epochs=2, logging_strategy ="epoch", save_strategy ="epoch", save_total_limit=2,
                                 load_best_model_at_end=True, evaluation_strategy="epoch", per_device_train_batch_size=32, per_device_eval_batch_size=32)
bert_peft_trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset, # training dataset requires column input_ids
    eval_dataset=tokenized_eval_dataset,
    compute_metrics=compute_metrics,
)
bert_peft_trainer.train()



