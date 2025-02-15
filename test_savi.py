from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch


def convert_labels(dataframe: pd.DataFrame, label_name: str) -> pd.DataFrame:
    """
    Converts the categorical string label column of the given pandas DataFrame into a numerical category.
    """
    if dataframe[label_name].dtype == "object":
        dataframe[label_name] = dataframe[label_name].astype("category").cat.codes
    return dataframe

TRAIN_DATASET_PATH = "./synthetic_insider_threat.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

df = pd.read_csv(TRAIN_DATASET_PATH)
pd.set_option('display.max_columns', None)

df = convert_labels(df, "Category")

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

model_checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Convert DataFrames to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

def tokenize_function(examples):
    return tokenizer(examples["Tweet"], padding="longest", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True, num_proc=4)
test_dataset = test_dataset.map(tokenize_function, batched=True, num_proc=4)

train_dataset = train_dataset.rename_column("Category", "labels")
test_dataset = test_dataset.rename_column("Category", "labels")

num_labels = df["Category"].nunique()

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=num_labels)
model.to(device)  # Move model to GPU

training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=500,
    eval_steps=500,
    fp16=True,
    gradient_accumulation_steps=2,
    dataloader_num_workers=4  # Enable parallel data loading
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
)

trainer.train()

metrics = trainer.evaluate()
print(metrics)


