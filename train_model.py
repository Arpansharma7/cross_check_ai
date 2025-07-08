from datasets import load_dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load dataset
dataset = load_dataset("Hello-SimpleAI/HC3", name="all", split="train")
print("Loaded dataset:", len(dataset))

#labeling and text flattening
def assign_label(ex):
    ex["label"] = 1 if ex.get("chatgpt_answers") else 0
    answers = ex["chatgpt_answers"] if ex["chatgpt_answers"] else ex["human_answers"]
    ex["text"] = " ".join(answers)
    return ex

dataset = dataset.map(assign_label)

#Load the tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

#perform Tokenization
def tokenize_batch(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
    tokens["label"] = batch["label"] 
    return tokens

tokenized_dataset = dataset.shuffle(seed=42).select(range(5000)).map(tokenize_batch, batched=True)

#Change the format for PyTorch
tokenized_dataset = tokenized_dataset.remove_columns(["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

#Load model
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

#Training setup
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    logging_steps=20,
    save_steps=1000,
    weight_decay=0.01
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)

#Train
trainer.train()

#Save
model.save_pretrained("saved_detector")
tokenizer.save_pretrained("saved_detector")
print("model saved") 
