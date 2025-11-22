import pandas as pd
from datasets import Dataset
from typing import Dict, List, Any
import evaluate
import numpy as np
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)


class SentimentModel:
    """
    Full OOP pipeline for training/evaluating a DistilBERT sentiment model
    using a local Excel dataset (amazon_test_2500.xlsx).
    """

    default_checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"

    def __init__(self, checkpoint: str = None, filepath: str = None):
        self.checkpoint = checkpoint or SentimentModel.default_checkpoint
        self.filepath = filepath or "amazon_test_2500.xlsx"

        # Placeholders
        self.tokenizer = None
        self.raw_datasets = None
        self.cleaned_dataset = None
        self.tokenized_datasets = None
        self.data_collator = None
        self.dataset_splits = None
        self.model = None
        self.training_args = None
        self.trainer = None

 
    # 1. Load Dataset
    def load_dataset(self):
        print(f"Loading Excel file: {self.filepath}")

        df = pd.read_excel(self.filepath)

        if "reviewText" not in df.columns:
            raise KeyError("Excel must contain a 'reviewText' column.")

        # If sentiment not provided, convert rating to sentiment labels
        if "sentiment" not in df.columns:
            if "rating" in df.columns:
                print("Converting rating → sentiment labels (0 or 1)...")
                df["sentiment"] = df["rating"].apply(
                    lambda x: 0 if x <= 2 else (1 if x >= 4 else 0)
                )
            else:
                raise KeyError("Dataset must contain either 'sentiment' or 'rating'.")

        # Convert pandas → HuggingFace Dataset
        self.raw_datasets = Dataset.from_pandas(df)
        print("Dataset loaded successfully.")

    # 2. Tokenizer
    def load_tokenizer(self):
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        print("Tokenizer loaded.")

    # 3. Cleaning
    def cleaning(self):
        print("Cleaning dataset...")

        texts = self.raw_datasets["reviewText"]

        good_indices = [
            i for i, t in enumerate(texts)
            if t is not None and isinstance(t, str)
        ]

        self.cleaned_dataset = self.raw_datasets.select(good_indices)
        print(f"Cleaned dataset size: {len(self.cleaned_dataset)}")

    # 4. Tokenization
    def tokenize_function(self, batch: Dict[str, List[Any]]):
        texts = [str(t) if t else "" for t in batch["reviewText"]]
        return self.tokenizer(texts, padding="max_length", truncation=True)

    def tokenization(self):
        print("Tokenizing...")
        self.tokenized_datasets = self.cleaned_dataset.map(
            self.tokenize_function,
            batched=True
        )

    # 5. Data Collator
    def load_data_collator(self):
        print("Loading data collator...")
        self.data_collator = DataCollatorWithPadding(self.tokenizer)

    # 6. Model Loading
    def load_model(self):
        print("Loading model...")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.checkpoint,
            num_labels=2
        )
        print("Model ready.")

    # 7. Split
    def split_dataset(self, test_ratio=0.2):
        print("Splitting train/test datasets...")
        self.dataset_splits = self.tokenized_datasets.train_test_split(
            test_size=test_ratio
        )
        print("Split complete.")

    # 8. Metrics
    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)

        accuracy = evaluate.load("accuracy")
        return accuracy.compute(predictions=preds, references=labels)

    # 9. Trainer
    def create_trainer(self):
        print("Creating Trainer...")

        self.training_args = TrainingArguments(
            output_dir="model_output",
            num_train_epochs=2,
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            evaluation_strategy="epoch",
            fp16=torch.cuda.is_available(),
        )

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.dataset_splits["train"],
            eval_dataset=self.dataset_splits["test"],
            tokenizer=self.tokenizer,
            data_collator=self.data_collator,
            compute_metrics=self.compute_metrics,
        )

    # 10. Train
    def train(self):
        print("Training...")
        self.trainer.train()

    # 11. Evaluate
    def evaluate_model(self):
        print("Evaluating model...")
        return self.trainer.evaluate()

    # 12. Predict
    def predict(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=-1).item()
        return "Positive" if prediction == 1 else "Negative"



def main():
    model = SentimentModel(filepath="amazon_test_2500.xlsx")

    model.load_dataset()
    model.load_tokenizer()
    model.cleaning()
    model.tokenization()
    model.load_data_collator()
    model.load_model()
    model.split_dataset()
    model.create_trainer()

    model.train()
    results = model.evaluate_model()
    print("Final Evaluation:", results)

    # Sample predictions
    print("\nTest Predictions:")
    print(model.predict("I love this product, amazing quality!"))
    print(model.predict("Terrible experience, I hate it."))


if __name__ == "__main__":
    main()
