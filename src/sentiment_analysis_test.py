import pandas as pd
import numpy as np

from typing import List, Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score
import os



class SVMSentimentModel:
    """
    OOP Sentiment Model using TF-IDF + Linear SVM.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.df = None
        self.vectorizer = None
        self.model = None
        
        # dataset splits
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None


    # Load Dataset from Excel
    def load_dataset_from_excel(self):
        """
        Load Excel file with columns: 'rating' and 'text'
        """

        try:
            print("Loading Excel file...")

            self.df = pd.read_excel(self.filepath)

            # Make sure required columns exist
            required_cols = {"rating", "text"}
            if not required_cols.issubset(self.df.columns):
                raise ValueError(f"Excel file must contain {required_cols}")

        except Exception as e:
            raise RuntimeError(f"Failed to load Excel file: {e}")

        print("Dataset loaded successfully.")
        print(self.df.head())
        return self.df


    # Cleaning
    def cleaning(self):
        if self.df is None:
            raise ValueError("Dataset must be loaded before cleaning.")

        try:
            # remove rows without text
            before = len(self.df)
            self.df = self.df.dropna(subset=['text'])
            self.df = self.df[self.df['text'].apply(lambda x: isinstance(x, str))]
            after = len(self.df)

            print(f"Removed {before - after} bad rows.")

            # create binary labels: negative (0), positive (1)
            self.df['label'] = self.df['rating'].apply(
                lambda x: 0 if x <= 2 else (1 if x >= 4 else None)
            )

            # remove rating==3 neutral rows
            before = len(self.df)
            self.df = self.df.dropna(subset=['label'])
            after = len(self.df)
            print(f"Removed {before - after} neutral rows (rating = 3).")

            self.df['label'] = self.df['label'].astype(int)

        except Exception as e:
            raise RuntimeError(f"Cleaning failed: {e}")

        print("Cleaning complete.")
        print(self.df.head())


    # Split Data
    def split_data(self, test_size=0.2):
        if self.df is None:
            raise ValueError("Dataset must be cleaned before splitting.")

        try:
            X = self.df['text']
            y = self.df['label']

            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

        except Exception as e:
            raise RuntimeError(f"Train/test split failed: {e}")

        print("Dataset split completed.")


    # Vectorize
    def vectorization(self):
        if self.X_train is None:
            raise ValueError("You must split data before vectorizing.")

        try:
            self.vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=50000
            )

            print("Fitting TF-IDF vectorizer...")
            self.X_train = self.vectorizer.fit_transform(self.X_train)
            self.X_test = self.vectorizer.transform(self.X_test)

        except Exception as e:
            raise RuntimeError(f"Vectorization failed: {e}")

        print("Vectorization complete.")


    # Train SVM Model
    def train(self):
        if self.vectorizer is None:
            raise ValueError("Vectorizer must be fitted before training.")

        try:
            print("Training Linear SVM...")
            self.model = LinearSVC()
            self.model.fit(self.X_train, self.y_train)

        except Exception as e:
            raise RuntimeError(f"Model training failed: {e}")

        print("Training complete.")


    # Evaluate
    def evaluate(self):
        if self.model is None:
            raise ValueError("Model must be trained before evaluation.")

        try:
            predictions = self.model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, predictions)
            f1 = f1_score(self.y_test, predictions)

        except Exception as e:
            raise RuntimeError(f"Evaluation failed: {e}")

        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 Score: {f1:.4f}")

        return {"accuracy": accuracy, "f1": f1}


    # Predict
    def predict(self, text: str):
        if self.model is None:
            raise ValueError("Model must be trained before prediction.")

        try:
            vectorized = self.vectorizer.transform([text])
            pred = self.model.predict(vectorized)[0]
            return "Positive" if pred == 1 else "Negative"

        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")


# RUN MODEL USING EXCEL FILE
model = SVMSentimentModel(
    filepath = os.path.join( "amazon_test_2500.xlsx")

)

model.load_dataset_from_excel()
model.cleaning()
model.split_data()
model.vectorization()
model.train()
model.evaluate()

print(model.predict("This product was amazing!"))
print(model.predict("Terrible, very disappointed."))
