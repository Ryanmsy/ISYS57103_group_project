import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import numpy as np
import os
from svm_sentiment import SVMSentimentModel

class TestSVMSentimentModel(unittest.TestCase):

    def setUp(self):
        self.db_path = "test_db.sqlite"
        self.model = SVMSentimentModel(db_filepath=self.db_path)

    def test_initialization(self):
        self.assertEqual(self.model.db_filepath, self.db_path)
        self.assertIsNone(self.model.df)
        self.assertIsNone(self.model.vectorizer)

    @patch('svm_sentiment.sqlite3')
    @patch('svm_sentiment.pd.read_sql')
    @patch('svm_sentiment.os.path.exists')
    def test_load_dataset_from_db(self, mock_exists, mock_read_sql, mock_sqlite):
        """Test loading data without needing a real DB file."""
        # 1. Mock file existence
        mock_exists.return_value = True
        
        # 2. Mock the dataframe returned by SQL
        mock_df = pd.DataFrame({
            'text': ['Great product', 'Bad product'],
            'rating': [5, 1]
        })
        mock_read_sql.return_value = mock_df
        
        # 3. Run method
        df = self.model.load_dataset_from_db()
        
        # 4. Assertions
        self.assertEqual(len(df), 2)
        self.assertIn('text', df.columns)
        mock_sqlite.connect.assert_called_with(self.db_path)

    def test_cleaning(self):
        """Test if cleaning removes bad rows and creates labels."""
        # Create a raw dummy dataframe
        raw_data = {
            'text': ['Good', None, 'Bad', 'Okay'],
            'rating': [5, 5, 1, 3] # 5=pos, 1=neg, 3=neutral (should be removed)
        }
        self.model.df = pd.DataFrame(raw_data)
        
        self.model.cleaning()
        
        # Expecting 'Good' (pos) and 'Bad' (neg). 
        # None text should be gone. Rating 3 (neutral) should be gone.
        self.assertEqual(len(self.model.df), 2)
        self.assertTrue('label' in self.model.df.columns)
        # Check if labels are integers
        self.assertEqual(self.model.df.iloc[0]['label'], 1)

    def test_vectorization_and_train(self):
        """Test the flow of vectorization and training using mocks."""
        # Setup dummy data
        self.model.X_train = ["bad movie", "good movie"]
        self.model.X_test = ["average movie"]
        self.model.y_train = [0, 1]
        self.model.y_test = [0]
        
        # Mock the vectorizer and model to avoid actual computation
        self.model.vectorizer = MagicMock()
        self.model.vectorizer.fit_transform.return_value = "Mock Matrix"
        self.model.vectorizer.transform.return_value = "Mock Matrix"
        
        # Run vectorization (we override the internal creation for the test)
        # Note: In your real code, vectorization() creates the object. 
        # We can run the real vectorization since it's fast, 
        # or just mock the sklearn import if we wanted strictly unit tests.
        # Let's run the real one for this test since Tfidf is fast.
        self.model.vectorization()
        self.assertIsNotNone(self.model.vectorizer)
        
        # Test Training
        self.model.train()
        self.assertIsNotNone(self.model.model)

    def test_predict_logic(self):
        """Test prediction mapping."""
        self.model.vectorizer = MagicMock()
        self.model.model = MagicMock()
        
        # Mock model returning [1] (Positive)
        self.model.model.predict.return_value = [1]
        result = self.model.predict("Test text")
        self.assertEqual(result, "Positive")
        
        # Mock model returning [0] (Negative)
        self.model.model.predict.return_value = [0]
        result = self.model.predict("Test text")
        self.assertEqual(result, "Negative")

if __name__ == '__main__':
    unittest.main()