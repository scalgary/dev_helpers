import unittest
# Ajoute ce bloc pour que Python trouve le module parent
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from set_up_ml import transform_df_for_model


import pandas as pd
import numpy as np

class TestTransformDfForModel(unittest.TestCase):

    def setUp(self):
        # Exemple de DataFrame avec des variables numériques et catégorielles
        self.df = pd.DataFrame({
            'age': [25, 35, 45],
            'gender': ['male', 'female', 'female'],
            'income': [50000, 60000, 70000]
        })
        self.df['gender'] = self.df['gender'].astype('category')

    def test_numeric_only(self):
        terms = ['age', 'income']
        X = transform_df_for_model(self.df, terms)
        self.assertIn('age', X.columns)
        self.assertIn('income', X.columns)
        self.assertIn('intercept', X.columns)
        self.assertEqual(X.shape[1], 3)

    def test_categorical_drop(self):
        terms = ['gender']
        X = transform_df_for_model(self.df, terms, contrast='drop')
        self.assertTrue(any(col.startswith('gender[') for col in X.columns))
        self.assertEqual(X.shape[1], 2)  # 1 category + intercept (with drop)

    def test_categorical_sum(self):
        terms = ['gender']
        X = transform_df_for_model(self.df, terms, contrast='sum')
        self.assertTrue(any(col.startswith('gender[') for col in X.columns))
        self.assertEqual(X.shape[1], 2)

    def test_add_interactions(self):
        terms = ['age', 'income']
        interactions = [('age', 'income')]
        X = transform_df_for_model(self.df, terms, interactions=interactions)
        self.assertIn('age:income', X.columns)

    def test_no_intercept(self):
        terms = ['age']
        X = transform_df_for_model(self.df, terms, add_intercept=False)
        self.assertNotIn('intercept', X.columns)

    def test_error_if_missing_column(self):
        terms = ['age', 'not_present']
        with self.assertRaises(ValueError):
            transform_df_for_model(self.df, terms)

    def test_error_if_not_dataframe(self):
        with self.assertRaises(TypeError):
            transform_df_for_model([1, 2, 3], ['age'])

if __name__ == '__main__':
    unittest.main()
    

   



