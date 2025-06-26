# utils.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DateFeatureExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, date_column):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.date_column] = pd.to_datetime(df[self.date_column])
        df["dayofweek"] = df[self.date_column].dt.dayofweek
        df["month"] = df[self.date_column].dt.month
        df["week"] = df[self.date_column].dt.isocalendar().week.astype(int)
        df["year"] = df[self.date_column].dt.year
        df = df.drop(columns=[self.date_column])
        return df
