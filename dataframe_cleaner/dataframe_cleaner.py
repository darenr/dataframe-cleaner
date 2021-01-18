from typing import List, Dict, Tuple, Sequence, Iterable

import pandas as pd
import numpy as np


class DataFrameCleaner(object):

    def __init__(self):
        pass

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        result = (df.
                pipe(self.copy_df).
                pipe(self.drop_missing_columns).
                pipe(self.to_category)
        )

        for column_name in result.select_dtypes(include='number').columns:
            result = self.remove_outlier_rows(result, column_name)

        return result

    def copy_df(self, df):
        return df.copy()

    def drop_missing_columns(self, df, min_values_percent=40):
        """drops the columns with `min_values_percent` percent or more missing values"""
        thresh = len(df) * ((100 - min_values_percent) / 100)
        df.dropna(axis=1, thresh=thresh, inplace=True)
        return df

    def remove_outlier_rows(self, df, column_name, lower_quantile=0.05, upper_quantile=0.95):
        """keep the values between the 5th and 95th quantiles for numeric types"""
        low = np.quantile(df[column_name].dropna(), 0.05)
        high = np.quantile(df[column_name].dropna(), 0.95)
        return df[df[column_name].between(low, high, inclusive=True)]


    def to_category(self, df):
        cols = df.select_dtypes(include='object').columns
        for col in cols:
            ratio = len(df[col].value_counts()) / len(df)
            if ratio < 0.05:
                df[col] = df[col].astype('category')
        return df
