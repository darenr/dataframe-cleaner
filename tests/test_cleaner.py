import pytest
import logging
import sys
import pandas as pd
import numpy as np

from dataframe_cleaner.dataframe_cleaner import DataFrameCleaner
logging.basicConfig(level=logging.DEBUG)

def test_direct_marketing():
    marketing = pd.read_csv("data/DirectMarketing.csv")
    cleaned = DataFrameCleaner().clean(marketing)
    assert cleaned.shape == (811, 10)

def test_boston_housing():
    boston = pd.read_csv("data/BostonHousing.csv")
    cleaned = DataFrameCleaner().clean(boston)
    assert cleaned.shape == (177, 14)


def test_titanic():
    titanic = pd.read_csv("data/titanic.csv")
    cleaned = DataFrameCleaner().clean(titanic)
    assert cleaned.shape == (496, 11)
