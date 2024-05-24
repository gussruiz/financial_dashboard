import pandas as pd
from functools import partial, reduce
from typing import Callable


class DataSchema:
    AMOUNT = "amount"
    CATEGORY = "category"
    DATE = "date"
    MONTH = "month"
    YEAR = "year"


Preprocessor = Callable[[pd.DataFrame], pd.DataFrame]


def create_year_column(df: pd.DataFrame) -> pd.DataFrame:
    df[DataSchema.YEAR] = df[DataSchema.DATE].dt.year.astype(str)
    return df


def create_month_column(df: pd.DataFrame) -> pd.DataFrame:
    df[DataSchema.MONTH] = df[DataSchema.DATE].dt.month.astype(str)
    return df


def compose(*functions: Preprocessor) -> Preprocessor:
    return reduce(lambda f, g: lambda x: g(f(x)), functions)


def load_transaction_data(path: str) -> pd.DataFrame:
    # load data from csv with pandas
    data = pd.read_csv(path, dtype={
        DataSchema.AMOUNT: float,
        DataSchema.CATEGORY: str
    },
        parse_dates=[DataSchema.DATE]
    )
    preprocessor = compose(create_month_column, create_year_column)

    return preprocessor(data)
