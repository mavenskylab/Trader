import os
import pandas as pd
import yfinance as yf
from yahoofinancials import YahooFinancials

PATH = "Data/"

for filename in os.listdir(PATH):
    ticker = filename.split(".")[0]
    print(ticker)
    df = yf.download(ticker, start="2000-01-01")
    df = df.rename(str.lower, axis="columns")
    df = df.drop(columns=["adj close"])
    df.to_csv(f"{PATH}{ticker}.csv", header = True)