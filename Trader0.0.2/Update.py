import os
import pandas as pd
import yfinance as yf

PATH = ["Data/", "Test/"]

for path in PATH:
    for filename in os.listdir(path):
        ticker = filename.split(".")[0]
        print(ticker)
        df = yf.download(ticker, start="2000-01-01")
        df = df.rename(str.lower, axis="columns")
        df = df.drop(columns=["adj close"])
        df.to_csv(f"{path}{ticker}.csv", header = True)