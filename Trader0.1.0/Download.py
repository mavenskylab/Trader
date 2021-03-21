import os
import pandas as pd
import yfinance as yf

# PATH = "Data/"
# LIST = "STOCK_LIST.txt"

PATH = "../Test/"
LIST = "Test_List.txt"

TICKER_LIST = []

f = open(LIST, "r")
for s in f:
    TICKER_LIST.append(f"{s[:-1]}.L")

for TICKER in TICKER_LIST:
    print(TICKER)
    df = yf.download(TICKER, start="2000-01-01")
    df = df.rename(str.lower, axis="columns")
    df = df.drop(columns=["adj close"])
    df.fillna(method="ffill", inplace=True)
    df.dropna(inplace=True)
    df.to_csv(f"{PATH}{TICKER}.csv", header = True)

print("Done!")