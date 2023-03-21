import pandas as pd
import numpy as np
import pickle


def main():
    with open("./data/round-1-pkl/BANANAS_trades_day0nn.pkl", "rb") as f:
        data = pickle.load(f)
    print(data)


if __name__ == "__main__":
    main()
