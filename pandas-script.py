import pandas as pd
import numpy as np
import pickle


def main():
    convert_trades("./data/round-1-raw/trades_round_1_day_-1_nn.csv", "trades_day1nn")
    convert_trades("./data/round-1-raw/trades_round_1_day_-2_nn.csv", "trades_day2nn")
    convert_trades("./data/round-1-raw/trades_round_1_day_0_nn.csv", "trades_day0nn")
    convert_prices("./data/round-1-raw/prices_round_1_day_-1.csv", "prices_day1")
    convert_prices("./data/round-1-raw/prices_round_1_day_-2.csv", "prices_day2")
    convert_prices("./data/round-1-raw/prices_round_1_day_0.csv", "prices_day0")


def convert_trades(csv_path, file_name):
    df = pd.read_csv(csv_path, delimiter=";")
    df["timestamp"] = pd.to_numeric(df["timestamp"])
    symbols = df["symbol"].unique()
    PEARLS = []
    BANANAS = []

    for symbol in symbols:
        symbol_df = df[df["symbol"] == symbol]
        timestamp_list = symbol_df["timestamp"].unique()
        symbol_dict_list = []

        for timestamp in timestamp_list:
            ts_df = symbol_df[symbol_df["timestamp"] == timestamp]
            price_quantity_dict = {}

            for i, row in ts_df.iterrows():
                price_quantity_dict[row["price"]] = row["quantity"]

            symbol_dict_list.append({"timestamp": timestamp, **price_quantity_dict})

        for i in range(1, len(symbol_dict_list)):
            prev_dict = symbol_dict_list[i - 1]
            curr_dict = symbol_dict_list[i]
            curr_dict.update(
                {k: prev_dict[k] for k in prev_dict.keys() - curr_dict.keys()}
            )

        symbol_dict_list.append({timestamp: -1})

        if symbol == "PEARLS":
            PEARLS = symbol_dict_list
        elif symbol == "BANANAS":
            BANANAS = symbol_dict_list

    with open(f"PEARLS_{file_name}.pkl", "wb") as f:
        pickle.dump(PEARLS, f)

    with open(f"BANANAS{file_name}.pkl", "wb") as f:
        pickle.dump(BANANAS, f)


def convert_prices(csv_path, file_name):
    df = pd.read_csv(csv_path, delimiter=";")
    symbols = df["product"].unique()
    PEARLS = []
    BANANAS = []

    for symbol in symbols:
        symbol_df = df[df["product"] == symbol]
        timestamp_list = symbol_df["timestamp"].unique()
        symbol_dict_list = []

        for timestamp in timestamp_list:
            ts_df = symbol_df[symbol_df["timestamp"] == timestamp]
            buy_orders = {}
            sell_orders = {}

            for _, row in ts_df.iterrows():
                for j in range(1, 4):
                    buy_price_col = f"bid_price_{j}"
                    buy_volume_col = f"bid_volume_{j}"
                    if not pd.isna(row[buy_price_col]):
                        buy_orders[row[buy_price_col]] = row[buy_volume_col]

                    sell_price_col = f"ask_price_{j}"
                    sell_volume_col = f"ask_volume_{j}"
                    if not pd.isna(row[sell_price_col]):
                        sell_orders[row[sell_price_col]] = row[sell_volume_col]

            buy_orders = {float(k): v for k, v in buy_orders.items()}
            sell_orders = {float(k): v for k, v in sell_orders.items()}

            symbol_dict_list.append(
                {
                    "timestamp": timestamp,
                    "buy_orders": buy_orders,
                    "sell_orders": sell_orders,
                }
            )

        symbol_dict_list.append({timestamp: -1})

        if symbol == "PEARLS":
            PEARLS = symbol_dict_list
        elif symbol == "BANANAS":
            BANANAS = symbol_dict_list

    with open(f"./data/round-1-pkl/PEARLS_{file_name}.pkl", "wb") as f:
        pickle.dump(PEARLS, f)

    with open(f"./data/round-1-pkl/BANANAS{file_name}.pkl", "wb") as f:
        pickle.dump(BANANAS, f)


if __name__ == "__main__":
    main()
