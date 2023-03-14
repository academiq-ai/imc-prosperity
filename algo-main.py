import pandas as pd
from typing import Dict, List
from datamodel import Listing, OrderDepth, Trade, TradingState, Order
from Trader import Trader
from alive_progress import alive_bar


def process_excel_data(row, market_bid_orders, market_ask_orders):

    bid_prices = [
        row[f"bid_price_{i}"] for i in range(1, 4) if not pd.isna(row[f"bid_price_{i}"])
    ]
    bid_volumes = [
        row[f"bid_volume_{i}"]
        for i in range(1, 4)
        if not pd.isna(row[f"bid_volume_{i}"])
    ]
    ask_prices = [
        row[f"ask_price_{i}"] for i in range(1, 4) if not pd.isna(row[f"ask_price_{i}"])
    ]
    ask_volumes = [
        row[f"ask_volume_{i}"]
        for i in range(1, 4)
        if not pd.isna(row[f"ask_volume_{i}"])
    ]

    for price, volume in zip(bid_prices, bid_volumes):
        if price in market_bid_orders:
            market_bid_orders[price] += volume
        else:
            market_bid_orders[price] = volume

    for price, volume in zip(ask_prices, ask_volumes):
        if price in market_ask_orders:
            market_ask_orders[price] += volume
        else:
            market_ask_orders[price] = volume

    return market_ask_orders, market_bid_orders


def generateTradingState(row, market_bid_orders, market_ask_orders):
    timestamp = row.timestamp

    listings = {
        "PEARLS": Listing(symbol="PEARLS", product="PEARLS", denomination="SEASHELLS")
    }

    order_depths = {
        "PEARLS": OrderDepth(
            # buy_orders=market_bid_orders,
            # sell_orders=market_ask_orders
        ),
        "PRODUCT2": OrderDepth(
            # buy_orders={142: 3, 141: 5},
            # sell_orders={144: -5, 145: -8}
        ),
    }

    # TODO: need to fetch from our actual algorithms
    own_trades = {
        "PEARLS": [
            Trade(
                symbol="PEARLS",
                price=11,
                quantity=4,
                buyer="SUBMISSION",
                seller="",
                timestamp=timestamp,
            ),
            Trade(
                symbol="PEARLS",
                price=12,
                quantity=3,
                buyer="SUBMISSION",
                seller="",
                timestamp=timestamp,
            ),
        ],
        "PRODUCT2": [
            Trade(
                symbol="PRODUCT2",
                price=143,
                quantity=2,
                buyer="",
                seller="SUBMISSION",
                timestamp=timestamp,
            ),
        ],
    }

    market_trades = {"PEARLS": []}

    position = {
        "PEARLS": 3,
    }

    observations = {}

    state = TradingState(
        timestamp=timestamp,
        listings=listings,
        order_depths=order_depths,
        own_trades=own_trades,
        market_trades=market_trades,
        position=position,
        observations=observations,
    )

    return state


def update_our_orders(result, market_bid_orders, market_ask_orders, our_bid_orders, our_ask_orders):
    for product, orders in result.items():
        for order in orders:
            if order.price in our_bid_orders: # add to existing bid order
                our_bid_orders[order.price][0] += order.quantity
            elif order.price in our_ask_orders: # add to existing ask order
                our_ask_orders[order.price][0] += order.quantity
            else: # new order
                if order.quantity > 0: # new bid position
                    our_bid_orders[order.price] = [order.quantity, 0] if order.price not in market_bid_orders \
                    else [[order.quantity], [market_bid_orders[order.price]]]
                elif order.quantity < 0: # new ask position
                    our_ask_orders[order.price] = [order.quantity, 0] if order.price not in market_ask_orders \
                    else [[order.quantity], [market_ask_orders[order.price]]]
                #our_orders[order.price] = [order.quantity, 0]


def match_orders(market_bid_orders, market_ask_orders, our_bid_orders, our_ask_orders):
    for price, bid_orders in our_bid_orders.items():
        if price in market_ask_orders:
            ask_orders = market_ask_orders[price]
            while bid_orders[0] > 0 and ask_orders > 0:
                quantity = min(bid_orders[0], ask_orders)
                bid_orders[0] -= quantity
                ask_orders -= quantity
                # TODO: track profit & loss and update our_bid_orders/our_ask_orders accordingly
            our_bid_orders[price][1] = ask_orders
    for price, ask_orders in our_ask_orders.items():
        if price in market_bid_orders:
            bid_orders = market_bid_orders[price]
            while ask_orders[0] > 0 and bid_orders > 0:
                quantity = min(ask_orders[0], bid_orders)
                ask_orders[0] -= quantity
                bid_orders -= quantity
                # TODO: track profit & loss and update our_bid_orders/our_ask_orders accordingly
            our_ask_orders[price][1] = bid_orders


if __name__ == "__main__":
    market_bid_orders = {}
    market_ask_orders = {}
    our_bid_orders = {}
    our_ask_orders = {}
    trader = Trader()

    data = pd.read_csv("./data/market_tutorial_0.csv", header=0, sep=";",)

    # read each line
    with alive_bar(len(data)) as bar:
        for _, row in data.iterrows():
            market_ask_orders, market_bid_orders = process_excel_data(
                row, market_ask_orders, market_bid_orders
            )
            # tradingstate = generateTradingState(
            #     row, market_bid_orders, market_ask_orders
            # )
            # result = trader.run(tradingstate)
            # update_our_orders(result, our_bid_orders, our_ask_orders)
            # match_orders(
            #     market_bid_orders, market_ask_orders, our_bid_orders, our_ask_orders
            # )
            bar()
        print("market_ask_order: ", market_ask_orders)
        print("market_bid_order: ", market_bid_orders)

    # TODO: update market_bid_orders/market_ask_orders with our_bid_orders/our_ask_orders
