from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics as st
import numpy as np
import math

"""NOTE: if volume > 0 -> bid; if volume < 0 -> ask"""


class Trader:
    POS_LIMIT: Dict[str, int] = {
        "PEARLS": 20,
        "BANANAS": 20,
    }  # Dict: {product_name -> pos_limit} NOTE: need to manually update dict of product limits
    PROD_LIST = (
        "PEARLS",
        "BANANAS",
    )  # Set: product_names NOTE: need to manually update list of products
    MAX_LOT_SIZE = 100
    PEARLS_PRICE = 10000
    BANANA_AVG_INTERVAL = 100
    REFRESH_CT = 20

    def __init__(self) -> None:
        self.state = TradingState
        self.result = (
            {}
        )  # tracks order placed in each iteration, {product_name -> List(Order)} NOTE: reassigns to empty dict every iteration
        self.hist_prices = (
            {}
        )  # tracks all historical market prices of each product, {product_name -> List(market_price)}
        self.price = (
            {}
        )  # tracks current market price of each product, {product_name -> market_price}
        self.mid_price = (
            {}
        )  # tracks current market mid price of each product, {product_name -> mid_price}
        self.hist_mid_prices = (
            {}
        )  # tracks all historical mid_prices of each product, {product_name -> List(mid_price)}
        self.best_ask_price = {}
        self.hist_best_ask_prices = {}
        self.best_bid_price = {}
        self.hist_best_bid_prices = {}
        self.avg_ask_price = {}
        self.hist_avg_ask_prices = {}
        self.avg_bid_price = {}
        self.hist_avg_bid_prices = {}
        self.hist_vol = (
            {}
        )  # tracks all historical volume of each product, {product_name -> List(volume)}
        self.vol = {}  # tracks current volume of each product, {product_name -> volume}
        self.position = (
            {}
        )  # tracks position of each product, {product_name -> position}
        self.order_depths = {}

        # Algo variables
        self.max_trade_spread = {}  # max_trade_coor - min_trade_coor
        self.max_trade_coor = {}
        self.min_trade_coor = {}
        self.trade_coors = {}  # theoratical price as origin
        self.trade_coors = {}  # theoratical price as origin
        self.t_trade_vol = {}
        self.avg_ask_ct = {}
        self.start_trade_tf = {}
        self.trade_weight = 1
        self.ask_weight = 1
        self.bid_step = 0
        self.ask_step = 0
        self.coor_data_refresh_ct = {}

        for product in self.PROD_LIST:  # initialize variables
            self.result[product] = []
            self.hist_prices[product] = []
            self.price[product] = 0
            self.mid_price[product] = 0
            self.hist_mid_prices[product] = []
            self.best_ask_price[product] = 0
            self.hist_best_ask_prices[product] = []
            self.best_bid_price[product] = 0
            self.hist_best_bid_prices[product] = []
            self.avg_ask_price[product] = 0
            self.hist_avg_ask_prices[product] = []
            self.avg_bid_price[product] = 0
            self.hist_avg_bid_prices[product] = []
            self.hist_vol[product] = []
            self.vol[product] = 0
            self.position[product] = 0
            self.order_depths[product] = OrderDepth

            self.max_trade_spread[product] = 0
            self.max_trade_coor[product] = 0
            self.min_trade_coor[product] = 999999999999
            self.trade_coors[product] = {}
            self.t_trade_vol[product] = 0
            self.avg_ask_ct[product] = 0
            self.start_trade_tf[product] = False
            self.coor_data_refresh_ct[product] = 0

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # try:
        # -----Data update start-----
        for product in self.PROD_LIST:
            self.result[product] = []
            self.position[product] = 0
        self.order_depths = state.order_depths
        self.timestamp = state.timestamp

        self.__update_postion(state)
        for product in self.PROD_LIST:
            print(f"product: {product}")
            self.__update_vol_and_price_weighted_by_vol(
                state, product
            )  # update price of [product]
            self.__update_mid_price(product)
        # -----Data update end

        # -----Algo start-----
        # place order
        print(state.observations)
        print(self.position)
        for product in self.PROD_LIST:
            theo_p = self.get_cur_theo_p(product)
            should_trade = (
                self.start_trade_tf[product]
                and theo_p != 0
                and self.price != 0
                and len(self.trade_coors[product]) > 1
            )
            if product == "BANANAS":
                if should_trade:
                    bid_price, ask_price = self.get_optimal_bid_ask_price(
                        product, theo_p
                    )
                    # calculating risk P(!S | B), prob of not being able to sell after we buy, when we're fucked.
                    trade_vol = self.t_trade_vol[product]
                    if trade_vol > 0:
                        bid_trade_vol = self.trade_coors[product].get(
                            bid_price - theo_p, 0
                        )
                        ask_trade_vol = self.trade_coors[product].get(
                            ask_price - theo_p, 0
                        )
                        prob_b = bid_trade_vol / trade_vol
                        prob_a = ask_trade_vol / trade_vol
                        if prob_b > 0 and prob_a > 0:
                            certainty = prob_b * prob_a * (1 / prob_b + 1 / prob_a)
                        else:
                            certainty = 0
                    else:
                        prob_b, prob_a, certainty = (
                            0,
                            0,
                            0,
                        )
                    remain_bid_size = self.get_max_bid_size(product)
                    remain_ask_size = self.get_max_ask_size(product)

                    certainty = max(0, min(certainty, 1))
                    prob_a = max(0, min(prob_a, 1))
                    prob_b = max(0, min(prob_b, 1))

                    if product != "PEARLS":
                        self.start_trade_tf[product] = False
                        self.coor_data_refresh_ct[product] = 0
                        self.max_trade_coor[product] = 0
                        self.min_trade_coor[product] = 999999999999
                        self.trade_coors[product] = {}
                        self.max_trade_spread[product] = 0
            elif (
                self.start_trade_tf[product]
                and theo_p != 0
                and self.price != 0
                and len(self.trade_coors[product]) > 1
            ):
                trade_vol = self.t_trade_vol[product]
                if trade_vol > 0:
                    bid_price, ask_price = self.get_optimal_bid_ask_price(
                        product, theo_p
                    )
                    bid_trade_vol = self.trade_coors[product].get(bid_price - theo_p, 0)
                    ask_trade_vol = self.trade_coors[product].get(ask_price - theo_p, 0)
                    prob_b = bid_trade_vol / trade_vol
                    prob_a = ask_trade_vol / trade_vol
                    certainty = (
                        prob_b * prob_a * (1 / prob_b + 1 / prob_a)
                        if prob_b and prob_a
                        else 0
                    )
                else:
                    prob_b, prob_a, certainty = 0, 0, 0

                prob_a = min(max(prob_a, 0), 1)
                prob_b = min(max(prob_b, 0), 1)
                certainty = min(max(certainty, 0), 1)

                remain_bid_size = self.get_max_bid_size(product)
                remain_ask_size = self.get_max_ask_size(product)
                self.place_order(product, bid_price, prob_a * abs(remain_bid_size))
                self.place_order(product, ask_price, -prob_b * abs(remain_ask_size))

        # update dataset
        # treat theo_p as origin so everything below is negative, above is positive (like coordinatees)
        for product in self.PROD_LIST:
            theo_p = self.get_prev_theo_p(product)
            if theo_p != 0 and (product in state.market_trades):
                for trade in state.market_trades[product]:
                    if trade.timestamp >= state.timestamp - 100:
                        trade_coor = self.custom_rd(trade.price - theo_p)
                        self.max_trade_coor[product] = (
                            trade_coor
                            if trade_coor > self.max_trade_coor[product]
                            else self.max_trade_coor[product]
                        )
                        self.min_trade_coor[product] = (
                            trade_coor
                            if trade_coor < self.min_trade_coor[product]
                            else self.min_trade_coor[product]
                        )
                        self.max_trade_spread[product] = (
                            self.max_trade_coor[product] - self.min_trade_coor[product]
                        )

                        if trade_coor in self.trade_coors[product]:
                            self.trade_coors[product][trade_coor] += abs(trade.quantity)
                        else:
                            self.trade_coors[product][trade_coor] = abs(trade.quantity)
                        self.t_trade_vol[product] += abs(trade.quantity)
                        # self.trade_weight += self.bid_step
            if theo_p != 0 and (product in state.own_trades):
                for trade in state.own_trades[product]:
                    # * to avoid double-counting order excecuted
                    if trade.timestamp >= state.timestamp - 100:
                        print(f"{product} traded")
                        trade_coor = self.custom_rd(trade.price - theo_p)
                        # * tracking max coord and min coord
                        self.max_trade_coor[product] = (
                            trade_coor
                            if trade_coor > self.max_trade_coor[product]
                            else self.max_trade_coor[product]
                        )
                        self.min_trade_coor[product] = (
                            trade_coor
                            if trade_coor < self.min_trade_coor[product]
                            else self.min_trade_coor[product]
                        )
                        self.max_trade_spread[product] = (
                            self.max_trade_coor[product] - self.min_trade_coor[product]
                        )

                        if trade_coor in self.trade_coors[product]:
                            self.trade_coors[product][trade_coor] += abs(trade.quantity)
                        else:
                            self.trade_coors[product][trade_coor] = abs(trade.quantity)
                        self.t_trade_vol[product] += abs(trade.quantity)
                        # self.trade_weight += self.bid_step
            self.coor_data_refresh_ct[product] += 1
            if (
                self.start_trade_tf[product] == False
                and self.max_trade_spread[product] > 0
            ):
                self.start_trade_tf[product] = True
            # """if (product != "PEARLS" and product != "BANANAS") and self.coor_data_refresh_ct[product] >= self.REFRESH_CT:
            #     self.start_trade_tf[product] = False
            #     self.coor_data_refresh_ct[product] = 0
            #     self.max_trade_coor[product] = 0
            #     self.min_trade_coor[product] = 999999999999
            #     self.trade_coors[product] = {}
            #     self.max_trade_spread[product] = 0
            #     self.t_trade_vol[product] = 0
            # print(self.trade_coors[product])"""

        # -----Algo end
        return self.result

    # except Exception:
    #    self.result = {}
    #    return self.result

    # -----Algo methods start-----
    def custom_rd(self, num):
        return round(num * 2, 3) / 2

    def get_cur_theo_p(self, product):
        if product == "PEARLS":
            return self.PEARLS_PRICE  #
        else:  # elif product == "BANANAS":
            # return self.price[product]
            return self.mid_price[
                product
            ]  # st.mean(self.hist_prices[product][-(1+self.BANANA_AVG_INTERVAL):-1]) if len(self.hist_prices[product]) > self.BANANA_AVG_INTERVAL else 0
        """elif product == "COCONUTS":
            return self.price["PINA_COLADAS"]
        elif product == "PINA_COLADAS":
            return self.price["COCONUTS"]"""

    def get_prev_theo_p(self, product):
        if product == "PEARLS":
            return self.PEARLS_PRICE  #
        else:  # elif product == "BANANAS":
            # return self.price[product]
            # return st.mean(self.hist_prices[product][-(1+self.BANANA_AVG_INTERVAL):-1]) if len(self.hist_prices[product]) > self.BANANA_AVG_INTERVAL else 0
            return (
                self.hist_mid_prices[product][-2]
                if len(self.hist_mid_prices[product]) > 1
                else 0
            )
        """elif product == "COCONUTS":
            return self.price["PINA_COLADAS"]
        elif product == "PINA_COLADAS":
            return self.price["COCONUTS"]"""

    def get_exp_val(self, product, price, theo_price):
        exp_val = 0
        trade_coors = np.array(list(self.trade_coors[product].keys()))
        trade_vols = np.array(list(self.trade_coors[product].values()))
        new_base = price - theo_price
        rebased_trade_coors = trade_coors - new_base
        exp_val = (
            np.dot(rebased_trade_coors, trade_vols) / self.t_trade_vol[product]
            if self.t_trade_vol[product] > 0
            else 0
        )
        return exp_val

    def get_optimal_bid_ask_price(self, product, theo_price):
        trade_coors = np.array(list(self.trade_coors[product].keys()))
        exp_vals = np.zeros(len(self.trade_coors[product]))
        id = 0
        for trade_coor in trade_coors:
            exp_vals[id] = (
                self.get_exp_val(product, trade_coor + theo_price, theo_price)
                * (self.trade_coors[product][trade_coor] / self.t_trade_vol[product])
                if self.t_trade_vol[product] > 0
                else 0
            )
            id += 1
        best_bid_price = trade_coors[np.argmax(exp_vals)] + theo_price
        best_ask_price = trade_coors[np.argmin(exp_vals)] + theo_price
        print(f"{product}: {best_bid_price} {best_ask_price}")
        return best_bid_price, best_ask_price

    # -----Algo methods end

    # -----Basic methods start-----
    def place_order(
        self, product, price, quantity
    ):  # NOTE: price and quantity do not need to be integers; this method will take care of it
        if (
            product in self.PROD_LIST
            and int(round(quantity)) != 0
            and int(round(price)) > 0
        ):
            self.result[product].append(
                Order(product, int(round(price)), int(round(quantity)))
            )

    def get_max_bid_size(self, product):
        net_bid_outstanding = 0
        for order in self.result[product]:
            net_bid_outstanding += order.quantity if order.quantity > 0 else 0
        net_max_bid_size = (
            self.POS_LIMIT[product] - self.position[product] - net_bid_outstanding
        )
        net_max_bid_size = net_max_bid_size if net_max_bid_size > 0 else 0
        return min(self.MAX_LOT_SIZE, net_max_bid_size)

    def get_max_ask_size(self, product):
        net_ask_outstanding = 0
        for order in self.result[product]:
            net_ask_outstanding += order.quantity if order.quantity < 0 else 0
        net_max_ask_size = (
            self.POS_LIMIT[product] + self.position[product] + net_ask_outstanding
        )
        net_max_ask_size = net_max_ask_size if net_max_ask_size > 0 else 0
        return min(
            self.MAX_LOT_SIZE, net_max_ask_size
        )  # NOTE: value is positive although it is ask volume

    def get_best_bid(self, product):
        best_bid = 0
        if (
            product in self.order_depths
            and len(self.order_depths[product].buy_orders) > 0
        ):
            bids = self.order_depths[product].buy_orders
            best_bid = max(bids.keys())
        return best_bid

    def get_best_ask(self, product):
        best_ask = 0
        if (
            product in self.order_depths
            and len(self.order_depths[product].sell_orders) > 0
        ):
            asks = self.order_depths[product].sell_orders
            best_ask = min(asks.keys())
        return best_ask

    def get_vol_mean(self, product):
        return (
            st.mean(self.hist_vol[product]) if len(self.hist_vol[product]) > 0 else -1
        )

    def get_vol_std(self, product):
        return (
            st.stdev(self.hist_vol[product]) if len(self.hist_vol[product]) > 1 else -1
        )

    def get_price_mean(self, product):
        return (
            st.mean(self.hist_prices[product])
            if len(self.hist_prices[product]) > 0
            else -1
        )

    def get_price_std(self, product):
        return (
            st.stdev(self.hist_prices[product])
            if len(self.hist_prices[product]) > 1
            else -1
        )

    def get_mid_price_mean(self, product):
        return (
            st.mean(self.hist_mid_prices[product])
            if len(self.hist_mid_prices[product]) > 0
            else -1
        )

    def get_mid_price_std(self, product):
        return (
            st.stdev(self.hist_mid_prices[product])
            if len(self.hist_mid_prices[product]) > 1
            else -1
        )

    def get_best_bid_mean(self, product):
        return (
            st.mean(self.hist_best_bid_prices[product])
            if len(self.hist_best_bid_prices[product]) > 0
            else -1
        )

    def get_best_bid_std(self, product):
        return (
            st.stdev(self.hist_best_bid_prices[product])
            if len(self.hist_best_bid_prices[product]) > 1
            else -1
        )

    def get_best_ask_mean(self, product):
        return (
            st.mean(self.hist_best_ask_prices[product])
            if len(self.hist_best_ask_prices[product]) > 0
            else -1
        )

    def get_best_ask_std(self, product):
        return (
            st.stdev(self.hist_best_ask_prices[product])
            if len(self.hist_best_ask_prices[product]) > 1
            else -1
        )

    def get_avg_bid_mean(self, product):
        return (
            st.mean(self.hist_avg_bid_prices[product])
            if len(self.hist_avg_bid_prices[product]) > 0
            else -1
        )

    def get_avg_bid_std(self, product):
        return (
            st.stdev(self.hist_avg_bid_prices[product])
            if len(self.hist_avg_bid_prices[product]) > 1
            else -1
        )

    def get_avg_ask_mean(self, product):
        return (
            st.mean(self.hist_avg_ask_prices[product])
            if len(self.hist_avg_ask_prices[product]) > 0
            else -1
        )

    def get_avg_ask_std(self, product):
        return (
            st.stdev(self.hist_avg_ask_prices[product])
            if len(self.hist_avg_ask_prices[product]) > 1
            else -1
        )

    # -----Basic methods end

    # -----Helper methods start (you should not need to call methods below)-----
    def __update_postion(self, state: TradingState):
        for product, pos in state.position.items():
            self.position[product] = pos

    def __update_vol_and_price_weighted_by_vol(
        self, state: TradingState, product
    ) -> None:
        t_vol = sum_price = 0
        if product in state.own_trades.keys():
            product_own_trades = state.own_trades[product]
            for trade in product_own_trades:
                sum_price += trade.price * abs(trade.quantity)
                t_vol += abs(trade.quantity)
        if product in state.market_trades.keys():
            product_market_trades = state.market_trades[product]
            for trade in product_market_trades:
                sum_price += trade.price * abs(trade.quantity)
                t_vol += abs(trade.quantity)
        if t_vol > 0:
            self.price[product] = sum_price / t_vol
        self.hist_prices[product].append(self.price[product])
        self.vol[product] = t_vol
        self.hist_vol[product].append(t_vol)

    def __update_mid_price(self, product) -> None:
        product_bids = product_asks = {}
        max_bid = min_ask = avg_ask = avg_bid = bid_sum = bid_ct = ask_sum = ask_ct = 0
        if product in self.order_depths.keys():
            product_bids = self.order_depths[product].buy_orders
            product_asks = self.order_depths[product].sell_orders
            max_bid = max(product_bids.keys()) if len(product_bids.keys()) > 0 else 0
            min_ask = min(product_asks.keys()) if len(product_asks.keys()) > 0 else 0
            for price, vol in product_bids.items():
                bid_sum += price * abs(vol)
                bid_ct += abs(vol)
            for price, vol in product_asks.items():
                ask_sum += price * abs(vol)
                ask_ct += abs(vol)
            if max_bid > 0 and min_ask > 0:
                self.mid_price[product] = (max_bid + min_ask) / 2
            elif max_bid > 0 or min_ask > 0:
                self.mid_price[product] = max_bid + min_ask
            if bid_ct > 0:
                avg_bid = bid_sum / bid_ct
            if ask_ct > 0:
                avg_ask = ask_sum / ask_ct
        self.hist_mid_prices[product].append(self.mid_price[product])
        self.best_ask_price[product] = min_ask
        self.hist_best_ask_prices[product].append(self.best_ask_price[product])
        self.best_bid_price[product] = max_bid
        self.hist_best_bid_prices[product].append(self.best_bid_price[product])
        self.avg_ask_price[product] = avg_ask
        self.hist_avg_ask_prices[product].append(avg_ask)
        self.avg_bid_price[product] = avg_bid
        self.hist_avg_bid_prices[product].append(avg_bid)

    # -----Helper methods end

