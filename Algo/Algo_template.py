from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics as st
import numpy as np

"""NOTE: if volume > 0 -> bid; if volume < 0 -> ask"""

class Trader:
    POS_LIMIT: Dict[str, int] = {} # Dict: {product_name -> pos_limit} NOTE: need to manually update dict of product limits
    PROD_LIST = ("PEARLS", "BANANAS") # Set: product_names NOTE: need to manually update list of products
    MAX_LOT_SIZE = 10

    def __init__(self) -> None:
        self.result = {} # tracks order placed in each iteration, {product_name -> List(Order)} NOTE: reassigns to empty dict every iteration
        self.hist_prices = {} # tracks all historical market prices of each product, {product_name -> List(market_price)}
        self.price = {} # tracks current market price of each product, {product_name -> market_price}
        self.mid_price = {} # tracks current market mid price of each product, {product_name -> mid_price}
        self.hist_mid_prices = {} # tracks all historical mid_prices of each product, {product_name -> List(mid_price)} 
        self.hist_vol = {} # tracks all historical volume of each product, {product_name -> List(volume)}
        self.vol = {} # tracks current volume of each product, {product_name -> volume}
        self.position = {} # tracks position of each product, {product_name -> position}
        self.order_depths = {}

        for product in self.PROD_LIST: # initialize 0/empty list
            self.result[product] = []
            self.hist_prices[product] = []
            self.price[product] = 0
            self.mid_price[product] = 0
            self.hist_mid_prices[product] = []
            self.hist_vol = []
            self.vol = 0
            self.position[product] = 0
            self.order_depths[product] = OrderDepth

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        #-----Data update start-----
        for product in state.order_depths.keys():
            self.result[product] = []
        self.position = state.position
        self.order_depths = state.order_depths
        for product in state.order_depths.keys():
            self.__update_vol_and_price_weighted_by_vol(self, state, product) # update price of [product]
            self.__update_mid_price(product)
        #-----Data update end
        #-----Algo start-----

        #-----Algo end
        return self.result

    #-----Algo methods start-----
    def write_methods_for_algo_computations_in_this_section(self):
        pass
    #-----Algo methods end

    #-----Basic methods start-----
    def place_order(self, product, price, quantity): # NOTE: price and quantity do not need to be integers; this method will take care of it
        if product in self.PROD_LIST and int(round(quantity)) != 0 and int(round(price)) != 0:
            self.result[product].append(Order(product, int(round(price)), int(round(quantity))))
    
    def get_max_bid_size(self, product):
        net_bid_outstanding = 0
        for order in self.result[product]:
            net_bid_outstanding += order.quantity if order.quantity > 0 else 0
        net_max_bid_size = self.POS_LIMIT[product] - self.position[product] - net_bid_outstanding
        net_max_bid_size = net_max_bid_size if net_max_bid_size > 0 else 0
        return min(self.MAX_LOT_SIZE, net_max_bid_size)
    
    def get_max_ask_size(self, product):
        net_ask_outstanding = 0
        for order in self.result[product]:
            net_ask_outstanding += order.quantity if order.quantity < 0 else 0
        net_max_ask_size = self.POS_LIMIT[product] + self.position[product] + net_ask_outstanding
        net_max_ask_size = net_max_ask_size if net_max_ask_size > 0 else 0
        return min(self.MAX_LOT_SIZE, net_max_ask_size) # NOTE: value is positive although it is ask volume
    
    def get_best_bid(self, product):
        bids = self.order_depths[product].buy_orders
        best_bid = 0
        if len(bids) > 0:
            best_bid = max(bids.keys())
        return best_bid
    
    def get_best_ask(self, product):
        asks = self.order_depths[product].sell_orders
        best_ask = 0
        if len(asks) > 0:
            best_ask = min(asks.keys())
        return best_ask

    def get_vol_mean(self, product):
        return st.mean(self.hist_vol[product])
    
    def get_vol_std(self, product):
        return st.stdev(self.hist_vol[product])
    
    def get_price_mean(self, product):
        return st.mean(self.hist_prices[product])
    
    def get_price_std(self, product):
        return st.stdev(self.hist_prices[product])
    
    def get_mid_price_mean(self, product):
        return st.mean(self.hist_mid_prices[product])
    
    def get_mid_price_std(self, product):
        return st.stdev(self.hist_mid_prices[product])
    #-----Basic methods end

    #-----Helper methods start (you should not need to call methods below)-----
    def __update_vol_and_price_weighted_by_vol(self, state, product) -> None:
        product_own_trades = state.own_trades[product]
        product_market_trades = state.market_trades[product]
        t_vol = 0
        sum_price = 0
        for trade in product_own_trades:
            sum_price += trade.price*abs(trade.quantity)
            t_vol += abs(trade.quantity)
        for trade in product_market_trades:
            sum_price += trade.price*abs(trade.quantity)
            t_vol += abs(trade.quantity)
        if t_vol > 0:
            self.price[product] = sum_price / t_vol
        self.hist_prices[product].append(self.price[product])
        self.vol[product] = t_vol
        self.hist_vol[product].append(t_vol)

    def __update_mid_price(self, product) -> None:
        product_bids = self.order_depths[product].buy_orders
        product_asks = self.order_depths[product].sell_orders
        if len(product_bids) > 0 and len(product_asks) > 0:
            self.mid_price[product] = (max(product_bids.keys()) + min(product_asks.keys())) / 2
            self.hist_mid_prices[product].append(self.mid_price)
    #-----Helper methods end