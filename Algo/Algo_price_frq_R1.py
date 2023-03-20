from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics as st
import numpy as np

"""NOTE: if volume > 0 -> bid; if volume < 0 -> ask"""

class Trader:
    POS_LIMIT: Dict[str, int] = {"PEARLS": 20, "BANANAS": 20} # Dict: {product_name -> pos_limit} NOTE: need to manually update dict of product limits
    PROD_LIST = ("PEARLS", "BANANAS") # Set: product_names NOTE: need to manually update list of products
    MAX_LOT_SIZE = 20

    def __init__(self) -> None:
        self.state = TradingState
        self.result = {} # tracks order placed in each iteration, {product_name -> List(Order)} NOTE: reassigns to empty dict every iteration
        self.hist_prices = {} # tracks all historical market prices of each product, {product_name -> List(market_price)}
        self.price = {} # tracks current market price of each product, {product_name -> market_price}
        self.mid_price = {} # tracks current market mid price of each product, {product_name -> mid_price}
        self.hist_mid_prices = {} # tracks all historical mid_prices of each product, {product_name -> List(mid_price)} 
        self.hist_vol = {} # tracks all historical volume of each product, {product_name -> List(volume)}
        self.vol = {} # tracks current volume of each product, {product_name -> volume}
        self.position = {} # tracks position of each product, {product_name -> position}
        self.order_depths = {}
        #self.avg_bid_prices = {}
        #self.avg_ask_prices = {}

        self.price_freq = {}
        self.t_price_ct = {}
        self.weight = 1
        self.step = 0.05 #0.05


        for product in self.PROD_LIST: # initialize variables
            self.result[product] = []
            self.hist_prices[product] = []
            self.price[product] = 0
            self.mid_price[product] = 0
            self.hist_mid_prices[product] = []
            self.hist_vol[product] = []
            self.vol[product] = 0
            self.position[product] = 0
            self.order_depths[product] = OrderDepth
            self.price_freq[product] = {}
            self.t_price_ct[product] = 0

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
            print("-"*10)
            print(state.observations)
        #try:
            #-----Data update start-----
            for product in self.PROD_LIST:
                self.result[product] = []
                self.position[product] = 0 
            self.order_depths = state.order_depths

            self.__update_postion(state)
            for product in self.PROD_LIST:
                print(f"product: {product}")
                self.__update_vol_and_price_weighted_by_vol(state, product) # update price of [product]
                self.__update_mid_price(product)
            #-----Data update end
            #-----Algo start-----
            for product in self.PROD_LIST:
                prob_below = 0
                prices = np.fromiter(self.price_freq[product].keys(), dtype=float)
                #print(f"*****{self.price[product]} | {prices} | {self.t_price_ct[product]}")
                if self.price[product] != 0 and self.t_price_ct[product] != 0:
                    prices_below = prices[prices<self.price[product]]
                    #print(f"{product} prices below: {prices_below}")
                    ct = 0
                    for price in prices_below:
                        ct += self.price_freq[product][price]
                    denom = self.t_price_ct[product] - self.price_freq[product][self.price[product]] if self.price[product] in self.price_freq[product].keys() else self.t_price_ct[product]
                    prob_below = ct / denom if denom != 0 else 0.5
                    print(f"prob_below: {prob_below} | {ct} / {denom}")
                    if prob_below < 0.5 and product == "PEARLS":
                        buy_prob = np.interp(prob_below, [0,0.5], [1,0])
                        #buy_prob = 2*buy_prob**2-buy_prob**6
                        #buy_prob = 1 if buy_prob > 1 else buy_prob
                        #print(f"buy at {self.price[product]}")
                        self.place_order(product,self.price[product],self.get_max_bid_size(product)*buy_prob)
                    elif prob_below > 0.5 and product == "PEARLS":
                        sell_prob = np.interp(prob_below, [0.5,1], [0,1])
                        #sell_prob = 2*sell_prob**2-sell_prob**6
                        #sell_prob = 1 if sell_prob > 1 else sell_prob
                        #print(f"sell at {self.price[product]}")
                        self.place_order(product,self.price[product],-self.get_max_ask_size(product)*sell_prob)
                    
            for product in self.PROD_LIST:
                rounded_p = self.round_to_p5(self.price[product])
                
                #print(f"{product} rounded_p: {rounded_p}")
                if rounded_p != 0:
                    if rounded_p in self.price_freq[product].keys():
                        self.price_freq[product][rounded_p] += self.weight
                        
                    else:
                        self.price_freq[product][rounded_p] = self.weight
                    self.t_price_ct[product] += self.weight
                    self.weight += self.step
                #print(self.price_freq[product])
            #-----Algo end
            return self.result
        #except Exception:
        #    self.result = {}
        #    return self.result

    #-----Algo methods start-----
    def write_methods_for_algo_computations_in_this_section(self):
        pass
    def round_to_p5(self, x):
        return round(x,1)
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
        best_bid = 0
        if product in self.order_depths and len(self.order_depths[product].buy_orders) > 0:
            bids = self.order_depths[product].buy_orders
            best_bid = max(bids.keys())     
        return best_bid
    
    def get_best_ask(self, product):
        best_ask = 0
        if product in self.order_depths and len(self.order_depths[product].sell_orders) > 0:
            asks = self.order_depths[product].sell_orders
            best_ask = min(asks.keys())
        return best_ask

    def get_vol_mean(self, product):
        return st.mean(self.hist_vol[product]) if len(self.hist_vol[product]) > 0 else -1
    
    def get_vol_std(self, product):
        return st.stdev(self.hist_vol[product]) if len(self.hist_vol[product]) > 0 else -1
    
    def get_price_mean(self, product):
        return st.mean(self.hist_prices[product]) if len(self.hist_prices[product]) > 0 else -1
    
    def get_price_std(self, product):
        return st.stdev(self.hist_prices[product]) if len(self.hist_prices[product]) > 0 else -1
    
    def get_mid_price_mean(self, product):
        return st.mean(self.hist_mid_prices[product]) if len(self.hist_mid_prices[product]) > 0 else -1
    
    def get_mid_price_std(self, product):
        return st.stdev(self.hist_mid_prices[product]) if len(self.hist_mid_prices[product]) > 0 else -1
    #-----Basic methods end

    #-----Helper methods start (you should not need to call methods below)-----
    def __update_postion(self, state: TradingState):
        for product,pos in state.position.items():
            self.position[product] = pos

    def __update_vol_and_price_weighted_by_vol(self, state: TradingState, product) -> None:
        t_vol = sum_price = 0
        if product in state.own_trades.keys():
            product_own_trades = state.own_trades[product]
            for trade in product_own_trades:
                sum_price += trade.price*abs(trade.quantity)
                t_vol += abs(trade.quantity)
        if product in state.market_trades.keys():
            product_market_trades = state.market_trades[product]
            for trade in product_market_trades:
                sum_price += trade.price*abs(trade.quantity)
                t_vol += abs(trade.quantity)
        if t_vol > 0:
            self.price[product] = sum_price / t_vol
        self.hist_prices[product].append(self.price[product])
        self.vol[product] = t_vol
        self.hist_vol[product].append(t_vol)
    
    def __update_mid_price(self, product) -> None:
        product_bids = product_asks = {}
        max_bid = min_ask = 0
        if product in self.order_depths.keys():
            product_bids = self.order_depths[product].buy_orders
            product_asks = self.order_depths[product].sell_orders
            max_bid = max(product_bids.keys()) if len(product_bids.keys()) > 0 else 0
            min_ask = min(product_asks.keys()) if len(product_asks.keys()) > 0 else 0
            if max_bid > 0 and min_ask > 0:
                self.mid_price[product] = (max_bid + min_ask) / 2
        self.hist_mid_prices[product].append(self.mid_price[product])
    #-----Helper methods end