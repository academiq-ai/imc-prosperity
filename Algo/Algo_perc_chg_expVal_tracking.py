from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics as st
import numpy as np

"""NOTE: if volume > 0 -> bid; if volume < 0 -> ask"""
"""This algo tracks the percentage change in price for every [PATTERN_TRACKING_INTERVAL] interval
    and track the expected value of each percentage change. It then places bets based on the 
    expected value of the percentage change of the current interval"""

class Trader:
    POS_LIMIT: Dict[str, int] = {"PEARLS": 20, "BANANAS": 20, "COCONUTS": 600, "PINA_COLADAS":300} # Dict: {product_name -> pos_limit} NOTE: need to manually update dict of product limits
    PROD_LIST = ("PEARLS", "BANANAS", "COCONUTS", "PINA_COLADAS") # Set: product_names NOTE: need to manually update list of products
    MAX_LOT_SIZE = 1000
    PATTERN_TRACKING_INTERVAL = 5
    PATTERN_TRACKING_INTERVAL += 1
    PERC_ROUND_DEC = 2

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

        self.price_chg_freq = {}
        self.t_price_chg_ct = {}
        self.track_start_tf = {}
        self.weight2 = 1
        self.step2 = 0 #0.05 #0.05
        self.up_frq = {} #[total up, no. of up days]
        self.down_frq = {}
        self.pattern_ct = {}
        #self.pattern_net_return_and_ct = {}


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
            self.price_chg_freq[product] = {}
            self.t_price_chg_ct[product] = 0
            self.track_start_tf[product] = False
            self.up_frq[product] = np.zeros(2)
            self.down_frq[product] = np.zeros(2)
            self.pattern_ct[product] = 0
            #self.pattern_net_return_and_ct[product] = [1,id]

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
            #print("-"*10)
            #print(state.observations)
        #try:
            #-----Data update start-----
            for product in self.PROD_LIST:
                self.result[product] = []
                self.position[product] = 0 
            self.order_depths = state.order_depths

            self.__update_postion(state)
            for product in self.PROD_LIST:
                #print(f"product: {product}")
                self.__update_vol_and_price_weighted_by_vol(state, product) # update price of [product]
                self.__update_mid_price(product)
            #-----Data update end
            #-----Algo start-----
            print(f"position:{self.position}")
            num = self.price["BANANAS"]
            #print(f",{num}")
            for product in self.PROD_LIST:
                if self.track_start_tf[product] == True and self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL+1] != 0:
                    cur_price_chg = 100 * (self.hist_prices[product][-1]-self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL+1]) / self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL+1]
                    cur_price_chg = round(cur_price_chg*2, self.PERC_ROUND_DEC)/2
                    #prob_below = 0
                    #prices = np.fromiter(self.price_chg_freq[product].keys(), dtype=float)
                    #print(f"*****{round(cur_price_chg, self.PERC_ROUND_DEC)} | {prices} | {self.t_price_chg_ct[product]}")
                    if self.price[product] != 0 and self.t_price_chg_ct[product] != 0 and cur_price_chg in self.price_chg_freq[product].keys():
                        #prices_below = prices[prices<cur_price_chg]
                        #print(f"{product} prices below: {prices_below}")
                        #ct = 0
                        #for price in prices_below:
                        #    ct += self.price_chg_freq[product][price]
                        #denom = self.t_price_chg_ct[product] - self.price_chg_freq[product][cur_price_chg] if cur_price_chg in self.price_chg_freq[product].keys() else self.t_price_chg_ct[product]
                        #prob_below = ct / denom if denom != 0 else 0.5
                        #print(f"prob_below: {prob_below} | {ct} / {denom}")
                        
                        exp_val = self.price_chg_freq[product][cur_price_chg][0] / self.price_chg_freq[product][cur_price_chg][1] if self.price_chg_freq[product][cur_price_chg][1] > 0 else 0
                        exp_val,ct = self.find_exp_val(product, cur_price_chg)
                        certainty = self.certainty_of_expected_val(product, exp_val, cur_price_chg,ct)
                        print(f"certainty: {certainty}")
                        #print(f"cur_price_chg: {self.price_chg_freq[product]}")
                        if exp_val > 0:
                            #best_ask_price = self.get_best_ask(product)
                            vol = self.get_max_bid_size(product)
                            #self.place_order(product, best_ask_price, abs(vol*certainty))
                            #self.place_order(product, self.get_best_bid(product), abs(vol*certainty))
                            self.place_order(product, self.price[product], abs(vol*certainty))
                        elif exp_val < 0:
                            #best_bid_price = self.get_best_bid(product)
                            vol = self.get_max_ask_size(product)
                            #self.place_order(product, best_bid_price, -abs(vol*certainty))
                            #self.place_order(product, self.get_best_ask(product), -abs(vol*certainty))
                            self.place_order(product, self.price[product], -abs(vol*certainty))
                        
            for product in self.PROD_LIST:
                if self.track_start_tf[product] == True:
                    if self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL] != 0:
                        price_chg = 100 * (self.hist_prices[product][-2]-self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL]) / self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL]
                        rounded_p = round(cur_price_chg*2, self.PERC_ROUND_DEC)/2
                        
                        cur_price_chg = ((self.price[product] - self.hist_prices[product][-2]) / self.hist_prices[product][-2]) * 100 if self.hist_prices[product][-2] != 0 else 0
                        #print(f"{product} rounded_p: {rounded_p}")
                        if rounded_p in self.price_chg_freq[product].keys():
                            self.price_chg_freq[product][rounded_p] += cur_price_chg,self.weight2
                        else:
                            self.price_chg_freq[product][rounded_p] = np.array([cur_price_chg,self.weight2])
                        self.pattern_ct[product] += self.weight2
                        self.t_price_chg_ct[product] += self.weight2
                        self.weight2 += self.step2
                        if cur_price_chg > 0: #increase in price
                            self.up_frq[product] += cur_price_chg, 1
                        elif cur_price_chg < 0: #decrease in price
                            self.down_frq[product] += cur_price_chg, 1
                        #print(self.price_chg_freq[product])
                else:
                    prices = np.array(self.hist_prices[product])
                    if len(prices[prices > 0]) >= self.PATTERN_TRACKING_INTERVAL:
                        self.track_start_tf[product] = True
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
    
    def find_exp_val(self,product, cur_price_chg):
        price_chgs = np.fromiter(self.price_chg_freq[product].keys(), dtype=float)
        sum = ct = 0
        if cur_price_chg > 0:
            price_chgs = price_chgs[price_chgs > 0]
            price_chgs = price_chgs[price_chgs<cur_price_chg]
        elif cur_price_chg < 0:
            price_chgs = price_chgs[price_chgs < 0]
            price_chgs = price_chgs[price_chgs>cur_price_chg]
        else:
            price_chgs = [0]
        for price in price_chgs:
            sum += self.price_chg_freq[product][price][0]
            ct += self.price_chg_freq[product][price][1]
        exp_val = sum / ct if ct != 0 else 0
        return exp_val, ct
    
    def certainty_of_expected_val(self, product, exp_val, cur_price_chg, ct):
        #denom = self.pattern_ct[product] - self.price_chg_freq[product][0][1]
        #certainty_pattern_freq = self.price_chg_freq[product][cur_price_chg][1] / denom if denom > 0 else 0
        denom = self.pattern_ct[product] - self.price_chg_freq[product][0][1] if cur_price_chg != 0 else self.pattern_ct[product]
        certainty_pattern_freq = ct / denom if denom > 0 else 0
        certainty_expVal_to_avgRet = 0
        avgRet = None
        if exp_val != 0:
            if exp_val > 0 and self.up_frq[product][1] > 0:
                avgRet = self.up_frq[product][0] / self.up_frq[product][1]
                certainty_expVal_to_avgRet = exp_val / avgRet
            elif exp_val < 0 and self.down_frq[product][1] > 0:
                avgRet = self.down_frq[product][0] / self.down_frq[product][1]
                certainty_expVal_to_avgRet = exp_val / avgRet
            if certainty_expVal_to_avgRet < 0:
                certainty_expVal_to_avgRet = 0
            elif certainty_expVal_to_avgRet > 1:
                certainty_expVal_to_avgRet = 1
        if product == self.PROD_LIST[-1] or product == self.PROD_LIST[-2]:
            print(f"{product} expVal(avgRet)/ct: {exp_val}%({avgRet}%) {ct}")
            print(f"{product} price_chg_freq: {self.price_chg_freq[product]}")
        #return (certainty_expVal_to_avgRet)*certainty_pattern_freq
        num = (certainty_expVal_to_avgRet) if certainty_pattern_freq >= 0.1 else 0
        if num > 1:
            num = 1
        elif num < 0:
            num = 0
        return num
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