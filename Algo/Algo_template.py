from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics as st
import numpy as np

"""NOTE: if volume > 0 -> bid; if volume < 0 -> ask"""

class Trader:
    POS_LIMIT: Dict[str, int] = {} # Dict: {product_name -> pos_limit} NOTE: need to manually update dict of product limits
    PROD_LIST = ("PEARLS", "BANANAS") # Set: product_names NOTE: need to manually update list of products
    MAX_LOT_SIZE = 10
    PATTERN_TRACKING_INTERVAL = 5
    PATTERN_MAX_INDEX = 2 ** PATTERN_TRACKING_INTERVAL - 1

    def __init__(self) -> None:
        self.result = {} # tracks order placed in each iteration, {product_name -> List(Order)} NOTE: reassigns to empty dict every iteration
        self.hist_prices = {} # tracks all historical market prices of each product, {product_name -> List(market_price)}
        self.price = {} # tracks current market price of each product, {product_name -> market_price}
        self.hist_vol = {} # tracks all historical volume of each product, {product_name -> List(volume)}
        self.vol = {} # tracks current volume of each product, {product_name -> volume}
        self.position = {} # tracks position of each product, {product_name -> position}
        self.order_depths = {}

        #variables for godric_pattern_tracking()
        self.pattern_net_return_and_ct = {} # first row is net return % of a pattern; second row is number of appearances of a pattern
        self.start_tracking = {}
        self.price_tracked_ct = {}
        self.pattern_interval_ct = {}
        self.pattern = {}
        self.prev_price = {}
        self.pattern_ct = {}
        self.up_frq = {}
        self.down_frq = {}

        for product in self.PROD_LIST: # initialize variables
            self.result[product] = []
            self.hist_prices[product] = []
            self.price[product] = 0
            self.hist_vol = []
            self.vol = 0
            self.position[product] = 0
            self.order_depths[product] = OrderDepth
            self.pattern_net_return_and_ct[product] = np.zeros((2,2**self.PATTERN_TRACKING_INTERVAL))
            self.start_tracking[product] = False
            self.price_tracked_ct[product] = 0
            self.pattern_interval_ct[product] = 0
            self.pattern[product] = 0
            self.prev_price[product] = 0
            self.pattern_ct[product] = 0
            self.up_frq[product] = np.zeros(2)
            self.down_frq[product] = np.zeros(2)

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        #-----Data update start-----
        for product in state.order_depths.keys():
            self.result[product] = []
        self.position = state.position
        self.order_depths = state.order_depths
        for product in state.order_depths.keys():
            self.__update_vol_and_price_weighted_by_vol(self, state, product) # update price of [product
            self.__pattern_tracking(product)
            #update best ask and bid price
        #-----Data update end
        #-----Algo start-----
        for product in state.order_depths.keys():
            if self.pattern_ct[product] > 1000:
                exp_val = self.expected_val_of_pattern(product)
                certainty = self.certainty_of_expected_val(self, product, exp_val)
                if exp_val > 0:
                    best_ask_price = self.get_best_ask(product)
                    vol = self.get_max_bid_size(product)
                    self.place_order(product, best_ask_price, abs(vol*certainty))
                elif exp_val < 0:
                    best_bid_price = self.get_best_bid(product)
                    vol = self.get_max_ask_size(product)
                    self.place_order(product, best_bid_price, -abs(vol*certainty))
        #-----Algo end
        return self.result

    #-----Algo methods start-----
    def write_methods_for_algo_computations_in_this_section(self):
        pass
    def __pattern_tracking(self, product):
        if self.start_tracking[product] == True and self.price_tracked_ct[product] >= (self.PATTERN_TRACKING_INTERVAL + 2):
            if self.price != self.prev_price and self.prev_price != 0:
                perc_change = ((self.price - self.prev_price) / self.prev_price) * 100
                if self.price > self.prev_price:
                    bit = 1
                    self.up_frq[product] += perc_change, 1
                else:
                    bit = 0
                    self.down_frq[product] += perc_change, 1
                self.pattern[product] = self.pattern[product] << 1 | bit
                self.pattern[product] = id = self.pattern[product] & self.PATTERN_MAX_INDEX
                if id <= self.PATTERN_MAX_INDEX:
                    self.pattern_net_return_and_ct[product][:,id] += perc_change, 1
                    self.pattern_ct[product] += 1
                self.prev_price = self.price
        elif self.start_tracking[product] == True and self.price_tracked_ct[product] < (self.PATTERN_TRACKING_INTERVAL + 2):
            if self.price != self.prev_price and self.prev_price != 0:
                perc_change = ((self.price - self.prev_price) / self.prev_price) * 100
                if self.price > self.prev_price:
                    bit = 1
                    self.up_frq[product] += perc_change, 1
                else:
                    bit = 0
                    self.down_frq[product] += perc_change,1
                self.pattern[product] = self.pattern[product] << 1 | bit
                self.pattern[product] = self.pattern[product] & self.PATTERN_MAX_INDEX
                self.price_tracked_ct[product] += 1
                self.prev_price = self.price
        elif self.start_tracking[product] == False and self.price[product] != 0: #ensure start tracking with non-zero number
            self.start_tracking[product] = True
            self.prev_price[product] = self.price[product]
        
    
    """(func) expected_val_of_pattern(product)
            Returns the expected percentage return"""
    def expected_val_of_pattern(self, product):
        if self.start_tracking[product] == True and self.price_tracked_ct[product] >= (self.PATTERN_TRACKING_INTERVAL + 2):
            id = self.pattern[product]
            net_return_perc = self.pattern_net_return_and_ct[product][0,id]
            occurences = self.pattern_net_return_and_ct[product][1,id]
            return net_return_perc / occurences if occurences > 0 else 0
        else:
            return 0 
    def certainty_of_expected_val(self, product, exp_val):
        id = self.pattern[product]
        #exp_val = self.expected_val_of_pattern(product)
        certainty_pattern_freq = self.pattern_net_return_and_ct[product][1,id] / self.pattern_ct[product] if self.pattern_ct[product] > 0 else 0
        certainty_expVal_to_avgRet = 0
        if exp_val > 0 and self.up_frq[product][1] > 0:
            certainty_expVal_to_avgRet = exp_val / (self.up_frq[product][0] / self.up_frq[product][1])
        elif exp_val < 0 and self.down_frq[product][1] > 0:
            certainty_expVal_to_avgRet = exp_val / (self.down_frq[product][0] / self.down_frq[product][1])
        if certainty_expVal_to_avgRet < 0:
            certainty_expVal_to_avgRet = 0
        elif certainty_expVal_to_avgRet > 1:
            certainty_expVal_to_avgRet = 1
        return certainty_expVal_to_avgRet * certainty_pattern_freq
    #-----Algo methods end

    #-----Basic methods start-----
    def place_order(self, product, price, quantity): # NOTE: price and quantity do not need to be integers; this method will take care of it
        if product in self.PROD_LIST and int(round(quantity)) != 0:
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
        best_bid = 99999
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
    
    def get_vol_std(self, product):
        return st.stdev(self.hist_prices[product])
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
    #-----Helper methods end