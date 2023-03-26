from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics as st
import numpy as np

"""NOTE: if volume > 0 -> bid; if volume < 0 -> ask"""

class Trader:
    POS_LIMIT: Dict[str, int] = {"PEARLS": 20, "BANANAS": 20} # Dict: {product_name -> pos_limit} NOTE: need to manually update dict of product limits
    PROD_LIST = ("PEARLS", "BANANAS") # Set: product_names NOTE: need to manually update list of products
    MAX_LOT_SIZE = 20
    #PERC_ROUND_DEC = 3
    #PATTERN_TRACKING_INTERVAL = 10
    PATTERN_TRACKING_INTERVAL = 4
    NUM_PAT_CLASS = 3
    PATTERN_MAX_INDEX = NUM_PAT_CLASS ** PATTERN_TRACKING_INTERVAL - 1
    START_PT_TRADE_CT = 0

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
        self.last_trade_price = {}
        self.prev_pos = {}
        self.account_balance = 0
        self.pnl = 0

        self.price_freq = {}
        self.t_price_ct = {}
        self.price_chg_freq = {}
        self.t_price_chg_ct = {}
        self.track_start_tf = {}
        self.weight = 1
        self.step = 0.1 #0.05
        self.weight2 = 1
        self.step2 = 0.05 #0.05

        self.pattern_net_return_and_ct = {} # first row is net return % of a pattern; second row is number of appearances of a pattern
        self.start_tracking = {}
        self.price_tracked_ct = {}
        self.pattern_interval_ct = {}
        self.pattern = {}
        self.prev_price = {}
        self.pattern_ct = {}
        self.up_frq = {} #[total up, no. of up days]
        self.down_frq = {}
        self.price_chg_tf = {}
        self.base_convert = np.zeros(self.PATTERN_TRACKING_INTERVAL)
        self.pattern_fill_ct = {}
        for i in reversed(range(self.PATTERN_TRACKING_INTERVAL)):
            self.base_convert[i] = self.NUM_PAT_CLASS ** i

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
            self.price_chg_freq[product] = {}
            self.t_price_chg_ct[product] = 0
            self.track_start_tf[product] = False
            self.last_trade_price[product] = 0
            self.prev_pos[product] = 0

            self.pattern_net_return_and_ct[product] = np.zeros((2,self.NUM_PAT_CLASS**self.PATTERN_TRACKING_INTERVAL))
            self.start_tracking[product] = False
            self.price_tracked_ct[product] = 0
            self.pattern_interval_ct[product] = 0
            self.pattern[product] = [0]*self.PATTERN_TRACKING_INTERVAL
            self.prev_price[product] = 0
            self.pattern_ct[product] = 0
            self.up_frq[product] = np.zeros(2)
            self.down_frq[product] = np.zeros(2)
            self.price_chg_tf[product] = False
            self.pattern_fill_ct[product] = 0

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
                self.__pattern_tracking(product)
            #self.__update_pnl()
            #-----Data update end
            #-----Algo start-----
            #PEARLS
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
                    denom = self.t_price_ct[product] - self.price_freq[product][self.round_to_p5(self.price[product])] if self.round_to_p5(self.price[product]) in self.price_freq[product].keys() else self.t_price_ct[product]
                    prob_below = ct / denom if denom != 0 else 0.5
                    #print(f"prob_below: {prob_below} | {ct} / {denom}")
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
            
            #BANANAS
            for product in self.PROD_LIST:
                #print(f"{product} max bid/ask size: {self.get_max_bid_size(product)} {self.get_max_ask_size(product)}")
                #print(f"{product} best bid/ask price: {self.get_best_bid(product)} {self.get_best_ask(product)}")
                #print(f"{product} price/mean price: {self.price[product]} {self.get_price_mean(product)}")
                #print(f"{product} mid-price/mean mid-price: {self.mid_price[product]} {self.get_mid_price_mean(product)}")
                #print(f"{product} volume/mean volume: {self.vol[product]} {self.get_vol_mean(product)}")
                #print(f"{product} ct/TF: {self.pattern_ct[product]} {self.price_chg_tf[product]}")
                if self.pattern_ct[product] > self.START_PT_TRADE_CT: #and self.price_chg_tf[product] == True:
                    #print(f"{product} max bid/ask size: {self.get_max_bid_size(product)} {self.get_max_ask_size(product)}")
                    exp_val = self.expected_val_of_pattern(product)
                    #print(f"{product} expected value: {exp_val}")
                    certainty = self.certainty_of_expected_val(product, exp_val)# ** (1/2)
                    #print(f"{product} certainty:{certainty}")
                    if exp_val > 0 and product == "BANANAS":
                        #best_ask_price = self.get_best_ask(product)
                        vol = self.get_max_bid_size(product)
                        #self.place_order(product, best_ask_price, abs(vol*certainty))
                        #self.place_order(product, self.get_best_bid(product), abs(vol*certainty))
                        self.place_order(product, self.price[product], abs(vol*certainty))
                    elif exp_val < 0 and product == "BANANAS":
                        #best_bid_price = self.get_best_bid(product)
                        vol = self.get_max_ask_size(product)
                        #self.place_order(product, best_bid_price, -abs(vol*certainty))
                        #self.place_order(product, self.get_best_ask(product), -abs(vol*certainty))
                        self.place_order(product, self.price[product], -abs(vol*certainty))

            """for product in self.PROD_LIST:
                if self.track_start_tf[product] == True:
                    cur_price_chg = 100 * (self.hist_prices[product][-1]-self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL]) / self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL]
                    prob_below = 0
                    prices = np.fromiter(self.price_chg_freq[product].keys(), dtype=float)
                    print(f"*****{round(cur_price_chg, self.PERC_ROUND_DEC)} | {prices} | {self.t_price_chg_ct[product]}")
                    if self.price[product] != 0 and self.t_price_chg_ct[product] != 0:
                        prices_below = prices[prices<cur_price_chg]
                        #print(f"{product} prices below: {prices_below}")
                        ct = 0
                        cur_price_chg = round(cur_price_chg, self.PERC_ROUND_DEC)
                        for price in prices_below:
                            ct += self.price_chg_freq[product][price]
                        denom = self.t_price_chg_ct[product] - self.price_chg_freq[product][cur_price_chg] if cur_price_chg in self.price_chg_freq[product].keys() else self.t_price_chg_ct[product]
                        prob_below = ct / denom if denom != 0 else 0.5
                        print(f"prob_below: {prob_below} | {ct} / {denom}")
                        if prob_below < 0.5 and product == "BANANAS" and len(self.hist_prices[product]) > 20:
                            buy_prob = np.interp(prob_below, [0,0.5], [0,1])
                            #buy_prob = 2*buy_prob**2-buy_prob**6
                            #buy_prob = 1 if buy_prob > 1 else buy_prob
                            #print(f"buy at {self.price[product]}")
                            self.place_order(product,self.price[product],-self.get_max_ask_size(product)*buy_prob)
                        elif prob_below > 0.5 and product == "BANANAS" and len(self.hist_prices[product]) > 20:
                            sell_prob = np.interp(prob_below, [0.5,1], [1,0])
                            #sell_prob = 2*sell_prob**2-sell_prob**6
                            #sell_prob = 1 if sell_prob > 1 else sell_prob
                            #print(f"sell at {self.price[product]}")
                            self.place_order(product,self.price[product],-self.get_max_bid_size(product)*sell_prob)
                        
            for product in self.PROD_LIST:
                if self.track_start_tf[product] == True:
                    if self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL] != 0:
                        price_chg = 100 * (self.hist_prices[product][-1]-self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL]) / self.hist_prices[product][-self.PATTERN_TRACKING_INTERVAL]
                        rounded_p = round(price_chg, self.PERC_ROUND_DEC)
                        
                        #print(f"{product} rounded_p: {rounded_p}")
                        if rounded_p in self.price_chg_freq[product].keys():
                            self.price_chg_freq[product][rounded_p] += self.weight2
                            
                        else:
                            self.price_chg_freq[product][rounded_p] = self.weight2
                        self.t_price_chg_ct[product] += self.weight2
                        self.weight2 += self.step2
                        #print(self.price_chg_freq[product])
                else:
                    prices = np.array(self.hist_prices[product])
                    if len(prices[prices > 0]) >= self.PATTERN_TRACKING_INTERVAL:
                        self.track_start_tf[product] = True"""
            #-----Algo end
            #self.__update_prev_pos()
            return self.result
        #except Exception:
        #    self.result = {}
        #    return self.result

    #-----Algo methods start-----
    def round_to_p5(self, x):
        return round(x,1)
    def __pattern_tracking(self,product):
        perc_change = 0
        if self.start_tracking[product] == True and self.price_tracked_ct[product] >= (self.PATTERN_TRACKING_INTERVAL):
            #print("pattern tracking--phase 3")
            if self.prev_price[product] != 0:
                ##print(1)
                perc_change = ((self.price[product] - self.prev_price[product]) / self.prev_price[product]) * 100
                #print(2)
                self.price_chg_tf[product] = True
                #print(3)
                id = self.__get_pattern_id(product)
                #print(f"id:{id}")
                if id <= self.PATTERN_MAX_INDEX:
                    self.pattern_net_return_and_ct[product][:,id] += perc_change, 1
                    self.pattern_ct[product] += 1
                if self.price[product] > self.prev_price[product]: #increase in price
                    bit = 2
                    self.up_frq[product] += perc_change, 1
                elif self.price[product] < self.prev_price[product]: #decrease in price
                    bit = 1
                    self.down_frq[product] += perc_change, 1
                else: #no change
                    bit = 0
                    self.down_frq[product] += perc_change, 1
                self.pattern[product].pop(0)
                self.pattern[product].append(bit)
                self.prev_price[product] = self.price[product]
        elif self.start_tracking[product] == True and self.price_tracked_ct[product] < (self.PATTERN_TRACKING_INTERVAL):
            #print("pattern tracking--phase 2")
            if self.prev_price[product] != 0:
                perc_change = ((self.price[product] - self.prev_price[product]) / self.prev_price[product]) * 100
                self.price_chg_tf[product] = True
                if self.price[product] > self.prev_price[product]: #increase in price
                    bit = 2
                    self.up_frq[product] += perc_change, 1
                elif self.price[product] < self.prev_price[product]: #decrease in price
                    bit = 1
                    self.down_frq[product] += perc_change, 1
                else: #no change
                    bit = 0
                index = self.pattern_fill_ct[product]
                #print(f"{product} index/length: {index} {len(self.pattern[product])}")
                self.pattern_fill_ct[product] += 1
                if index < len(self.pattern[product]):
                    self.pattern[product][index] = bit
                    self.price_tracked_ct[product] += 1
                self.prev_price[product] = self.price[product]
        elif self.start_tracking[product] == False and self.price[product] != 0: #ensure start tracking with non-zero number
            #print("pattern tracking--phase 1")
            self.start_tracking[product] = True
            self.prev_price[product] = self.price[product]
    
    def __get_pattern_id(self, product):
        #print(self.pattern[product])
        #print(self.base_convert)
        return int(np.dot(self.pattern[product], self.base_convert))

    """(func) expected_val_of_pattern(product)
            Returns the expected percentage return"""
    def expected_val_of_pattern(self, product):
        #print(f"{product} ct/interval: {self.price_tracked_ct[product]} {self.PATTERN_TRACKING_INTERVAL}")
        if self.start_tracking[product] == True and self.price_tracked_ct[product] >= (self.PATTERN_TRACKING_INTERVAL):
            id = self.__get_pattern_id(product)
            net_return_perc = self.pattern_net_return_and_ct[product][0,id]
            occurences = self.pattern_net_return_and_ct[product][1,id]
            return net_return_perc / occurences if occurences > 0 else 0
        else:
            return 0 
    def certainty_of_expected_val(self, product, exp_val):
        id = self.__get_pattern_id(product)
        #exp_val = self.expected_val_of_pattern(product)
        certainty_pattern_freq = self.pattern_net_return_and_ct[product][1,id] / self.pattern_ct[product] if self.pattern_ct[product] > 0 else 0
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
        #print(f"{product} certainty 1/pattern frequency: {certainty_expVal_to_avgRet}({avgRet}%) {certainty_pattern_freq}")
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
            self.last_trade_price[product] = int(round(price))
    
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
    
    """def __update_pnl(self):
        order_trans = {}
        for product in self.PROD_LIST:
            order_trans[product] = self.position[product] - self.prev_pos[product]
            print(f"{product} ordered exec: {order_trans[product]}")
            self.account_balance += self.last_trade_price[product] * -order_trans[product]
        self.pnl = self.account_balance
        for product in self.PROD_LIST:
            self.pnl += self.position[product] * self.price[product]
        print(f",{self.pnl}")
        print(f"pos{self.position}")
        print("-"*10)
    
    def __update_prev_pos(self):
        for product in self.PROD_LIST:
            self.prev_pos[product] = self.position[product]"""

    #-----Helper methods end