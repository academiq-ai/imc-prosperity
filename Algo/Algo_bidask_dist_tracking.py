from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics as st
import numpy as np

"""NOTE: if volume > 0 -> bid; if volume < 0 -> ask"""

class Trader:
    POS_LIMIT: Dict[str, int] = {"PEARLS": 20, "BANANAS": 20} # Dict: {product_name -> pos_limit} NOTE: need to manually update dict of product limits
    PROD_LIST = ("PEARLS", "BANANAS") # Set: product_names NOTE: need to manually update list of products
    MAX_LOT_SIZE = 10
    PEARL_PRICE = 10000

    def __init__(self) -> None:
        self.state = TradingState
        self.result = {} # tracks order placed in each iteration, {product_name -> List(Order)} NOTE: reassigns to empty dict every iteration
        self.hist_prices = {} # tracks all historical market prices of each product, {product_name -> List(market_price)}
        self.price = {} # tracks current market price of each product, {product_name -> market_price}
        self.mid_price = {} # tracks current market mid price of each product, {product_name -> mid_price}
        self.hist_mid_prices = {} # tracks all historical mid_prices of each product, {product_name -> List(mid_price)} 
        self.best_ask_price = {}
        self.hist_best_ask_prices = {}
        self.best_bid_price = {}
        self.hist_best_bid_prices = {}
        self.avg_ask_price = {}
        self.hist_avg_ask_prices = {}
        self.avg_bid_price = {}
        self.hist_avg_bid_prices = {}
        self.hist_vol = {} # tracks all historical volume of each product, {product_name -> List(volume)}
        self.vol = {} # tracks current volume of each product, {product_name -> volume}
        self.position = {} # tracks position of each product, {product_name -> position}
        self.order_depths = {}

        #Algo variables
        self.max_avg_bid_ask_spread = {} #max_avg_bid - min_avg_ask
        self.max_avg_bid = {}
        self.min_avg_ask = {}
        self.avg_bids_coor = {} #theoratical price as origin
        self.avg_asks_coor = {} #theoratical price as origin
        self.avg_bid_ct = {}
        self.avg_ask_ct = {}
        self.start_trade_tf = {}
        self.bid_weight = 1
        self.ask_weight = 1
        self.bid_step = 0
        self.ask_step = 0

        for product in self.PROD_LIST: # initialize variables
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

            self.max_avg_bid_ask_spread[product] = 0
            self.max_avg_bid[product] = 0
            self.min_avg_ask[product] = 0
            self.avg_bids_coor[product] = {}
            self.avg_asks_coor[product] = {}
            self.avg_bid_ct[product] = 0
            self.avg_ask_ct[product] = 0
            self.start_trade_tf[product] = False

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        #try:
            #-----Data update start-----
            for product in self.PROD_LIST:
                self.result[product] = []
                self.position[product] = 0 
            self.order_depths = state.order_depths
            self.timestamp = state.timestamp

            self.__update_postion(state)
            for product in self.PROD_LIST:
                print(f"product: {product}")
                self.__update_vol_and_price_weighted_by_vol(state, product) # update price of [product]
                self.__update_mid_price(product)
            #-----Data update end
                
            #-----Algo start-----
            #place order
            for product in self.PROD_LIST:
                if self.start_trade_tf[product]:
                    bid_prices = state.order_depths[product].buy_orders.keys().sort(reverse = True) # descending order
                    ask_prices = state.order_depths[product].sell_orders.keys().sort() # ascending order
                    avg_bids_coor = np.fromiter(self.avg_bids_coor[product].keys(), dtype = float)
                    avg_asks_coor = np.fromiter(self.avg_asks_coor[product].keys(), dtype = float)
                    avg_bids_coor_ct = np.fromiter(self.avg_bids_coor[product].values(), dtype = float)
                    avg_asks_coor_ct = np.fromiter(self.avg_asks_coor[product].values(), dtype = float)
                    remain_bid_size = self.get_max_bid_size(product)
                    remain_ask_size = self.get_max_ask_size(product)
                    ask_prob = 0
                    for ask_price in ask_prices: # bid
                        if remain_bid_size <= 0:
                            break
                        bid_prob = 0
                        vol_avail = state.order_depths[product].sell_orders[ask_price]
                        adj_avg_bids_coor = avg_bids_coor - ask_price
                        numer = np.sum(adj_avg_bids_coor * avg_bids_coor_ct)
                        denom = self.avg_bid_ct[product]
                        exp_val = numer/denom if denom > 0 else 0
                        if exp_val > 0 and self.max_avg_bid_ask_spread[product] > 0:
                            bid_prob = exp_val / self.max_avg_bid_ask_spread[product]
                            bid_prob = 1 if bid_prob > 1 else bid_prob
                            vol_desired = round(remain_bid_size * bid_prob)
                            bid_vol = min(vol_desired, vol_avail)
                            remain_bid_size -= bid_vol
                            self.place_order(product, ask_price, bid_vol)
                        
                        
                        avg_bids_above_bAsk = avg_bids_coor[avg_bids_coor > self.best_ask_price[product]]
                        avg_asks_below_bBid = avg_asks_coor[avg_asks_coor < self.best_bid_price[product]]


            #update dataset
            #treat theo_p as origin so everything below is negative, above is positive (like coordinatees)
            for product in self.PROD_LIST:
                theo_p = self.get_theo_price(product)
                avg_bid_coor = self.custom_rd(self.avg_bid_price[product] - theo_p)
                avg_ask_coor = self.custom_rd(self.avg_ask_price[product] - theo_p)

                self.max_avg_bid[product] = self.avg_bid_price[product] if self.avg_bid_price[product] > self.max_avg_bid[product] else self.max_avg_bid[product]
                self.min_avg_ask[product] = self.avg_ask_price[product] if self.avg_ask_price[product] < self.min_avg_ask[product] else self.min_avg_ask[product]
                self.max_avg_bid_ask_spread[product] = self.max_avg_bid[product] - self.min_avg_ask[product]

                if avg_bid_coor in self.avg_bids_coor[product].keys():
                    self.avg_bids_coor[product][avg_bid_coor] += self.bid_weight
                else:
                    self.avg_bids_coor[product][avg_bid_coor] = self.bid_weight
                if avg_ask_coor in self.avg_asks_coor[product].keys():
                    self.avg_asks_coor[product][avg_ask_coor] += self.ask_weight
                else:
                    self.avg_asks_coor[product][avg_ask_coor] = self.ask_weight
                self.avg_bid_ct[product] += self.bid_weight
                self.avg_ask_ct[product] += self.ask_weight
                self.bid_weight += self.bid_step
                self.ask_weight += self.ask_step

                if self.start_trade_tf[product] == False and self.max_avg_bid_ask_spread[product] > 0:
                    self.start_trade_tf[product] = True
            #-----Algo end
            return self.result
        #except Exception:
        #    self.result = {}
        #    return self.result

    #-----Algo methods start-----
    def custom_rd(self, num):
        return round(num*2,1)/2
    def get_theo_price(self, product):
        if product == "PEARLS":
            return self.PEARL_PRICE
        elif product == "BANANAS":
            return 0

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
        return st.stdev(self.hist_vol[product]) if len(self.hist_vol[product]) > 1 else -1
    
    def get_price_mean(self, product):
        return st.mean(self.hist_prices[product]) if len(self.hist_prices[product]) > 0 else -1
    
    def get_price_std(self, product):
        return st.stdev(self.hist_prices[product]) if len(self.hist_prices[product]) > 1 else -1
    
    def get_mid_price_mean(self, product):
        return st.mean(self.hist_mid_prices[product]) if len(self.hist_mid_prices[product]) > 0 else -1
    
    def get_mid_price_std(self, product):
        return st.stdev(self.hist_mid_prices[product]) if len(self.hist_mid_prices[product]) > 1 else -1
    
    def get_best_bid_mean(self, product):
        return st.mean(self.hist_best_bid_prices[product]) if len(self.hist_best_bid_prices[product]) > 0 else -1
    
    def get_best_bid_std(self, product):
        return st.stdev(self.hist_best_bid_prices[product]) if len(self.hist_best_bid_prices[product]) > 1 else -1
    
    def get_best_ask_mean(self, product):
        return st.mean(self.hist_best_ask_prices[product]) if len(self.hist_best_ask_prices[product]) > 0 else -1
    
    def get_best_ask_std(self, product):
        return st.stdev(self.hist_best_ask_prices[product]) if len(self.hist_best_ask_prices[product]) > 1 else -1
    
    def get_avg_bid_mean(self, product):
        return st.mean(self.hist_avg_bid_prices[product]) if len(self.hist_avg_bid_prices[product]) > 0 else -1
    
    def get_avg_bid_std(self, product):
        return st.stdev(self.hist_avg_bid_prices[product]) if len(self.hist_avg_bid_prices[product]) > 1 else -1
    
    def get_avg_ask_mean(self, product):
        return st.mean(self.hist_avg_ask_prices[product]) if len(self.hist_avg_ask_prices[product]) > 0 else -1
    
    def get_avg_ask_std(self, product):
        return st.stdev(self.hist_avg_ask_prices[product]) if len(self.hist_avg_ask_prices[product]) > 1 else -1
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
                if trade.timestamp == self.timestamp:
                    sum_price += trade.price*abs(trade.quantity)
                    t_vol += abs(trade.quantity)
        if product in state.market_trades.keys():
            product_market_trades = state.market_trades[product]
            for trade in product_market_trades:
                if trade.timestamp == self.timestamp:
                    sum_price += trade.price*abs(trade.quantity)
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
                bid_sum += price*abs(vol)
                bid_ct += abs(vol)
            for price, vol in product_asks.items():
                ask_sum += price*abs(vol)
                ask_ct += abs(vol)  
            if max_bid > 0 and min_ask > 0:
                self.mid_price[product] = (max_bid + min_ask) / 2
            elif max_bid > 0 or min_ask > 0:
                self.mid_price[product] = (max_bid + min_ask)
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
    #-----Helper methods end