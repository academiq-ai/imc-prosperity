from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order
import statistics as st


"""NOTE: if volume > 0 -> bid; if volume < 0 -> ask"""

class Trader:
    POS_LIMIT: Dict[str, int] = {"PEARLS": 20, "BANANAS": 20} # Dict: {product_name -> pos_limit} NOTE: need to manually update dict of product limits
    PROD_LIST = ("PEARLS", "BANANAS") # Set: product_names NOTE: need to manually update list of products
    MAX_LOT_SIZE = 40

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

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        # try:
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


            # -----Algo start-----



            for prod in self.PROD_LIST:
                theo_price = self.price[prod]

                if prod == "PEARLS":

                    # Market Making Plays
                        #   When theoretical price is in the spread, quoting the average of theo and best bid/ask, quoting only above and below theoretical
                        if self.get_best_bid(prod) < theo_price and theo_price < self.get_best_ask(prod):
                            print(2)
                            self.place_order(prod, (0.7*self.get_best_ask(prod) + 0.3*theo_price), -self.get_max_ask_size(prod)) # best pearls is 0.7,0.3
                            self.place_order(prod, (0.7*self.get_best_bid(prod) + 0.3*theo_price), self.get_max_bid_size(prod))

                        # Buying and selling whenever the spread is crossed
                        if self.get_best_ask(prod) < self.get_best_bid(prod):
                            print(3)
                            self.place_order(prod, self.get_best_ask(prod), -self.get_max_ask_size(prod))
                            self.place_order(prod, self.get_best_ask(prod), self.get_max_bid_size(prod))

                if prod == "BANANAS":
                    # Theo_price is only mean for pearls because it is stationary
                    if len(self.hist_prices[prod]) > 0:
                        theo_price = self.price[prod]
                        print(0)

                    # Market Making Plays
                        #   When theoretical price is in the spread, quoting the average of theo and best bid/ask, quoting only above and below theoretical
                        if self.get_best_bid(prod) < theo_price and theo_price < self.get_best_ask(prod):
                            print(2)
                            self.place_order(prod, (0.5*self.get_best_ask(prod) + 0.5*theo_price), -self.get_max_ask_size(prod)) # best banana is 0.5, 0.5
                            self.place_order(prod, (0.5*self.get_best_bid(prod) + 0.5*theo_price), self.get_max_bid_size(prod))

                        # Buying and selling whenever the spread is crossed
                        if self.get_best_ask(prod) < self.get_best_bid(prod):
                            print(3)
                            self.place_order(prod, self.get_best_ask(prod), -self.get_max_ask_size(prod))
                            self.place_order(prod, self.get_best_ask(prod), self.get_max_bid_size(prod))


            #-----Algo end

            return self.result
        # except Exception:
        #     self.result = {}
        #     return self.result

    #-----Algo methods start-----




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
        return st.mean(self.hist_vol[product]) if len(self.hist_vol[product]) > 1 else -1

    def get_vol_std(self, product):
        return st.stdev(self.hist_vol[product]) if len(self.hist_vol[product]) > 1 else -1

    def get_price_mean(self, product):
        return st.mean(self.hist_prices[product]) if len(self.hist_prices[product]) > 1 else -1

    def get_price_std(self, product):
            return st.stdev(self.hist_prices[product]) if len(self.hist_prices[product]) > 1 else -1

    def get_mid_price_mean(self, product):
        return st.mean(self.hist_mid_prices[product]) if len(self.hist_mid_prices[product]) > 1 else -1

    def get_mid_price_std(self, product):
        return st.stdev(self.hist_mid_prices[product]) if len(self.hist_mid_prices[product]) > 1 else -1
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








# extra
    # Stationarity Test (Augmented Dickey Fuller Test)

    # Load time series data into a pandas DataFrame
    # data = pd.read_csv('time_series_data.csv', index_col='date', parse_dates=True)
    #
    # # Define a function to run the ADF test and print the results
    # def adf_test(series):
    #     # Calculate the first difference of the series
    #     diff_series = series.diff().dropna()
    #     # Calculate the ADF test statistic
    #     adf_stat = adf_test_stat(diff_series)
    #     print('ADF Statistic: {:.4f}'.format(adf_stat))
    #     # Calculate the p-value
    #     p_value = adf_test_pvalue(adf_stat, len(series))
    #     print('p-value: {:.4f}'.format(p_value))
    #     # Calculate the critical values
    #     crit_vals = adf_test_critvals(len(series))
    #     print('Critical Values:')
    #     for key, value in crit_vals.items():
    #         print('\t{}: {:.4f}'.format(key, value))
    #     if adf_stat < crit_vals['5%']:
    #         print('Reject the null hypothesis. The data is stationary.')
    #     else:
    #         print('Fail to reject the null hypothesis. The data is non-stationary.')

    # Define a function to calculate the ADF test statistic
    # def adf_test_stat(series):
    #     # Calculate the mean and standard deviation of the first differences
    #     diff_mean = st.mean(series)
    #     diff_std = st.stdev(series)
    #     # Calculate the test statistic
    #     t_stat = (diff_mean / diff_std) * math.sqrt(len(series))
    #     return t_stat
    #
    # # Define a function to calculate the p-value
    # def adf_test_pvalue(t_stat, n_obs):
    #     # Calculate the degrees of freedom
    #     df = n_obs - 2
    #     # Calculate the p-value using a two-tailed t-test
    #     p_value = 1 - stats.t.cdf(abs(t_stat), df)
    #     return p_value
    #
    # # Define a function to calculate the critical values
    # def adf_test_critvals(n_obs):
    #     # Define the critical values for various significance levels
    #     crit_vals = {'1%': -3.430,
    #                  '5%': -2.861,
    #                  '10%': -2.566}
    #     # Adjust the critical values for small sample sizes
    #     adj_factor = (1.0 / n_obs) + 0.25
    #     for key, value in crit_vals.items():
    #         crit_vals[key] = crit_vals[key] - adj_factor
    #     return crit_vals
    #
    # # Call the adf_test function on the time series data
    # adf_test(data['value'])

