import pickle

def main():
    with open("./data/round-1-pkl/PEARLS_trades_day0nn.pkl", "rb") as f:
        data = pickle.load(f)
    print(data[0])


    POS_LIMIT= {"PEARLS": 20, "BANANAS": 20} # Dict: {product_name -> pos_limit} NOTE: need to manually update dict of product limits
    PROD_LIST = ("PEARLS", "BANANAS") # Set: product_names NOTE: need to manually update list of products
    MAX_LOT_SIZE = 10
    PEARLS_PRICE = 10000
    price_coor = {}
    for trades in data:
        for price,ct in trades.items():
            w_price = 0
            coor = price - PEARLS_PRICE
            if coor in price_coor.keys():
                price_coor[coor] += ct
            else:
                price_coor[coor] = ct
    print(price_coor)

    """for product in self.PROD_LIST:
                theo_p = 10000
                if theo_p != 0 and self.price[product] != 0:
                    avg_bid_coor = self.custom_rd(self.price[product] - theo_p)
                    avg_ask_coor = self.custom_rd(self.price[product] - theo_p)

                    self.max_avg_bid[product] = self.price[product] if self.price[product] > self.max_avg_bid[product] else self.max_avg_bid[product]
                    self.min_avg_ask[product] = self.price[product] if self.price[product] < self.min_avg_ask[product] else self.min_avg_ask[product]
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
                    if product == "PEARLS":
                        print(self.avg_bids_coor[product])
                        print(self.avg_asks_coor[product])"""
if __name__ == "__main__":
    main()