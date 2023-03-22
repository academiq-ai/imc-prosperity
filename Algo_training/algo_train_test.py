import pickle

def main():
    with open("../data/round-1-pkl/BANANAS_prices_day0.pkl", "rb") as f:
        data = pickle.load(f)
    """print(data[0])
    print(data[1])
    print(data[2])
    print(data[4])"""
    for i in range(10):
        print(data[i])
    """ct = 0
    for key,val in data.items():
        print(key, val)
        ct += 1
        if ct > 5:
            break"""


if __name__ == "__main__":
    main()