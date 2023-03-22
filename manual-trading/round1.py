import numpy as np


def main():

    # products = {
    #     pizza: 0,
    #     wasabi: 1,
    #     snowball: 2,
    #     seashell: 3,
    # }

    # combination: seashell -> pizza -> wasabi -> snowball -> pizza -> seashell
    # combination: 3, 0, 1, 2, 0, 3

    profits = []
    for p2 in [0, 1, 2]:
        for p3 in [0, 1, 2]:
            for p4 in [0, 1, 2]:
                combination = [3, p2, p3, p4, 3]
                profit = calcProfit(combination)
                profits.append({"strategy": combination, "profit": profit})

    result = max(
        [p for p in profits if p["profit"] is not None], key=lambda x: x["profit"]
    )
    print(result)


def calcProfit(combination):
    balance = 2000000

    # from pizza to snowball = tradeTable[0, 2]
    tradeTable = np.array(
        [
            [1, 0.5, 1.45, 0.75],
            [1.95, 0.5, 3.1, 1.49],
            [0.67, 0.31, 1, 0.48],
            [1.34, 0.64, 1.98, 1],
        ]
    )

    for i in range(len(combination) - 1):
        balance *= tradeTable[combination[i + 1], combination[i]]
    return balance


if __name__ == "__main__":
    main()
