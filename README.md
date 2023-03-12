

# IMC Prosperity

## Installation
### Anaconda (See Documentation [here](https://docs.anaconda.com/anaconda/install/)):

```bash
conda create --name <env> --file requirements_CONDA.txt
```

### VENV Virtual Environment (See Documentation [here](https://docs.python.org/3/library/venv.html))
```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements_PIP.txt
```

File Directory
-----
```python
.
├── README.md
├── Trader.py # Our Algorithm (example-program.py for now)
├── algo-main.py # main script (Godric's Algorithm atm)
├── backtest-main.py # script to run backtesting.py
├── backtesting # local-fitted backtest.py library
│   ├── __init__.py
│   ├── _plotting.py
│   ├── _stats.py
│   ├── _util.py
│   ├── _version.py
│   ├── autoscale_cb.js
│   ├── backtesting.py
│   ├── lib.py
│   └── test # backtest.py test files
│       ├── EURUSD.csv
│       ├── GOOG.csv
│       ├── __init__.py
│       ├── __main__.py
│       ├── __pycache__
│       │   └── __init__.cpython-39.pyc
│       └── _test.py
├── data # log/csv downloaded from IMC
│   ├── market_tutorial_0.csv
│   └── sandboxLogs.log
├── datamodel.py # basic collections objects from IMC
├── doc
├── pyproject.toml
├── requirements_CONDA.txt # requirements for CONDA (see installation)
├── requirements_PIP.txt # requirements for VENV (see installation)
├── setup.cfg
└── setup.py
```


Usage
-----
```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


bt = Backtest(GOOG, SmaCross, commission=.002,
              exclusive_orders=True)
stats = bt.run()
bt.plot()
```

Results in:

```text
Start                     2004-08-19 00:00:00
End                       2013-03-01 00:00:00
Duration                   3116 days 00:00:00
Exposure Time [%]                       94.27
Equity Final [$]                     68935.12
Equity Peak [$]                      68991.22
Return [%]                             589.35
Buy & Hold Return [%]                  703.46
Return (Ann.) [%]                       25.42
Volatility (Ann.) [%]                   38.43
Sharpe Ratio                             0.66
Sortino Ratio                            1.30
Calmar Ratio                             0.77
Max. Drawdown [%]                      -33.08
Avg. Drawdown [%]                       -5.58
Max. Drawdown Duration      688 days 00:00:00
Avg. Drawdown Duration       41 days 00:00:00
# Trades                                   93
Win Rate [%]                            53.76
Best Trade [%]                          57.12
Worst Trade [%]                        -16.63
Avg. Trade [%]                           1.96
Max. Trade Duration         121 days 00:00:00
Avg. Trade Duration          32 days 00:00:00
Profit Factor                            2.13
Expectancy [%]                           6.91
SQN                                      1.78
Kelly Criterion                        0.6134
_strategy              SmaCross(n1=10, n2=20)
_equity_curve                          Equ...
_trades                       Size  EntryB...
dtype: object
```
[![plot of trading simulation](https://i.imgur.com/xRFNHfg.png)](https://kernc.github.io/backtesting.py/#example)

