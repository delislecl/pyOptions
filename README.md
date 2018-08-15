# pyOptions

Options pricing library including different methods such as Black Scholes, Binomial trees or Monte Carlo simulations.
All methods are vectorized and can be directly applied on numpy arrays or pandas series.

## Requirements

- Pandas
- Numpy
- Scipy

## Installation

Can be installed using pip :

```
pip install git+https://github.com/delislecl/pyOptions.git
```

## Examples

We can define options parameters as follow :

```
S, K, T, sigma, r, typ, div = 200, 220, 2*252, 0.25, 0.05, 'C', 0.03
```

For european options using  Black & Scholes :

```
BS = pyOptions.Black_Scholes()

opt_price = BS.pricing(S, K, T, r, sigma, typ, div)

iv = BS.implied_vol(S, K, T, r, typ, opt_price, div)

delta, gamma, theta, vega = BS.greeks(S, K, T, r, sigma, typ, div)
```

For american options using Binomial Trees :

```
BT = pyOptions.Binomial_Tree()

opt_price = BT.pricing(S, K, T, r, sigma, typ, div, american=True, time_steps=1000)

iv = BT.implied_vol(S, K, T, r, typ, opt_price, div, american=True, time_steps=1000)

delta, gamma, theta, vega = BT.greeks(S, K, T, r, sigma, typ, div, american=True, time_steps=1000)
```

For european options using Monte Carlo simulations  :

```
MC = pyOptions.Monte_Carlo()

ITERATIONS = 100000
TIME_STEPS = 100

opt_price = MC.pricing(S, K, T, r, sigma, typ, div, iterations=ITERATIONS, time_steps=TIME_STEPS)

iv = MC.implied_vol(S, K, T, r, typ, opt_price, div, iterations=ITERATIONS, time_steps=TIME_STEPS)

delta, gamma, theta, vega = MC.greeks(S, K, T, r, sigma, typ, iterations=ITERATIONS, time_steps=TIME_STEPS)
```

using convergence optimization methods :

```
opt_price = MC.pricing(S, K, T, r, sigma, typ, div, iterations=ITERATIONS, time_steps=TIME_STEPS, antithetic_variates=True, moment_matching=True)
```

More examples [here](examples/examples.ipynb).

## Vectorization

Previous functions can also directly be applied to vectors to gain convenience and performance.

For example if we define the following vectors :

```
VECT_LENGTH = 1000

S_array = np.array([S for i in range(VECT_LENGTH)])
K_array = np.array([K for i in range(VECT_LENGTH)])
T_array = np.array([T for i in range(VECT_LENGTH)])
sigma_array = np.array([sigma for i in range(VECT_LENGTH)])
r_array = np.array([r for i in range(VECT_LENGTH)])
typ_array = np.array([typ for i in range(VECT_LENGTH)])
div_array = np.array([div for i in range(VECT_LENGTH)])
```

We can apply Black Scholes to get results as a vector :

```
BS = pyOptions.Black_Scholes()

opt_price = BS.pricing(S_array, K_array, T_array, r_array, sigma_array, typ_array, div_array)

iv = BS.implied_vol(S_array, K_array, T_array, r_array, typ_array, opt_price, div_array)

delta, gamma, theta, vega = BS.greeks(S_array, K_array, T_array, r_array, sigma_array, typ_array, div_array)
```

More examples with measured processsing times [here](examples/examples.ipynb).

## Others

Some other functions that can be usefull for options manipulations :

- days_to_maturity(endDate, hours_in_day=6.5, hour_close=16, minute_close=0) : will return the exact number of opening days of the NYSE remaining between now and endDate
(using calendar from [NYSE](https://www.nyse.com/markets/hours-calendars)) as a float (including current time).
Usefull for calculating options remaining days to expiration. For example :

```
expiration = dt.datetime.strptime('8/15/2020', '%m/%d/%Y')

days_remaining = pyOptions.days_to_maturity(expiration)
```
This will returns 505.50 days (ran 8/15/2018).
It can also be ran on vectors.

- payoff(spotMat, strike, typ) : simply return the payoff of an option.

- random_walk_generator(mu=0.05, sigma=0.2, S0=100, T=1) : simulate a stock path using a random walk with drift and volatility.

- statistics_backtest(daily_pnls) : generates statistics on a backtest result such as max_drawdown, sharp, sortino... It can be usefull for quickly analyzing a strategy attractiveness.
For example :

```
stock_simulated = pyOptions.random_walk_generator(mu=0.10, sigma=0.05, S0=100, T=2)

daily_pnl_simulated = 100000 * stock_simulated.pct_change().dropna()

stats = pyOptions.statistics_backtest(daily_pnl_simulated)
```
This will return :
```
{'avg_pnl': 33.474944500508855,
 'max_drawdown': -5407.4252657000725,
 'max_drawdown_begin': 221,
 'max_drawdown_end': 301,
 'max_pnl': 1031.0854184457207,
 'mdn_pnl': 29.515536772661832,
 'min_pnl': -761.08109214232388,
 'proba_up': 0.5427435387673957,
 'sharpe': 1.683991016930604,
 'sortino': 3.0739093013735892,
 'std_pnl': 315.55885064564615,
 'total_pnl': 16837.897083755954}
 ```







