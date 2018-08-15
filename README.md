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

More examples [here](examples/examples.ipynb)







