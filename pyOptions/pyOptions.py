import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from math import pi
import datetime as dt
import time

class Black_Scholes(object):

    def __init__(self, days_in_year=252):
        self.days_in_year = days_in_year

    def pricing(self, spot, strike, dmat, rate, vol, typ, div=None):
        #Vectorized

        if div is None:
            div=0.0

        d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol * vol) * (dmat / self.days_in_year)) / (vol * np.sqrt(dmat / self.days_in_year))
        d2 = d1 - vol * np.sqrt(dmat / self.days_in_year)
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        nd1n = norm.cdf(-d1)
        nd2n = norm.cdf(-d2)

        priceBS = np.where(typ == "C",
                           # Call
                           spot * np.exp(-div * (dmat / self.days_in_year)) * nd1 - strike * np.exp(-rate * (dmat / self.days_in_year)) * nd2,
                           # Put
                           strike * np.exp(-rate * (dmat / self.days_in_year)) * nd2n - spot * np.exp(-div * (dmat / self.days_in_year))* nd1n)

        # If option expired
        priceBS = np.where(dmat <= 0, payoff(spot, strike, typ), priceBS)

        return priceBS

    def greeks(self, spot, strike, dmat, rate, vol, typ, div=None, greek = 'all'):
        #Vectorized

        if div is None:
            div=0.0

        d1 = (np.log(spot / strike) + (rate - div + 0.5 * vol * vol) * (dmat / self.days_in_year)) / (vol * np.sqrt(dmat / self.days_in_year))
        d2 = d1 - vol * np.sqrt(dmat / self.days_in_year)
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        nd1n = norm.cdf(-d1)
        nd2n = norm.cdf(-d2)
        div_term = np.exp(-rate * (dmat / self.days_in_year))
        n_dash_d1 = div_term * np.exp(-d1 * d1 / 2) / (np.sqrt(2 * pi))


        if greek == 'delta':
            # for a 1 move of underlying
            return np.where(typ == "C", nd1 * div_term, div_term * (nd1 - 1))
        elif greek == 'gamma':
            # for a 1 move amplitude of underlying
            return n_dash_d1 / (spot * vol * np.sqrt(dmat / self.days_in_year))
        elif greek == 'theta':
            # for 1 business day
            return np.where(typ == "C",
                            (1 / self.days_in_year) * (-(spot * n_dash_d1 * vol) / (2 * np.sqrt(dmat / self.days_in_year)) - (rate * strike * np.exp(-rate * (dmat / self.days_in_year)) * nd2) + div * spot * div_term * nd1),
                            (1 / self.days_in_year) * (-(spot * n_dash_d1 * vol) / (2 * np.sqrt(dmat / self.days_in_year)) + (rate * strike * np.exp(-rate * (dmat / self.days_in_year)) * nd2n) - div * spot * div_term * nd1n))
        elif greek == 'vega':
            # for a 1% move of iv
            return spot * np.sqrt(dmat / self.days_in_year) * n_dash_d1 / 100
        else:
            delta = np.where(typ == "C", nd1 * div_term, div_term * (nd1 - 1))
            gamma = n_dash_d1 / (spot * vol * np.sqrt(dmat / self.days_in_year))
            theta = np.where(typ == "C",
                            (1 / self.days_in_year) * (-(spot * n_dash_d1 * vol) / (2 * np.sqrt(dmat / self.days_in_year)) - (rate * strike * np.exp(-rate * (dmat / self.days_in_year)) * nd2) + div * spot * div_term * nd1),
                            (1 / self.days_in_year) * (-(spot * n_dash_d1 * vol) / (2 * np.sqrt(dmat / self.days_in_year)) + (rate * strike * np.exp(-rate * (dmat / self.days_in_year)) * nd2n) - div * spot * div_term * nd1n))
            vega = spot * np.sqrt(dmat / self.days_in_year) * n_dash_d1 / 100
            return [delta, gamma, theta, vega]

    def implied_vol(self, spot, strike, dmat, rate, typ, price, div=None):
        #Vectorized

        if div is None:
            div=0.0

        if isinstance(price, pd.Series)  or isinstance(price, np.ndarray):
            return fsolve(func=lambda x: self.pricing(spot, strike, dmat, rate, x, typ, div) - price, x0=np.array([0.2 for d in price]))
        else:
            return fsolve(func=lambda x: self.pricing(spot, strike, dmat, rate, x, typ, div) - price, x0=np.array(0.2))[0]

class Monte_Carlo(object):

    def __init__(self, days_in_year=252):
        self.days_in_year = days_in_year

    def pricing(self, spot, strike, dmat, rate, vol, div=None, iterations=100000, time_steps=100, antithetic_variates=False, moment_matching=False):

        if div is None:
            div=0.0

        #time interval
        interval = (dmat / self.days_in_year) / time_steps

        S = np.zeros((time_steps+1, iterations))
        S[0] = spot
        for step in range(1, time_steps+1):
            gauss = self.__gauss_generator(iterations, antithetic_variates, moment_matching)
            S[step] = S[step-1]*np.exp((rate-div-0.5*vol*vol)*interval + vol*np.sqrt(interval)*gauss)

        call_price = np.exp(-rate * (dmat / self.days_in_year)) * np.sum(np.maximum(S[-1] - strike, 0))/iterations
        put_price = np.exp(-rate * (dmat / self.days_in_year)) * np.sum(np.maximum(strike - S[-1], 0))/iterations
        return {'call': call_price, 'put' : put_price}

    @staticmethod
    def __gauss_generator(d, antithetic_variates, moment_matching):
        if antithetic_variates:
            gauss = np.random.randn(int(d/2))
            gauss = np.concatenate((gauss, -gauss))
        else:
            gauss = np.random.randn(d)

        if moment_matching:
            gauss = gauss-np.mean(gauss)
            gauss = gauss / np.std(gauss)
        return gauss

class Binomial_Tree(object):
    def __init__(self, days_in_year=252):
        self.days_in_year = days_in_year

    def pricing(self, spot, strike, dmat, rate, vol, typ, div=None, american=False, time_steps=2000):

        interval = (dmat/self.days_in_year)/time_steps

        u = np.exp(vol * np.sqrt(interval))
        d = 1.0 / u
        a = np.exp((rate-div) * interval)
        p_up = (a-d) / (u-d)
        p_down = 1.0 - p_up

        vector_prices = np.zeros(time_steps+1)

        vector_spots = np.array([spot * u ** j * d ** (time_steps - j) for j  in range(time_steps+1)])
        vector_strikes = np.array([strike for j in range(time_steps+1)])

        if typ == 'C':
            vector_prices = np.maximum(vector_spots - vector_strikes, 0.0)
        else:
            vector_prices = np.maximum(vector_strikes - vector_spots, 0.0)

        for i in range(time_steps - 1, -1, -1):
            vector_prices[:-1] = np.exp(-rate * interval) * (p_up * vector_prices[1:] + p_down * vector_prices[:-1])

            if american:
                vector_spots = vector_spots * u
                if typ == 'C':
                    vector_prices = np.maximum(vector_prices, vector_spots - vector_strikes)
                else:
                    vector_prices = np.maximum(vector_prices, vector_strikes - vector_spots)

        return vector_prices[0]

def payoff(spotMat, strike, typ):
    #Vectorized

    #Returns options payoff

    return np.where(typ == "C", np.maximum(spotMat - strike, 0), np.maximum(strike - spotMat, 0))

def days_to_maturity(endDate, hours_in_day=6.5, hour_close=16, minute_close=0):
    #Vectorizes

    #Return exact number of days when market is open (useful fro calculating number of days till expiration)

    # Parameters
    seconds_in_hour = 3600
    startDate = dt.datetime.today()

    # Holidays
    # ref : https://www.nyse.com/markets/hours-calendars
    closedList = ["01/02/2017", "01/16/2017", "02/20/2017", "04/14/2017", "05/29/2017", "07/04/2017", "09/04/2017",
                  "11/23/2017", "12/25/2017", "01/01/2018", "01/15/2018", "02/19/2018", "03/30/2018", "05/28/2018",
                  "07/04/2018", "09/03/2018", "11/22/2018", "12/25/2018", "01/01/2019", "01/21/2019", "02/18/2019",
                  "04/19/2019", "05/27/2019", "07/04/2019", "09/02/2019", "11/28/2019", "12/25/2019", "01/01/2020",
                  "01/20/2020","02/17/2020", "04/10/2020", "05/25/2020", "07/03/2020", "09/07/2020", "11/26/2020",
                  "12/25/2020"]
    holidays_list = [dt.datetime.strptime(date, "%m/%d/%Y").date() for date in closedList]

    # NbDays
    NbDays = np.busday_count(startDate, endDate, holidays=holidays_list)
    NbDays = np.maximum(NbDays, 0)

    # NbSeconds
    end_of_day = dt.datetime(startDate.year, startDate.month, startDate.day, hour_close, minute_close, 0)
    NbSeconds = np.maximum(np.minimum((end_of_day - startDate).seconds / (seconds_in_hour * hours_in_day), hours_in_day), 0)

    return NbDays + NbSeconds

def random_walk_generator(mu=0.05, sigma=0.2, S0=100, T=1):
    #Generate a random walk to perform analysis

    deltaT = 1 / 252
    N = round(T/deltaT)
    step = np.exp((mu - 0.5 * sigma**2) * deltaT + sigma * np.sqrt(deltaT) * np.random.randn(N))
    S = S0 * step.cumprod()
    return pd.Series(S)

def statistics_backtest(daily_pnls):
    #Generate useful backtest statistics on pnl series

    daily_pnls.dropna(inplace=True)
    total_pnl = np.sum(daily_pnls)
    sum_cummuled = daily_pnls.cumsum()
    max_drawdown = np.min(sum_cummuled - np.maximum.accumulate(sum_cummuled))
    max_drawdown_end = np.argmin(sum_cummuled - np.maximum.accumulate(sum_cummuled))
    max_drawdown_begin = np.argmax(sum_cummuled[:max_drawdown_end])
    max_pnl = np.max(daily_pnls)
    avg_pnl = np.mean(daily_pnls)
    mdn_pnl = np.median(daily_pnls)
    std_pnl = np.std(daily_pnls)
    min_pnl = np.min(daily_pnls)
    proba_up = len(daily_pnls[daily_pnls >= 0]) / len(daily_pnls)
    sharpe = avg_pnl * np.sqrt(252) / std_pnl
    sortino = avg_pnl * np.sqrt(252) / np.std(daily_pnls[daily_pnls < 0])

    return {'total_pnl': total_pnl,
            'max_drawdown': max_drawdown,
            'max_drawdown_end': max_drawdown_end,
            'max_drawdown_begin': max_drawdown_begin,
            'max_pnl': max_pnl,
            'avg_pnl': avg_pnl,
            'mdn_pnl': mdn_pnl,
            'std_pnl': std_pnl,
            'min_pnl': min_pnl,
            'proba_up': proba_up,
            'sharpe': sharpe,
            'sortino': sortino}

def main():

    spot = 200
    strike = 220
    dmat = 2*252
    vol = 25/100
    rate = 5/100
    div = 2/100
    typ = 'C'

    spot_list = pd.Series(np.array([spot for d in range(3)]))
    strike_list = pd.Series(np.array([strike for d in range(3)]))
    dmat_list = pd.Series(np.array([dmat for d in range(3)]))
    vol_list = pd.Series(np.array([vol for d in range(3)]))
    rate_list = pd.Series(np.array([rate for d in range(3)]))
    div_list = pd.Series(np.array([div for d in range(3)]))
    typ_list = pd.Series(np.array([typ for d in range(3)]))

    price_list = pd.Series(np.array([24.13 for d in range(3)]))

    BS = Black_Scholes()
    call_price1 = BS.pricing(spot,strike, dmat, rate, vol, typ, div)
    print("Call value : {price:.3f}".format(price=float(call_price1)))
    call_greeks = BS.greeks(spot, strike, dmat, rate, vol, typ, div)
    print("Call delta : {price:.3f}".format(price=float(call_greeks[0])))
    print("Call gamma : {price:.3f}".format(price=float(call_greeks[1])))
    print("Call theta : {price:.3f}".format(price=float(call_greeks[2])))
    print("Call vega : {price:.3f} \n".format(price=float(call_greeks[3])))

    put_price1 = BS.pricing(spot, strike, dmat, rate, vol, 'P', div)
    print("Put value : {price:.3f}".format(price=float(put_price1)))
    put_greeks = BS.greeks(spot, strike, dmat, rate, vol, 'P', div)
    print("Put delta : {price:.3f}".format(price=float(put_greeks[0])))
    print("Put gamma : {price:.3f}".format(price=float(put_greeks[1])))
    print("Put theta : {price:.3f}".format(price=float(put_greeks[2])))
    print("Put vega : {price:.3f} \n".format(price=float(put_greeks[3])))

    iv_vect = BS.implied_vol(spot_list, strike_list, dmat_list, rate_list, typ_list, price_list, div_list)
    iv_call = BS.implied_vol(spot,strike, dmat, rate, 'C', 24.13, div)
    print("Call iv : {price:.3f}".format(price=float(iv_call)))
    iv_put = BS.implied_vol(spot, strike, dmat, rate, 'P', 31.036, div)
    print("Put iv : {price:.3f} \n".format(price=float(iv_put)))

    maturities = [pd.to_datetime('1/1/2019'), pd.to_datetime('9/1/2018'), pd.to_datetime('10/1/2018'), pd.to_datetime('12/1/2018')]
    n_days = days_to_maturity(maturities)
    print("Days to maturity : {price:.3f} \n".format(price=float(n_days[0])))

    MC = Monte_Carlo()
    mc_price = MC.pricing(spot, strike, dmat, rate, vol, div, moment_matching=True, antithetic_variates=True)
    print("MC call price : {price:.3f}".format(price=mc_price['call']))
    print("MC put price : {price:.3f} \n".format(price=mc_price['put']))

    BT = Binomial_Tree()
    t0 = time.clock()
    bt_call_price = BT.pricing(spot, strike, dmat, rate, vol, 'C', div, american=True)
    t1 = time.clock()
    print("Binomial tree call price : {price:.3f} in {timing:.2f} ms.".format(price=bt_call_price, timing=(t1-t0)*1000))

    bt_put_price = BT.pricing(spot, strike, dmat, rate, vol, 'P', div)
    print("Binomial tree put price : {price:.3f}".format(price=bt_put_price))


if __name__ == "__main__":
    main()