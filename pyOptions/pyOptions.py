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

        if priceBS.size > 1:
            return priceBS
        else:
            return float(priceBS)

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

        if isinstance(price, pd.Series) or isinstance(price, np.ndarray):
            df = pd.DataFrame({'spot': spot, 'strike': strike, 'dmat': dmat, 'rate': rate, 'typ': typ, 'price': price, 'div': div})
            return df.apply(lambda x: self.__calculate_iv(x['spot'], x['strike'], x['dmat'], x['rate'], x['typ'], x['price'], x['div']), axis=1)
        else:
            return self.__calculate_iv(spot, strike, dmat, rate, typ, price, div)

    def implied_vol_old(self, spot, strike, dmat, rate, typ, price, div=None):
        # Vectorized

        if div is None:
            div = 0.0

        if isinstance(price, pd.Series) or isinstance(price, np.ndarray):
            return fsolve(func=lambda x: self.pricing(spot, strike, dmat, rate, x, typ, div) - price, x0=np.array([0.2 for d in spot]))
        else:
            return fsolve(func=lambda x: self.pricing(spot, strike, dmat, rate, x, typ, div) - price, x0=np.array(0.2))[0]


    def __calculate_iv(self, spot, strike, dmat, rate, typ, price, div):
            return fsolve(func=lambda x: self.pricing(spot, strike, dmat, rate, x, typ, div) - price, x0=np.array(0.2))[0]

class Monte_Carlo(object):

    def __init__(self, days_in_year=252):
        self.days_in_year = days_in_year

    def pricing(self, spot, strike, dmat, rate, vol, div=None, iterations=100000, time_steps=100, antithetic_variates=False, moment_matching=False):

        if div is None:
            div=0.0

        if isinstance(spot, pd.Series) or isinstance(spot, np.ndarray):
            df = pd.DataFrame({'spot': spot, 'strike' : strike, 'dmat': dmat, 'rate': rate, 'vol': vol, 'div': div})
            return df.apply(lambda x: self.__pricer(x['spot'], x['strike'], x['dmat'], x['rate'], x['vol'], x['div'], iterations, time_steps, antithetic_variates, moment_matching), axis=1)
        else:
            return self.__pricer(spot, strike, dmat, rate, vol, div, iterations, time_steps, antithetic_variates, moment_matching)


    def __pricer(self, spot, strike, dmat, rate, vol, div, iterations, time_steps, antithetic_variates, moment_matching):

        # time interval
        interval = (dmat / self.days_in_year) / time_steps

        S = np.zeros((time_steps + 1, iterations))
        S[0] = spot
        for step in range(1, time_steps + 1):
            gauss = self.__gauss_generator(iterations, antithetic_variates, moment_matching)
            actu = (rate - div - 0.5 * vol ** 2) * interval + gauss * vol * np.sqrt(interval)
            S[step] = S[step - 1] * np.exp(actu)

        actualization = np.exp(-rate * (dmat / self.days_in_year))
        call_price = actualization * np.sum(np.maximum(S[-1] - strike, 0)) / iterations
        put_price = actualization * np.sum(np.maximum(strike - S[-1], 0)) / iterations

        return [float(call_price), float(put_price)]

    @staticmethod
    def __gauss_generator(d1, antithetic_variates, moment_matching):
        if antithetic_variates:
            gauss = np.random.randn(int(d1 / 2))
            gauss = np.concatenate((gauss, -gauss))
        else:
            gauss = np.random.randn(d1)

        if moment_matching:
            gauss = gauss - np.mean(gauss)
            gauss = gauss / np.std(gauss)
        return gauss

    def pricing_old(self, spot, strike, dmat, rate, vol, div=None, iterations=100000, time_steps=100, antithetic_variates=False, moment_matching=False):

        if div is None:
            div=0.0

        dim = 1
        if isinstance(spot, pd.Series) or isinstance(spot, np.ndarray):
            dim = len(spot)

        spot = np.array(spot)
        strike = np.array(strike)
        dmat = np.array(dmat)
        rate = np.array(rate)
        vol = np.array(vol)
        div = np.array(div)

        #time interval
        interval = (dmat / self.days_in_year) / time_steps

        S = np.zeros((time_steps+1, iterations, dim))
        S[0] = spot
        for step in range(1, time_steps+1):
            gauss = self.__gauss_generator_old(iterations, dim, antithetic_variates, moment_matching)
            prev_step = S[step-1]
            brownian = gauss * np.array(vol * np.sqrt(interval))
            drift = np.array((rate-div-0.5* vol ** 2)*interval)
            actu = drift + brownian
            S[step] = prev_step * np.exp(actu)

        actualization = np.array(np.exp(-rate * (dmat / self.days_in_year)))
        call_price = actualization * np.sum(np.maximum(S[-1] - strike, 0), axis=0) / iterations
        put_price = actualization * np.sum(np.maximum(strike - S[-1], 0), axis=0)/iterations

        if dim == 1:
            return [float(call_price), float(put_price)]
        else:
            return [call_price, put_price]

    @staticmethod
    def __gauss_generator_old(d1, d2, antithetic_variates, moment_matching):
        if antithetic_variates:
            gauss = np.random.randn(int(d1/2), d2)
            gauss = np.concatenate((gauss, -gauss))
        else:
            gauss = np.random.randn(d1, d2)

        if moment_matching:
            gauss = gauss-np.mean(gauss)
            gauss = gauss / np.std(gauss)
        return gauss

class Binomial_Tree(object):
    def __init__(self, days_in_year=252):
        self.days_in_year = days_in_year

    def pricing(self, spot, strike, dmat, rate, vol, typ, div=None, american=False, time_steps=2000):
        #Vectorized

        if div is None:
            div = 0.0

        if isinstance(spot, pd.Series) or isinstance(spot, np.ndarray):
            df = pd.DataFrame({'spot': spot, 'strike': strike, 'dmat': dmat, 'rate': rate, 'vol': vol, 'typ': typ,  'div': div})
            return df.apply(lambda x: self.__pricer(x['spot'], x['strike'], x['dmat'], x['rate'], x['vol'], x['typ'], x['div'], american, time_steps), axis=1)
        else:
            return self.__pricer(spot, strike, dmat, rate, vol, typ, div, american, time_steps)

    def __pricer(self, spot, strike, dmat, rate, vol, typ, div, american, time_steps):
        interval = (dmat / self.days_in_year) / time_steps

        u = np.exp(vol * np.sqrt(interval))
        d = 1.0 / u
        a = np.exp((rate - div) * interval)
        p_up = (a - d) / (u - d)
        p_down = 1.0 - p_up

        vector_prices = np.zeros(time_steps + 1)

        vector_spots = np.array([spot * u ** j * d ** (time_steps - j) for j in range(time_steps + 1)])
        vector_strikes = np.array([strike for j in range(time_steps + 1)])

        if typ == 'C':
            vector_prices = np.maximum(vector_spots - vector_strikes, 0.0)
        else:
            vector_prices = np.maximum(vector_strikes - vector_spots, 0.0)

        for i in range(time_steps - 1, -1, -1):
            vector_prices[:-1] = (vector_prices[1:] * p_up + vector_prices[:-1] * p_down) * np.exp(-interval * rate)

            if american:
                vector_spots = vector_spots * u
                if typ == 'C':
                    vector_prices = np.maximum(vector_prices, vector_spots - vector_strikes)
                else:
                    np.maximum(vector_prices, vector_strikes - vector_spots)

        return vector_prices[0]

    def pricing_old(self, spot, strike, dmat, rate, vol, typ, div=None, american=False, time_steps=2000):
        #Vectorized

        if div is None:
            div = 0

        spot = np.array(spot)
        strike = np.array(strike)
        dmat = np.array(dmat)
        rate = np.array(rate)
        vol = np.array(vol)
        typ = np.array(typ)
        div = np.array(div)

        interval = (dmat/self.days_in_year)/time_steps

        u = np.array(np.exp(vol * np.sqrt(interval)))
        d = 1.0 / u
        a = np.exp((rate-div) * interval)
        p_up = np.array((a-d) / (u-d))
        p_down = 1.0 - p_up

        vector_prices = np.zeros(time_steps+1)

        vector_spots = np.array([spot * u ** j * d ** (time_steps - j) for j  in range(time_steps+1)])
        vector_strikes = np.array([strike for j in range(time_steps+1)])

        vector_prices = np.where(typ == 'C', np.maximum(vector_spots - vector_strikes, 0.0), np.maximum(vector_strikes - vector_spots, 0.0))

        for i in range(time_steps - 1, -1, -1):
            up = vector_prices[1:]
            down = vector_prices[:-1]
            exp_factor = np.exp( -interval * rate)
            vector_prices[:-1] = (up * p_up  + down * p_down) * exp_factor

            if american:
                vector_spots = vector_spots * u
                vector_prices = np.where(typ=='C', np.maximum(vector_prices, vector_spots - vector_strikes), np.maximum(vector_prices, vector_strikes - vector_spots))

        return vector_prices[0]

class Tester(object):

    def __init__(self, VECTOR_SIZES=100, spot=200, strike=220, dmat=2*252, vol = 0.25, rate=0.05, iv=0.02, typ='C', price=24.13, div=0):
        self.VECTOR_SIZES = VECTOR_SIZES
        self.spot_series = pd.Series([spot for d in range(VECTOR_SIZES)])
        self.strike_series = pd.Series([strike for d in range(VECTOR_SIZES)])
        self.dmat_series = pd.Series([dmat for d in range(VECTOR_SIZES)])
        self.vol_series = pd.Series([vol for d in range(VECTOR_SIZES)])
        self.rate_series = pd.Series([rate for d in range(VECTOR_SIZES)])
        self.div_series = pd.Series([div for d in range(VECTOR_SIZES)])
        self.typ_series = pd.Series([typ for d in range(VECTOR_SIZES)])
        self.price_series = pd.Series([price for d in range(VECTOR_SIZES)])

        self.spot_array = np.array([spot for d in range(VECTOR_SIZES)])
        self.strike_array = np.array([strike for d in range(VECTOR_SIZES)])
        self.dmat_array = np.array([dmat for d in range(VECTOR_SIZES)])
        self.vol_array = np.array([vol for d in range(VECTOR_SIZES)])
        self.rate_array = np.array([rate for d in range(VECTOR_SIZES)])
        self.div_array = np.array([div for d in range(VECTOR_SIZES)])
        self.typ_array = np.array([typ for d in range(VECTOR_SIZES)])
        self.price_array = np.array([price for d in range(VECTOR_SIZES)])

    def Black_Scholes(self):
        BS = Black_Scholes()

        #Price
        t0 = time.clock()
        call_price1 = BS.pricing(self.spot_series, self.strike_series, self.dmat_series, self.rate_series, self.vol_series, self.typ_series, self.div_series)
        t1 = time.clock()
        print("Price BS vectorized series : {price:.3f} in {timing:.2f} ms.".format(price=call_price1[0],timing=(t1 - t0) * 1000))

        t0 = time.clock()
        call_price1 = BS.pricing(self.spot_array, self.strike_array, self.dmat_array, self.rate_array, self.vol_array, self.typ_array, self.div_array)
        t1 = time.clock()
        print("Price BS vectorized array : {price:.3f} in {timing:.2f} ms.".format(price=call_price1[0],timing=(t1 - t0) * 1000))

        t0 = time.clock()
        for i in range(self.VECTOR_SIZES):
            call_price1 = BS.pricing(self.spot_series[i], self.strike_series[i], self.dmat_series[i], self.rate_series[i], self.vol_series[i], self.typ_series[i], self.div_series[i])
        t1 = time.clock()
        print("Price BS looping series : {price:.3f} in {timing:.2f} ms.".format(price=call_price1, timing=(t1 - t0) * 1000))

        t0 = time.clock()
        for i in np.ndindex(self.VECTOR_SIZES):
            call_price1 = BS.pricing(self.spot_array[i], self.strike_array[i], self.dmat_array[i], self.rate_array[i], self.vol_array[i], self.typ_array[i], self.div_array[i])
        t1 = time.clock()
        print("Price BS looping array : {price:.3f} in {timing:.2f} ms. \n".format(price=call_price1, timing=(t1 - t0) * 1000))

        #Greeks
        t0 = time.clock()
        call_greeks = BS.greeks(self.spot_series, self.strike_series, self.dmat_series, self.rate_series, self.vol_series, self.typ_series, self.div_series)
        t1 = time.clock()
        print("Greeks BS vectorized series : {timing:.2f} ms.".format(timing=(t1 - t0) * 1000))
        print("delta : {price:.3f}".format(price=float(call_greeks[0][0])))
        print("gamma : {price:.3f}".format(price=float(call_greeks[1][0])))
        print("theta : {price:.3f}".format(price=float(call_greeks[2][0])))
        print("vega : {price:.3f} \n".format(price=float(call_greeks[3][0])))

        t0 = time.clock()
        call_greeks = BS.greeks(self.spot_array, self.strike_array, self.dmat_array, self.rate_array, self.vol_array, self.typ_array, self.div_array)
        t1 = time.clock()
        print("Greeks BS vectorized array : {timing:.2f} ms.".format(timing=(t1 - t0) * 1000))
        print("delta : {price:.3f}".format(price=float(call_greeks[0][0])))
        print("gamma : {price:.3f}".format(price=float(call_greeks[1][0])))
        print("theta : {price:.3f}".format(price=float(call_greeks[2][0])))
        print("vega : {price:.3f} \n".format(price=float(call_greeks[3][0])))

        t0 = time.clock()
        for i in range(self.VECTOR_SIZES):
            call_greeks = BS.greeks(self.spot_series[i], self.strike_series[i], self.dmat_series[i], self.rate_series[i], self.vol_series[i],self.typ_series[i], self.div_series[i])
        t1 = time.clock()
        print("Greeks BS looping series : {timing:.3f} ms.".format(timing=(t1 - t0) * 1000))
        print("delta : {price:.3f}".format(price=float(call_greeks[0])))
        print("gamma : {price:.3f}".format(price=float(call_greeks[1])))
        print("theta : {price:.3f}".format(price=float(call_greeks[2])))
        print("vega : {price:.3f} \n".format(price=float(call_greeks[3])))

        t0 = time.clock()
        for i in np.ndindex(self.VECTOR_SIZES):
            call_greeks = BS.greeks(self.spot_array[i], self.strike_array[i], self.dmat_array[i],self.rate_array[i], self.vol_array[i], self.typ_array[i], self.div_array[i])
        t1 = time.clock()
        print("Greeks BS looping series : {timing:.3f} ms.".format(timing=(t1 - t0) * 1000))
        print("delta : {price:.3f}".format(price=float(call_greeks[0])))
        print("gamma : {price:.3f}".format(price=float(call_greeks[1])))
        print("theta : {price:.3f}".format(price=float(call_greeks[2])))
        print("vega : {price:.3f} \n".format(price=float(call_greeks[3])))


        #IV
        t0 = time.clock()
        iv_call = BS.implied_vol(self.spot_series, self.strike_series, self.dmat_series, self.rate_series, self.typ_series, self.price_series, self.div_series)
        t1 = time.clock()
        print("iv vectorized series : {price:.3f} in {timing:.2f} ms.".format(price=float(iv_call[0]),timing=(t1 - t0) * 1000))

        # IV
        t0 = time.clock()
        iv_call = BS.implied_vol(self.spot_array, self.strike_array, self.dmat_array, self.rate_array,self.typ_array, self.price_array, self.div_array)
        t1 = time.clock()
        print("iv vectorized array : {price:.3f} in {timing:.2f} ms.".format(price=float(iv_call[0]),timing=(t1 - t0) * 1000))

        t0 = time.clock()
        for i in range(self.VECTOR_SIZES):
            iv_call = BS.implied_vol(self.spot_series[i], self.strike_series[i], self.dmat_series[i], self.rate_series[i], self.typ_series[i], self.price_series[i], self.div_series[i])
        t1 = time.clock()
        print("iv looping series : {price:.3f} in {timing:.2f} ms.".format(price=float(iv_call), timing=(t1 - t0) * 1000))

        t0 = time.clock()
        for i in np.ndindex(self.VECTOR_SIZES):
            iv_call = BS.implied_vol(self.spot_array[i], self.strike_array[i], self.dmat_array[i], self.rate_array[i], self.typ_array[i], self.price_array[i], self.div_array[i])
        t1 = time.clock()
        print("iv looping array : {price:.3f} in {timing:.2f} ms.".format(price=float(iv_call), timing=(t1 - t0) * 1000))

        t0 = time.clock()
        iv_call = BS.implied_vol(self.spot_series, self.strike_series, self.dmat_series, self.rate_series, self.typ_series, self.price_series, self.div_series)
        t1 = time.clock()
        print("iv vectorized OLD series : {price:.3f} in {timing:.2f} ms.".format(price=float(iv_call[0]),timing=(t1 - t0) * 1000))

        t0 = time.clock()
        iv_call = BS.implied_vol(self.spot_array, self.strike_array, self.dmat_array, self.rate_array, self.typ_array, self.price_array, self.div_array)
        t1 = time.clock()
        print("iv vectorized OLD array : {price:.3f} in {timing:.2f} ms. \n".format(price=float(iv_call[0]),timing=(t1 - t0) * 1000))

    def Monte_Carlo(self):
        MC = Monte_Carlo()

        t0 = time.clock()
        mc_price = MC.pricing(self.spot_series, self.strike_series, self.dmat_series, self.rate_series, self.vol_series, self.div_series, antithetic_variates=True, moment_matching=True)
        t1 = time.clock()
        print("MC price vectorized series : {price:.3f} in {timing:.2f} ms.".format(price=mc_price[0][0],timing=(t1 - t0) * 1000))

        t0 = time.clock()
        mc_price = MC.pricing(self.spot_array, self.strike_array, self.dmat_array, self.rate_array, self.vol_array, self.div_array, antithetic_variates=True, moment_matching=True)
        t1 = time.clock()
        print("MC price vectorized array : {price:.3f} in {timing:.2f} ms.".format(price=mc_price[0][0],timing=(t1 - t0) * 1000))

        t0 = time.clock()
        for i in range(self.VECTOR_SIZES):
            mc_price = MC.pricing(self.spot_series[i], self.strike_series[i], self.dmat_series[i], self.rate_series[i], self.vol_series[i], self.div_series[i], antithetic_variates=True, moment_matching=True)
        t1 = time.clock()
        print("MC price looping series : {price:.3f} in {timing:.2f} ms.".format(price=mc_price[0], timing=(t1 - t0) * 1000))

        t0 = time.clock()
        for i in np.ndindex(self.VECTOR_SIZES):
            mc_price = MC.pricing(self.spot_array[i], self.strike_array[i], self.dmat_array[i], self.rate_array[i], self.vol_array[i], self.div_array[i], antithetic_variates=True,moment_matching=True)
        t1 = time.clock()
        print("MC price looping array : {price:.3f} in {timing:.2f} ms.".format(price=mc_price[0], timing=(t1 - t0) * 1000))

        t0 = time.clock()
        mc_price = MC.pricing_old(self.spot_series, self.strike_series, self.dmat_series, self.rate_series, self.vol_series,self.div_series, antithetic_variates=True, moment_matching=True)
        t1 = time.clock()
        print("MC price vectorized OLD series : {price:.3f} in {timing:.2f} ms.".format(price=mc_price[0][0],timing=(t1 - t0) * 1000))

        t0 = time.clock()
        mc_price = MC.pricing_old(self.spot_array, self.strike_array, self.dmat_array, self.rate_array,self.vol_array, self.div_array, antithetic_variates=True, moment_matching=True)
        t1 = time.clock()
        print("MC price vectorized OLD series : {price:.3f} in {timing:.2f} ms. \n".format(price=mc_price[0][0],timing=(t1 - t0) * 1000))

    def Binomial_Tree(self, time_steps=2000):

        BT = Binomial_Tree()
        t0 = time.clock()
        bt_call_price = BT.pricing(self.spot_series, self.strike_series, self.dmat_series, self.rate_series, self.vol_series, self.typ_series, self.div_series, american=True, time_steps=time_steps)
        t1 = time.clock()
        print("Binomial tree price vectorized series : {price:.3f} in {timing:.2f} ms.".format(price=bt_call_price[0],timing=(t1 - t0) * 1000))

        BT = Binomial_Tree()
        t0 = time.clock()
        bt_call_price = BT.pricing(self.spot_series, self.strike_series, self.dmat_series, self.rate_series,self.vol_series, self.typ_series, self.div_series, american=True, time_steps=time_steps)
        t1 = time.clock()
        print("Binomial tree price vectorized array : {price:.3f} in {timing:.2f} ms.".format(price=bt_call_price[0], timing=(t1 - t0) * 1000))

        t0 = time.clock()
        for i in range(self.VECTOR_SIZES):
            bt_call_price = BT.pricing(self.spot_series[i], self.strike_series[i], self.dmat_series[i], self.rate_series[i], self.vol_series[i], self.typ_series[i], self.div_series[i], american=True, time_steps=time_steps)
        t1 = time.clock()
        print("Binomial tree price looping series : {price:.3f} in {timing:.2f} ms.".format(price=bt_call_price, timing=(t1 - t0) * 1000))

        t0 = time.clock()
        for i in np.ndindex(self.VECTOR_SIZES):
            bt_call_price = BT.pricing(self.spot_array[i], self.strike_array[i], self.dmat_array[i], self.rate_array[i], self.vol_array[i], self.typ_array[i], self.div_array[i], american=True, time_steps=time_steps)
        t1 = time.clock()
        print("Binomial tree price looping array : {price:.3f} in {timing:.2f} ms.".format(price=bt_call_price, timing=(t1 - t0) * 1000))

        t0 = time.clock()
        bt_call_price = BT.pricing_old(self.spot_series, self.strike_series, self.dmat_series, self.rate_series, self.vol_series, self.typ_series, self.div_series, american=True, time_steps=time_steps)
        t1 = time.clock()
        print("Binomial tree price vectorized OLD series : {price:.3f} in {timing:.2f} ms.".format(price=bt_call_price[0], timing=(t1 - t0) * 1000))

        t0 = time.clock()
        bt_call_price = BT.pricing_old(self.spot_array, self.strike_array, self.dmat_array, self.rate_array,self.vol_array, self.typ_array, self.div_array, american=True, time_steps=time_steps)
        t1 = time.clock()
        print("Binomial tree price vectorized OLD array : {price:.3f} in {timing:.2f} ms. \n".format(price=bt_call_price[0],timing=(t1 - t0) * 1000))

def payoff(spotMat, strike, typ):
    #Vectorized

    #Returns options payoff

    return np.where(typ == "C", np.maximum(spotMat - strike, 0), np.maximum(strike - spotMat, 0))

def days_to_maturity(endDate, hours_in_day=6.5, hour_close=16, minute_close=0):
    #Vectorizes

    #Return exact number of days when market is open (useful fro calculating number of days till expiration)
    if isinstance(endDate, pd.Series):
        endDate = endDate.dt.date

    endDate = np.array(endDate, dtype='datetime64')

    # Parameters
    seconds_in_hour = 3600
    startDatetime = dt.datetime.today()
    startDate = np.array([startDatetime] , dtype='datetime64')


    # Holidays
    # ref : https://www.nyse.com/markets/hours-calendars
    closedList = ["07/04/2018", "09/03/2018", "11/22/2018", "12/25/2018", "01/01/2019", "01/21/2019", "02/18/2019",
                  "04/19/2019", "05/27/2019", "07/04/2019", "09/02/2019", "11/28/2019", "12/25/2019", "01/01/2020",
                  "01/20/2020","02/17/2020", "04/10/2020", "05/25/2020", "07/03/2020", "09/07/2020", "11/26/2020",
                  "12/25/2020"]
    holidays_list = [dt.datetime.strptime(date, "%m/%d/%Y").date() for date in closedList]

    # NbDays
    NbDays = np.busday_count(startDate, endDate, holidays=holidays_list)
    NbDays = np.maximum(NbDays, 0)

    # NbSeconds
    end_of_day = dt.datetime(startDatetime.year, startDatetime.month, startDatetime.day, hour_close, minute_close, 0)
    NbSeconds = np.maximum(np.minimum((end_of_day - startDatetime).seconds / (seconds_in_hour * hours_in_day), hours_in_day), 0)

    test = NbDays.size
    if NbDays.size == 1:
        return NbDays[0] + NbSeconds
    else:
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

    performance_tester = Tester(VECTOR_SIZES=200)
    #performance_tester.Black_Scholes()
    #performance_tester.Monte_Carlo()
    performance_tester.Binomial_Tree(time_steps=5000)


if __name__ == "__main__":
    main()