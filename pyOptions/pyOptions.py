import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import fsolve
from math import pi
import datetime as dt

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
        if div is None:
            div=0.0
        return  fsolve(func=self.__IV_error , x0=np.array(0.2), args=(spot, strike, dmat, rate, price, typ, div))[0]

    def __IV_error(self, x, spot, strike, dmat, rate, price, typ, div):
        return self.pricing(spot, strike, dmat, rate, x, typ, div) - price

class Monte_Carlo(object):

    def __init__(self, days_in_year=252):
        self.days_in_year = days_in_year

    def pricing(self, spot, strike, dmat, rate, vol, typ, div=None, iterations=100000, time_steps=100, antithetic_variates=False, moment_matching=False):

        if div is None:
            div=0.0

        #time interval
        interval = (dmat / self.days_in_year) / time_steps

        S = np.zeros((time_steps+1, iterations))
        S[0] = spot
        for step in range(1, time_steps+1):
            gauss = self.gauss_generator(iterations, antithetic_variates, moment_matching)
            S[step] = S[step-1]*np.exp((rate-div-0.5*vol*vol)*interval + vol*np.sqrt(interval)*gauss)

        call_price = np.exp(-rate * (dmat / self.days_in_year)) * np.sum(np.maximum(S[-1] - strike, 0))/iterations
        put_price = np.exp(-rate * (dmat / self.days_in_year)) * np.sum(np.maximum(strike - S[-1], 0))/iterations
        return {'call': call_price, 'put' : put_price}

    @staticmethod
    def gauss_generator(d, antithetic_variates, moment_matching):
        if antithetic_variates:
            gauss = np.random.randn(int(d/2))
            gauss = np.concatenate((gauss, -gauss))
        else:
            gauss = np.random.randn(d)

        if moment_matching:
            gauss = gauss-np.mean(gauss)
            gauss = gauss / np.std(gauss)
        return gauss

def payoff(spotMat, strike, typ):
    #Vectorized
    return np.where(typ == "C", np.maximum(spotMat - strike, 0), np.maximum(strike - spotMat, 0))

def days_to_maturity(endDate, hours_in_day=6.5, hour_close=16, minute_close=0):
    #Vectorized

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

    BS = Black_Scholes()
    call_price1 = BS.pricing(spot_list,strike_list, dmat_list, rate_list, vol_list, typ_list, div_list)
    print("Call value : {price:.3f}".format(price=float(call_price1[0])))
    call_greeks = BS.greeks(spot_list, strike_list, dmat_list, rate_list, vol_list, typ_list, div_list)
    print("Call delta : {price:.3f}".format(price=float(call_greeks[0][0])))
    print("Call gamma : {price:.3f}".format(price=float(call_greeks[1][0])))
    print("Call theta1 : {price:.3f}".format(price=float(call_greeks[2][0])))
    print("Call vega : {price:.3f}".format(price=float(call_greeks[3][0])))

    put_price1 = BS.pricing(spot, strike, dmat, rate, vol, 'P', div)
    print("Put value : {price:.3f}".format(price=float(put_price1)))
    put_greeks = BS.greeks(spot, strike, dmat, rate, vol, 'P', div)
    print("Put delta : {price:.3f}".format(price=float(put_greeks[0])))
    print("Put gamma : {price:.3f}".format(price=float(put_greeks[1])))
    print("Put theta1 : {price:.3f}".format(price=float(put_greeks[2])))
    print("Put vega : {price:.3f}".format(price=float(put_greeks[3])))


    iv_call = BS.implied_vol(spot,strike, dmat, rate, 'C', 24.13, div)
    print("Call iv : {price:.3f}".format(price=float(iv_call)))

    maturities = [pd.to_datetime('1/1/2019'), pd.to_datetime('9/1/2018'), pd.to_datetime('10/1/2018'), pd.to_datetime('12/1/2018')]
    n_days = days_to_maturity(maturities)
    print("Days to maturity : {price:.3f}".format(price=float(n_days[0])))

    MC = Monte_Carlo()
    mc_price = MC.pricing(spot, strike, dmat, rate, vol, typ, div, moment_matching=True, antithetic_variates=True)
    #mc_call_price = MC.pricing(spot, strike, dmat, rate, vol, typ, div, moment_matching=False, antithetic_variates=False)
    print("MC call price : {price:.3f}".format(price=mc_price['call']))
    print("MC put price : {price:.3f}".format(price=mc_price['put']))

if __name__ == "__main__":
    main()