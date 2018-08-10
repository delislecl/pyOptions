import pytest
from pyOptions import Black_Scholes, Monte_Carlo, days_to_maturity


class Test_pyOptions(object):

    def test_Black_Scholes(self):
        BS = Black_Scholes()

        # Parameters to test
        spot = 200
        strike = 220
        dmat = 2 * 252
        vol = 25 / 100
        rate = 5 / 100
        div = 2 / 100

        #Pricing Calls
        assert BS.pricing(spot, strike, dmat, rate, vol, 'C', div) == pytest.approx(24.13, 0.01)

        #Pricing Puts
        assert BS.pricing(spot, strike, dmat, rate, vol, 'P', div) == pytest.approx(31.04, 0.01)

        #Call Greeks
        call_greeks = BS.greeks(spot, strike, dmat, rate, vol, 'C', div)
        assert call_greeks[0] == pytest.approx(0.48, 0.1) #delta
        assert call_greeks[1] == pytest.approx(0.005, 0.1) #gamma
        assert call_greeks[2] == pytest.approx(-0.033, 0.1)  #theta
        assert call_greeks[3] == pytest.approx(1.018, 0.1)  #vega

        #Put Greeks
        put_greeks = BS.greeks(spot, strike, dmat, rate, vol, 'P', div)
        assert put_greeks[0] == pytest.approx(-0.43, 0.1)  # delta
        assert put_greeks[1] == pytest.approx(0.005, 0.1)  # gamma
        assert put_greeks[2] == pytest.approx(-0.008, 0.1)  # theta
        assert put_greeks[3] == pytest.approx(1.018, 0.1)  # vega

        #Call IV
        iv_call = BS.implied_vol(spot, strike, dmat, rate, 'C', 24.13, div)
        assert iv_call == pytest.approx(0.25, 0.1)

        #Put IV
        iv_call = BS.implied_vol(spot, strike, dmat, rate, 'P', 31.04, div)
        assert iv_call == pytest.approx(0.25, 0.1)

    def test_Monte_Carlo(self):
        MC = Monte_Carlo()

        # Parameters to test
        spot = 200
        strike = 220
        dmat = 2 * 252
        vol = 25 / 100
        rate = 5 / 100
        div = 2 / 100

        #Normal
        mc_prices = MC.pricing(spot, strike, dmat, rate, vol, div)
        assert mc_prices['call'] == pytest.approx(24.13, 0.01)
        assert mc_prices['put'] == pytest.approx(31.04, 0.01)

        #Antithetic variates
        mc_prices = MC.pricing(spot, strike, dmat, rate, vol, div, antithetic_variates=True)
        assert mc_prices['call'] == pytest.approx(24.13, 0.01)
        assert mc_prices['put'] == pytest.approx(31.04, 0.01)

        #Moment matching
        mc_prices = MC.pricing(spot, strike, dmat, rate, vol, div, antithetic_variates=True, moment_matching=True)
        assert mc_prices['call'] == pytest.approx(24.13, 0.01)
        assert mc_prices['put'] == pytest.approx(31.04, 0.01)


if __name__ == '__main__':
    Test_pyOptions().test_Black_Scholes()
