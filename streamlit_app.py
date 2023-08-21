import matplotlib.pyplot as plt
import streamlit as st
from functions import *
st.set_option('deprecation.showPyplotGlobalUse', False)

opt_delta = []
default_config={"max_lookback_years": 5.0,"annual_risk_free_rate": 0.008,"max_position_size": 0.35,"min_position_size": 0.0,"price_data": "fmp",
        "views": {"BABA": [-0.1, 0.1,0.2],
                "NVDA": [-0.1,0.1,0.3],
                "DIS": [-0.1,0.07,0.15],
                "BA": [-0.05,0.07,0.15],
                "XOM": [-0.05,0.07,0.15],
                "META": [-0.05,0.07,0.15],
                "GOOG": [-0.05,0.07,0.15],
                "BAC": [0.0,0.1,0.25]
                }
        }

# Input for config parameters
st.subheader("Parameters")
max_lookback_years = st.number_input('Enter max lookback years:', value=default_config["max_lookback_years"])
annual_risk_free_rate = st.number_input('Enter annual risk free rate:', value=default_config["annual_risk_free_rate"])
max_position_size = st.number_input('Enter max position size:', value=default_config["max_position_size"])
min_position_size = st.number_input('Enter min position size:', value=default_config["min_position_size"])

# Input for views
st.subheader("Views")
default_tickers = "\n".join(default_config["views"].keys())
tickers = st.text_area("Enter tickers, one per line: (ctrl-enter to update)", value=default_tickers).split('\n')
views = {}

st.subheader("-------------------------------------------------------------------------------\n 1. Lower bound annual return for a one standard deviation downward move \n 2. Estimated annual return \n 3. Upper bound for a 1 standard deviation upward move")
for ticker in tickers:
    default_view = ",".join(map(str, default_config["views"].get(ticker, [-0.1, 0.1, 0.2])))
    view_values = st.text_input(f"{ticker}", value=default_view)
    views[ticker] = list(map(float, view_values.split(',')))

st.session_state.config = {
    "max_lookback_years": max_lookback_years,
    "annual_risk_free_rate": annual_risk_free_rate,
    "max_position_size": max_position_size,
    "min_position_size": min_position_size,
    "price_data": "fmp",
    "views": views
}

if "valid" not in st.session_state:
    st.session_state.valid = False

run_litterman_button = st.button("Run Black Litterman")

run_heatmap_button = st.button("Aquire weights")

if run_litterman_button:
    st.session_state.valid = True
    st.session_state.prices, st.session_state.market_prices, st.session_state.mkt_caps, st.session_state.symbols, st.session_state.config = load_data(st.session_state.config)
    st.session_state.covar = risk_models.risk_matrix(st.session_state.prices, method='oracle_approximating')
    st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.rets_fig, st.session_state.covar_bl_fig, st.session_state.corr_bl_fig = calc_black_litterman(st.session_state.market_prices, st.session_state.mkt_caps, st.session_state.covar, st.session_state.config, st.session_state.symbols)
    st.pyplot(st.session_state.rets_fig)
    st.pyplot(st.session_state.covar_bl_fig)
    st.pyplot(st.session_state.corr_bl_fig)
    st.session_state.kelly_w = kelly_optimize(st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.config)
    st.session_state.front_fig = effecient_frontier(st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.config)
    st.pyplot(st.session_state.front_fig)
    st.session_state.opt_delta = 2
    st.session_state.opt_delta = st.number_input("Enter point on frontier:", value=st.session_state.opt_delta)

if run_heatmap_button:
    if st.session_state.valid == None:
        st.warning("Please run Black Litterman first")
    else:

        st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.rets_fig, st.session_state.covar_bl_fig, st.session_state.corr_bl_fig = calc_black_litterman(st.session_state.market_prices, st.session_state.mkt_caps, st.session_state.covar, st.session_state.config, st.session_state.symbols)
        st.pyplot(st.session_state.rets_fig)
        st.pyplot(st.session_state.covar_bl_fig)
        st.pyplot(st.session_state.corr_bl_fig)
        st.session_state.kelly_w = kelly_optimize(st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.config)
        st.session_state.front_fig = effecient_frontier(st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.config)
        st.pyplot(st.session_state.front_fig)
        
        max_quad_util_w, max_quad_util_ef = max_quad_utility_weights(st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.config, st.session_state.opt_delta)
        min_vol_w, min_vol_ef = min_volatility_weights(st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.config)
        max_sharpe_w, max_sharpe_ef = max_sharpe_weights(st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.config)
        cla_max_sharpe_w, cla_max_sharpe_cla = cla_max_sharpe_weights(st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.config)
        cla_min_vol_w, cla_min_vol_cla = cla_min_vol_weights(st.session_state.rets_bl, st.session_state.covar_bl, st.session_state.config)

        weights_df = pd.merge(st.session_state.kelly_w, max_quad_util_w, left_index=True, right_index=True)
        weights_df = pd.merge(weights_df, max_sharpe_w, left_index=True, right_index=True) 
        weights_df = pd.merge(weights_df, cla_max_sharpe_w, left_index=True, right_index=True) 
        weights_df = pd.merge(weights_df, min_vol_w, left_index=True, right_index=True) 
        weights_df = pd.merge(weights_df, cla_min_vol_w, left_index=True, right_index=True)
        heatmap = plot_heatmap(weights_df, 'Portfolio Weighting (%)','Optimization Method', 'Security')
        st.pyplot(heatmap)
