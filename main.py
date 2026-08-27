import yfinance as yf
import numpy as np
import streamlit as st
import pandas as pd
from dict import TICKER_DICT, PERIOD_OPTIONS

# WARNING: run "streamlit run main.py" in terminal

# Streamlit page setup

st.set_page_config(page_title="Fourier Market Filter", layout="wide")
st.title("Fourier Market Signal Filter")

#Choose ticker
selected_label = st.sidebar.selectbox("Select Stock/Index", list(TICKER_DICT.keys()))

#Choose timeframe
selected_period_label = st.sidebar.selectbox("Select Time Horizon", list(PERIOD_OPTIONS.keys()), index=2) # Defaults to 2 Years
selected_period = PERIOD_OPTIONS[selected_period_label]

ticker = TICKER_DICT[selected_label]

#Yes/No log return
log_ret = st.toggle("Log Returns")
#Yes/No Hanning Windowing
hann_window = st.toggle("Hanning Window (for log returns)")

#SLIDER TO CHOOSE CUTOFF POINT (0.001, cut off anything with period >1000 days and so on)
cutoff = st.sidebar.slider("Frequency Cutoff", 0.01, 0.20, 0.05, step=0.005) #min, max,

@st.cache_data(ttl="1d") #as long as inputs no change, it won't continuously get the data from yahoo finance
def fetch_clean_data(symbol, period):
    try:
        stock_data = yf.Ticker(symbol).history(period)
        if stock_data.empty:
            return None
        # Takes close price, drop NaNs
        clean_series = stock_data['Close'].dropna()
        return clean_series
    except Exception:
        return None



prices_series = fetch_clean_data(ticker, selected_period)

if prices_series is not None and len(prices_series) > 10:
    prices = prices_series.values
    dates = prices_series.index

    if log_ret:
        # r_t = ln(P_t / P_{t-1})
        log_returns = np.log(prices[1:] / prices[:-1]) #gets rid of the first value and gets rid of last value respectfully
        #now second day / first day, third / second and so on
        return_dates = dates[1:]

        mean_drift = np.mean(log_returns)
        demeaned_returns = log_returns - mean_drift

        if hann_window:
            #apply hann window
            window = np.hanning(len(demeaned_returns))
            windowed_returns = demeaned_returns * window

            #Divide by 0.5 (the mean) to restore full amplitude
            coherent_gain = np.mean(window)
            fft_spectrum_log = np.fft.fft(windowed_returns) / coherent_gain

            frequencies_log = np.fft.fftfreq(len(windowed_returns)) #len(return) gives no. of days as it is daily, then the function calculates physical frequency for each k
            filtered_spectrum_log = fft_spectrum_log.copy()
            filtered_spectrum_log[np.abs(frequencies_log) > cutoff] = 0

            clean_trend_log = np.fft.ifft(filtered_spectrum_log).real + mean_drift
        else:
            fft_spectrum_log = np.fft.fft(demeaned_returns)
            frequencies_log = np.fft.fftfreq(len(demeaned_returns))
            filtered_spectrum_log = fft_spectrum_log.copy()
            filtered_spectrum_log[np.abs(frequencies_log) > cutoff] = 0

            clean_trend_log = np.fft.ifft(filtered_spectrum_log).real + mean_drift

        chart_data_log = pd.DataFrame({
            "Raw Log Returns": log_returns,
            "Filtered Log Returns": clean_trend_log
        }, index=return_dates)

        st.subheader(f"Raw vs Filtered Log Returns: {selected_label} ({ticker})")
        st.line_chart(chart_data_log)

    # FFT Transformation
    fft_spectrum = np.fft.fft(prices)
    frequencies = np.fft.fftfreq(len(prices)) #taking length to find no. data points, then f=k/n
    #frequency function gives the k, input gives the number of days

    # Low-Pass Filter
    filtered_spectrum = fft_spectrum.copy()
    filtered_spectrum[np.abs(frequencies) > cutoff] = 0 #gets the index of when frequency > cutoff, zeroes them

    # Inverse FFT Reconstruction
    clean_trend = np.fft.ifft(filtered_spectrum).real
    #Do real because python is a bit weird and may give very small imaginary values

    #Converting into DataFrame
    chart_data = pd.DataFrame({
        "Raw Price": prices,
        "Low-Pass Filtered Trend": clean_trend
    }, index=dates)#replaces index with the dates

    st.subheader(f"Price vs. Filtered Trend: {selected_label} ({ticker})")
    st.line_chart(chart_data)

else:
    st.error(f"Could not load valid market data for '{ticker}'. Please verify ticker symbol or internet connection.")