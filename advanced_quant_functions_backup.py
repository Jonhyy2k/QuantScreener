import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from pykalman import KalmanFilter
import pywt  # PyWavelets for wavelet analysis
from scipy import signal
import scipy.stats as stats
from scipy.stats import genpareto
from scipy.optimize import minimize
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Bidirectional, Attention, Dropout, Input
from hmmlearn import hmm
import warnings
import time
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
import requests
from collections import deque
import talib as ta  # Technical analysis library
import pymc as pm  # For Bayesian analysis
import theano.tensor as tt
import prophet  # Facebook Prophet for time series forecasting
from arch import arch_model  # For GARCH models
from scipy.stats import entropy as scipy_entropy
import empyrical as ep  # For financial risk metrics
from joblib import Parallel, delayed  # For parallel processing

# Suppress warnings
warnings.filterwarnings('ignore')


#########################
# 1. Statistical Arbitrage / Pair Trading Signals
#########################

def find_cointegrated_pairs(dataframe, threshold=0.05):
    """
    Find cointegrated pairs among a set of stock prices.

    Parameters:
    -----------
    dataframe: pandas DataFrame
        DataFrame containing stock price time series as columns
    threshold: float
        p-value threshold to consider a pair cointegrated

    Returns:
    --------
    list of tuples
        List of cointegrated pairs (symbol1, symbol2, p_value, hedge_ratio)
    """
    n = dataframe.shape[1]
    pvalue_matrix = np.ones((n, n))
    pairs = []
    keys = dataframe.columns

    # Fill matrix with p-values from cointegration test
    for i in range(n):
        for j in range(i + 1, n):
            # Skip if any NaN values
            if (dataframe.iloc[:, i].isnull().any() or
                    dataframe.iloc[:, j].isnull().any()):
                continue

            # Get stock price series
            stock1 = dataframe.iloc[:, i]
            stock2 = dataframe.iloc[:, j]

            # Ensure both series are positive and non-zero
            if (stock1 <= 0).any() or (stock2 <= 0).any():
                continue

            try:
                # Perform cointegration test
                result = coint(stock1, stock2)
                pvalue = result[1]

                # Store p-value
                pvalue_matrix[i, j] = pvalue
                pvalue_matrix[j, i] = pvalue

                # If p-value is below threshold, consider as cointegrated
                if pvalue < threshold:
                    # Calculate hedge ratio using OLS
                    model = sm.OLS(stock1, stock2).fit()
                    hedge_ratio = model.params[0]
                    pairs.append((keys[i], keys[j], pvalue, hedge_ratio))
            except:
                # If test fails, continue to next pair
                continue

    # Return pairs sorted by p-value
    return sorted(pairs, key=lambda x: x[2])


def calculate_spread_zscore(data, symbols, hedge_ratio=None, window=60):
    """
    Calculate the z-score of the spread between two stocks.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data for both stocks
    symbols: tuple
        Tuple of two stock symbols
    hedge_ratio: float, optional
        Hedge ratio between the stocks. If None, calculated using OLS
    window: int
        Rolling window for z-score calculation

    Returns:
    --------
    pandas Series
        Z-scores of the spread
    """
    stock1 = data[symbols[0]]
    stock2 = data[symbols[1]]

    # Calculate hedge ratio if not provided
    if hedge_ratio is None:
        model = sm.OLS(stock1, stock2).fit()
        hedge_ratio = model.params[0]

    # Calculate spread
    spread = stock1 - hedge_ratio * stock2

    # Calculate z-score
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    z_score = (spread - mean) / std

    return z_score, spread, hedge_ratio


def kalman_filter_pairs(data, symbols, transition_covariance=0.01):
    """
    Apply Kalman filter for dynamic hedge ratio estimation.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data for both stocks
    symbols: tuple
        Tuple of two stock symbols
    transition_covariance: float
        Transition covariance for the Kalman filter

    Returns:
    --------
    tuple
        (hedge_ratios, spreads, spread_mean, spread_std)
    """
    stock1 = data[symbols[0]].values
    stock2 = data[symbols[1]].values

    # Reshape for Kalman filter
    stock1 = stock1.reshape(-1, 1)
    stock2 = stock2.reshape(-1, 1)

    # Define observation matrix
    observation_matrix = np.vstack(
        [stock2, np.ones(stock2.shape)]
    ).T[:, np.newaxis]

    # Initialize Kalman filter
    kf = KalmanFilter(
        n_dim_obs=1,
        n_dim_state=1,
        initial_state_mean=0,
        initial_state_covariance=1,
        transition_matrices=np.array([[1]]),
        observation_matrices=observation_matrix,
        observation_covariance=1.0,
        transition_covariance=transition_covariance
    )

    # Run Kalman filter
    state_means, state_covs = kf.filter(stock1)

    # Extract hedge ratios
    hedge_ratios = state_means.flatten()

    # Calculate spreads using dynamic hedge ratios
    spreads = []
    for i in range(len(stock1)):
        spread = stock1[i] - hedge_ratios[i] * stock2[i]
        spreads.append(spread[0])

    spreads = np.array(spreads)

    # Calculate rolling mean and std for z-score
    window = 60
    spread_mean = np.zeros_like(spreads)
    spread_std = np.zeros_like(spreads)

    for i in range(window, len(spreads)):
        spread_mean[i] = np.mean(spreads[i - window:i])
        spread_std[i] = np.std(spreads[i - window:i])

    return hedge_ratios, spreads, spread_mean, spread_std


def pair_trading_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    """
    Generate pair trading signals based on z-score.

    Parameters:
    -----------
    z_score: pandas Series
        Z-scores of the spread
    entry_threshold: float
        Z-score threshold to enter a position
    exit_threshold: float
        Z-score threshold to exit a position

    Returns:
    --------
    pandas Series
        Trading signals (-1: short, 0: neutral, 1: long)
    """
    signals = pd.Series(0, index=z_score.index)

    position = 0

    for i in range(len(z_score)):
        # Skip NaN values
        if np.isnan(z_score.iloc[i]):
            continue

        # If no position, check entry conditions
        if position == 0:
            if z_score.iloc[i] < -entry_threshold:
                # Spread is too low, go long
                signals.iloc[i] = 1
                position = 1
            elif z_score.iloc[i] > entry_threshold:
                # Spread is too high, go short
                signals.iloc[i] = -1
                position = -1

        # If already in position, check exit conditions
        elif position == 1:  # Long position
            if z_score.iloc[i] > -exit_threshold:
                # Exit long position
                signals.iloc[i] = 0
                position = 0

        elif position == -1:  # Short position
            if z_score.iloc[i] < exit_threshold:
                # Exit short position
                signals.iloc[i] = 0
                position = 0

    return signals


def statistical_arbitrage_analysis(data, top_n_pairs=5, coint_threshold=0.05,
                                   entry_threshold=2.0, exit_threshold=0.5,
                                   use_kalman=True):
    """
    Perform full statistical arbitrage analysis on a set of price data.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data for multiple stocks
    top_n_pairs: int
        Number of top cointegrated pairs to analyze
    coint_threshold: float
        p-value threshold for cointegration test
    entry_threshold: float
        Z-score threshold to enter a position
    exit_threshold: float
        Z-score threshold to exit a position
    use_kalman: bool
        Whether to use Kalman filter for dynamic hedge ratio

    Returns:
    --------
    dict
        Dictionary containing analysis results for top pairs
    """
    # Step 1: Find cointegrated pairs
    print("[INFO] Finding cointegrated pairs...")
    cointegrated_pairs = find_cointegrated_pairs(data, threshold=coint_threshold)

    if not cointegrated_pairs:
        print("[WARNING] No cointegrated pairs found")
        return {}

    # Take top N pairs
    top_pairs = cointegrated_pairs[:top_n_pairs]

    results = {}

    # Step 2: Analyze each pair
    for pair in top_pairs:
        stock1, stock2, pvalue, hedge_ratio = pair
        pair_key = f"{stock1}_{stock2}"

        print(f"[INFO] Analyzing pair: {stock1} - {stock2} (p-value: {pvalue:.5f})")

        # Step 3: Calculate spread and z-score
        if use_kalman:
            # Use Kalman filter for dynamic hedge ratio
            hedge_ratios, spreads, spread_mean, spread_std = kalman_filter_pairs(
                data, (stock1, stock2))

            # Calculate z-scores
            z_scores = np.zeros_like(spreads)
            mask = spread_std > 0
            z_scores[mask] = (spreads[mask] - spread_mean[mask]) / spread_std[mask]

            # Convert to pandas Series
            z_score = pd.Series(z_scores, index=data.index)
            spread = pd.Series(spreads, index=data.index)
            dynamic_hedge_ratio = pd.Series(hedge_ratios, index=data.index)

            results[pair_key] = {
                'stock1': stock1,
                'stock2': stock2,
                'pvalue': pvalue,
                'dynamic_hedge_ratio': dynamic_hedge_ratio,
                'spread': spread,
                'z_score': z_score
            }
        else:
            # Use static hedge ratio
            z_score, spread, static_hedge_ratio = calculate_spread_zscore(
                data, (stock1, stock2), hedge_ratio)

            results[pair_key] = {
                'stock1': stock1,
                'stock2': stock2,
                'pvalue': pvalue,
                'hedge_ratio': static_hedge_ratio,
                'spread': spread,
                'z_score': z_score
            }

        # Step 4: Generate trading signals
        signals = pair_trading_signals(z_score, entry_threshold, exit_threshold)
        results[pair_key]['signals'] = signals

        # Step 5: Calculate performance metrics
        # These will be calculated if you provide actual trading results

    return results


#########################
# 2. Adaptive Time Series Decomposition
#########################

def singular_spectrum_analysis(time_series, window_length=30, n_components=3):
    """
    Apply Singular Spectrum Analysis (SSA) to decompose time series.

    Parameters:
    -----------
    time_series: pandas Series or numpy array
        Input time series data
    window_length: int
        Window length for SSA (typical values: N/4 to N/3)
    n_components: int
        Number of components to reconstruct (trend, cycle, noise)

    Returns:
    --------
    dict
        Dictionary with decomposed components
    """
    if isinstance(time_series, pd.Series):
        time_series = time_series.values

    N = len(time_series)

    # Ensure window length is valid
    if window_length >= N:
        window_length = N // 2

    # Step 1: Create the trajectory matrix (embedding)
    K = N - window_length + 1
    trajectory_matrix = np.zeros((window_length, K))

    for i in range(K):
        trajectory_matrix[:, i] = time_series[i:i + window_length]

    # Step 2: SVD decomposition
    U, sigma, Vt = np.linalg.svd(trajectory_matrix, full_matrices=False)

    # Step 3: Grouping and reconstruction
    components = {}

    if n_components >= len(sigma):
        n_components = len(sigma) - 1

    # Reconstruct components
    for i in range(n_components):
        component = np.zeros(N)

        for j in range(N):
            # Calculate the number of elements in the diagonal
            count = 0

            # Fill the component values
            for k in range(max(0, j - window_length + 1), min(K, j + 1)):
                l = j - k
                component[j] += U[l, i] * sigma[i] * Vt[i, k]
                count += 1

            # Average the values
            if count > 0:
                component[j] /= count

        components[f'component_{i + 1}'] = component

    # Classification of components (simplified approach)
    components['trend'] = components['component_1']

    if n_components >= 3:
        components['cyclical'] = components['component_2'] + components['component_3']
    else:
        components['cyclical'] = components['component_2']

    # Reconstruct noise as residual
    reconstructed = np.zeros(N)
    for i in range(n_components):
        reconstructed += components[f'component_{i + 1}']

    components['noise'] = time_series - reconstructed

    return components


def empirical_mode_decomposition(time_series, max_imfs=10):
    """
    Apply Empirical Mode Decomposition to identify intrinsic market cycles.

    Parameters:
    -----------
    time_series: pandas Series or numpy array
        Input time series data
    max_imfs: int
        Maximum number of Intrinsic Mode Functions to extract

    Returns:
    --------
    dict
        Dictionary with IMFs and residue
    """
    try:
        from PyEMD import EMD
    except ImportError:
        raise ImportError("Please install PyEMD: pip install EMD-signal")

    if isinstance(time_series, pd.Series):
        time_series = time_series.values

    # Initialize EMD
    emd = EMD()

    # Extract IMFs
    imfs = emd.emd(time_series, max_imfs=max_imfs)

    # Store results
    results = {
        'imfs': imfs,
        'n_imfs': imfs.shape[0]
    }

    # Separate IMFs into trend, cycle, and noise (simplified approach)
    if imfs.shape[0] >= 3:
        results['trend'] = imfs[-1]  # Last IMF is usually the trend
        results['cycle'] = np.sum(imfs[1:-1], axis=0)  # Middle IMFs are cyclical components
        results['noise'] = imfs[0]  # First IMF is usually noise
    elif imfs.shape[0] == 2:
        results['trend'] = imfs[1]
        results['cycle'] = np.zeros_like(time_series)
        results['noise'] = imfs[0]
    else:
        results['trend'] = imfs[0]
        results['cycle'] = np.zeros_like(time_series)
        results['noise'] = np.zeros_like(time_series)

    return results


def hilbert_huang_transform(time_series, max_imfs=10):
    """
    Apply Hilbert-Huang Transform to analyze non-stationary data.

    Parameters:
    -----------
    time_series: pandas Series or numpy array
        Input time series data
    max_imfs: int
        Maximum number of Intrinsic Mode Functions to extract

    Returns:
    --------
    dict
        Dictionary with IMFs, instantaneous frequencies, and amplitudes
    """
    try:
        from PyEMD import EMD, Visualisation
        from scipy.signal import hilbert
    except ImportError:
        raise ImportError("Please install PyEMD: pip install EMD-signal")

    if isinstance(time_series, pd.Series):
        time_series_values = time_series.values
        index = time_series.index
    else:
        time_series_values = time_series
        index = np.arange(len(time_series))

    # Step 1: Empirical Mode Decomposition
    emd = EMD()
    imfs = emd.emd(time_series_values, max_imfs=max_imfs)

    # Step 2: Hilbert Transform
    n_imfs = imfs.shape[0]

    # Initialize dictionaries for frequencies and amplitudes
    inst_freqs = {}
    inst_amps = {}

    for i in range(n_imfs):
        # Apply Hilbert transform
        analytic_signal = hilbert(imfs[i])

        # Calculate instantaneous amplitude
        amplitude = np.abs(analytic_signal)

        # Calculate instantaneous phase
        phase = np.unwrap(np.angle(analytic_signal))

        # Calculate instantaneous frequency from phase
        # (converting to radians per sample, then to cycles per sample)
        frequency = np.diff(phase) / (2.0 * np.pi)

        # Append a zero to match original length
        frequency = np.append(frequency, frequency[-1])

        # Store results
        inst_freqs[f'imf_{i + 1}'] = frequency
        inst_amps[f'imf_{i + 1}'] = amplitude

    # Create a dictionary to store all results
    results = {
        'imfs': imfs,
        'n_imfs': n_imfs,
        'inst_freqs': inst_freqs,
        'inst_amps': inst_amps
    }

    # Calculate the Hilbert spectrum (frequency-time-energy distribution)
    # This is a simplified approach
    time_points = np.arange(len(time_series_values))
    hht_spectrum = np.zeros((n_imfs, len(time_points)))

    for i in range(n_imfs):
        for j in range(len(time_points)):
            if j < len(inst_freqs[f'imf_{i + 1}']):
                hht_spectrum[i, j] = inst_amps[f'imf_{i + 1}'][j] * inst_freqs[f'imf_{i + 1}'][j]

    results['hht_spectrum'] = hht_spectrum

    return results


def time_series_decomposition_analysis(data, symbol, methods='all'):
    """
    Analyze time series using multiple decomposition methods.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    methods: str or list
        Methods to use: 'ssa', 'emd', 'hht', or 'all'

    Returns:
    --------
    dict
        Dictionary with decomposition results
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    time_series = data[symbol]

    # Ensure no NaN values
    time_series = time_series.dropna()

    if len(time_series) < 60:
        print(f"[WARNING] Insufficient data for {symbol} ({len(time_series)} points)")
        return None

    results = {}

    # Apply requested methods
    if methods == 'all' or 'ssa' in methods:
        print(f"[INFO] Applying Singular Spectrum Analysis to {symbol}")
        window_length = min(len(time_series) // 4, 50)  # Adaptive window length
        ssa_results = singular_spectrum_analysis(time_series, window_length=window_length)
        results['ssa'] = ssa_results

    if methods == 'all' or 'emd' in methods:
        print(f"[INFO] Applying Empirical Mode Decomposition to {symbol}")
        try:
            emd_results = empirical_mode_decomposition(time_series)
            results['emd'] = emd_results
        except Exception as e:
            print(f"[WARNING] EMD failed for {symbol}: {e}")

    if methods == 'all' or 'hht' in methods:
        print(f"[INFO] Applying Hilbert-Huang Transform to {symbol}")
        try:
            hht_results = hilbert_huang_transform(time_series)
            results['hht'] = hht_results
        except Exception as e:
            print(f"[WARNING] HHT failed for {symbol}: {e}")

    # Add trend strength and cyclicality metrics
    if 'ssa' in results:
        trend = results['ssa']['trend']
        noise = results['ssa']['noise']

        # Trend strength: ratio of trend variance to total variance
        trend_strength = np.var(trend) / np.var(time_series) if np.var(time_series) > 0 else 0

        # Noise ratio: ratio of noise variance to total variance
        noise_ratio = np.var(noise) / np.var(time_series) if np.var(time_series) > 0 else 0

        results['metrics'] = {
            'trend_strength': trend_strength,
            'noise_ratio': noise_ratio
        }

    return results


#########################
# 3. Multi-Timeframe Momentum Integration
#########################

def calculate_adaptive_rsi(data, symbol, alpha=0.2, lookback=14):
    """
    Calculate Adaptive RSI that adjusts to volatility regimes.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    alpha: float
        Smoothing factor for adaptivity (0 < alpha < 1)
    lookback: int
        Base lookback period

    Returns:
    --------
    pandas Series
        Adaptive RSI values
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    prices = data[symbol]

    # Calculate returns
    returns = prices.pct_change().dropna()

    # Calculate volatility
    volatility = returns.rolling(window=lookback * 2).std()

    # Normalize volatility to range [0.5, 2]
    vol_max = volatility.max()
    vol_min = volatility.min()
    if vol_max > vol_min:
        norm_vol = 0.5 + 1.5 * (volatility - vol_min) / (vol_max - vol_min)
    else:
        norm_vol = np.ones_like(volatility)

    # Adaptive lookback periods
    adaptive_lookback = (lookback * norm_vol).round().astype(int)

    # Ensure minimum lookback
    adaptive_lookback = adaptive_lookback.clip(lower=5)

    # Calculate adaptive RSI
    adaptive_rsi = pd.Series(index=prices.index)

    for i in range(len(prices)):
        if i < lookback:
            adaptive_rsi.iloc[i] = np.nan
            continue

        # Get adaptive lookback for this point
        current_lookback = adaptive_lookback.iloc[i]
        if np.isnan(current_lookback) or i < current_lookback:
            adaptive_rsi.iloc[i] = np.nan
            continue

        # Calculate RSI with adaptive lookback
        price_window = prices.iloc[i - current_lookback:i + 1]
        delta = price_window.diff().dropna()

        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=current_lookback).mean().iloc[-1]
        avg_loss = loss.rolling(window=current_lookback).mean().iloc[-1]

        if avg_loss == 0:
            adaptive_rsi.iloc[i] = 100
        else:
            rs = avg_gain / avg_loss
            adaptive_rsi.iloc[i] = 100 - (100 / (1 + rs))

    return adaptive_rsi


def fractal_adaptive_moving_average(prices, lookback=20, fc=0.5, sc=0.05):
    """
    Calculate Fractal Adaptive Moving Average (FRAMA) for noise reduction.

    Parameters:
    -----------
    prices: pandas Series
        Price series
    lookback: int
        Lookback period for FRAMA calculation
    fc: float
        Fast constant (shorter EMA factor)
    sc: float
        Slow constant (longer EMA factor)

    Returns:
    --------
    pandas Series
        FRAMA values
    """
    if len(prices) < lookback * 2:
        return pd.Series(np.nan, index=prices.index)

    # Initialize FRAMA series
    frama = pd.Series(np.nan, index=prices.index)

    # Set initial value
    frama.iloc[lookback - 1] = prices.iloc[lookback - 1]

    # Calculate FRAMA
    for i in range(lookback, len(prices)):
        # Get lookback windows
        window = prices.iloc[i - lookback:i]
        window1 = prices.iloc[i - lookback:i - lookback // 2]
        window2 = prices.iloc[i - lookback // 2:i]

        # Calculate the fractal dimension
        N1 = (window1.max() - window1.min()) / (lookback // 2)
        N2 = (window2.max() - window2.min()) / (lookback // 2)
        N3 = (window.max() - window.min()) / lookback

        # Avoid division by zero
        if N1 + N2 == 0:
            D = 1
        else:
            D = (np.log(N1 + N2) - np.log(N3)) / np.log(2)

        # Ensure D is within valid range [1, 2]
        D = max(1, min(2, D))

        # Calculate alpha
        alpha = np.exp(-4.6 * (D - 1))

        # Ensure alpha is within valid range [sc, fc]
        alpha = max(sc, min(fc, alpha))

        # Calculate FRAMA
        frama.iloc[i] = alpha * prices.iloc[i] + (1 - alpha) * frama.iloc[i - 1]

    return frama


def calculate_multi_timeframe_momentum(data, symbol, timeframes=(5, 20, 60, 120)):
    """
    Calculate momentum across multiple timeframes and create a confluence indicator.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    timeframes: tuple
        Timeframes (lookback periods) to consider

    Returns:
    --------
    dict
        Dictionary with momentum indicators and confluence score
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    prices = data[symbol]

    # Ensure we have enough data
    min_req_data = max(timeframes) * 2
    if len(prices) < min_req_data:
        print(f"[WARNING] Insufficient data for {symbol} ({len(prices)} points, need {min_req_data})")
        return None

    # Initialize results dictionary
    results = {}

    # Calculate price momentum for each timeframe
    momentum = {}
    normalized_momentum = {}

    for tf in timeframes:
        # Price rate of change
        roc = prices.pct_change(periods=tf)
        momentum[f'roc_{tf}'] = roc

        # Normalize to Z-score
        z_score = (roc - roc.rolling(window=tf * 2).mean()) / roc.rolling(window=tf * 2).std()
        normalized_momentum[f'z_roc_{tf}'] = z_score

    # Calculate RSI for each timeframe
    rsi = {}
    for tf in timeframes:
        rsi[f'rsi_{tf}'] = calculate_adaptive_rsi(data, symbol, lookback=tf)

    # Calculate FRAMA for each timeframe
    frama = {}
    for tf in timeframes:
        frama[f'frama_{tf}'] = fractal_adaptive_moving_average(prices, lookback=tf)

    # Create momentum confluence score
    # 1. For ROC: count how many timeframes show positive momentum
    roc_confluence = pd.Series(0, index=prices.index)
    for tf in timeframes:
        roc_confluence += (normalized_momentum[f'z_roc_{tf}'] > 0.5).astype(int)
        roc_confluence -= (normalized_momentum[f'z_roc_{tf}'] < -0.5).astype(int)

    # Normalize to [-1, 1] range
    roc_confluence = roc_confluence / len(timeframes)

    # 2. For RSI: count how many timeframes show bullish or bearish RSI
    rsi_confluence = pd.Series(0, index=prices.index)
    for tf in timeframes:
        rsi_confluence += (rsi[f'rsi_{tf}'] > 60).astype(int)
        rsi_confluence -= (rsi[f'rsi_{tf}'] < 40).astype(int)

    # Normalize to [-1, 1] range
    rsi_confluence = rsi_confluence / len(timeframes)

    # 3. For FRAMA: count how many timeframes show price above or below FRAMA
    frama_confluence = pd.Series(0, index=prices.index)
    for tf in timeframes:
        frama_confluence += (prices > frama[f'frama_{tf}']).astype(int)
        frama_confluence -= (prices < frama[f'frama_{tf}']).astype(int)

    # Normalize to [-1, 1] range
    frama_confluence = frama_confluence / len(timeframes)

    # Combined confluence score (equally weighted)
    combined_confluence = (roc_confluence + rsi_confluence + frama_confluence) / 3

    # Store results
    results['momentum'] = momentum
    results['normalized_momentum'] = normalized_momentum
    results['rsi'] = rsi
    results['frama'] = frama
    results['confluence'] = {
        'roc_confluence': roc_confluence,
        'rsi_confluence': rsi_confluence,
        'frama_confluence': frama_confluence,
        'combined_confluence': combined_confluence
    }

    return results


def adaptive_momentum_indicator(data, symbol, slow_period=26, fast_period=12,
                                volatility_lookback=50, adjust_factor=0.5):
    """
    Create an adaptive momentum oscillator that adjusts to volatility regimes.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    slow_period: int
        Slow period for MACD
    fast_period: int
        Fast period for MACD
    volatility_lookback: int
        Lookback period for volatility calculation
    adjust_factor: float
        Factor to adjust periods based on volatility (0 to 1)

    Returns:
    --------
    dict
        Dictionary with adaptive momentum indicators
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    prices = data[symbol]

    # Ensure no NaN values
    prices = prices.dropna()

    if len(prices) < max(slow_period, volatility_lookback) * 2:
        print(f"[WARNING] Insufficient data for {symbol} ({len(prices)} points)")
        return None

    # Calculate returns and volatility
    returns = prices.pct_change().dropna()
    volatility = returns.rolling(window=volatility_lookback).std()

    # Normalize volatility to range [0.5, 1.5]
    # High volatility -> shorter periods (more responsive)
    # Low volatility -> longer periods (less responsive)
    vol_max = volatility.max()
    vol_min = volatility.min()
    if vol_max > vol_min:
        vol_factor = 1.5 - adjust_factor * (volatility - vol_min) / (vol_max - vol_min)
    else:
        vol_factor = pd.Series(1, index=volatility.index)

    # Calculate adaptive periods
    adaptive_slow = (slow_period * vol_factor).round().astype(int)
    adaptive_fast = (fast_period * vol_factor).round().astype(int)

    # Ensure minimum periods
    adaptive_slow = adaptive_slow.clip(lower=slow_period // 2)
    adaptive_fast = adaptive_fast.clip(lower=fast_period // 2)

    # Initialize adaptive MACD series
    adaptive_macd = pd.Series(np.nan, index=prices.index)
    adaptive_signal = pd.Series(np.nan, index=prices.index)
    adaptive_histogram = pd.Series(np.nan, index=prices.index)

    # Calculate adaptive MACD for each point
    for i in range(max(adaptive_slow.max(), volatility_lookback), len(prices)):
        # Get adaptive periods for this point
        current_slow = adaptive_slow.iloc[i]
        current_fast = adaptive_fast.iloc[i]

        if np.isnan(current_slow) or np.isnan(current_fast) or i < max(current_slow, current_fast):
            continue

        # Calculate exponential moving averages
        slow_ema = prices.iloc[i - current_slow:i + 1].ewm(span=current_slow, adjust=False).mean().iloc[-1]
        fast_ema = prices.iloc[i - current_fast:i + 1].ewm(span=current_fast, adjust=False).mean().iloc[-1]

        # Calculate MACD
        macd = fast_ema - slow_ema
        adaptive_macd.iloc[i] = macd

        # Calculate signal line (9-period EMA of MACD)
        signal_period = max(3, min(9, int(9 * vol_factor.iloc[i])))

        if i >= signal_period and i >= signal_period + max(current_slow, current_fast):
            # Extract MACD values for signal calculation
            macd_window = adaptive_macd.iloc[i - signal_period:i + 1].dropna()

            if len(macd_window) >= signal_period:
                signal = macd_window.ewm(span=signal_period, adjust=False).mean().iloc[-1]
                adaptive_signal.iloc[i] = signal
                adaptive_histogram.iloc[i] = macd - signal

    # Compile results
    results = {
        'adaptive_macd': adaptive_macd,
        'adaptive_signal': adaptive_signal,
        'adaptive_histogram': adaptive_histogram,
        'vol_factor': vol_factor,
        'adaptive_slow_period': adaptive_slow,
        'adaptive_fast_period': adaptive_fast
    }

    return results


#########################
# 4. Non-Linear Machine Learning Predictors
#########################

def prepare_features(data, symbol, lookback_periods=(5, 10, 20, 50),
                     prediction_horizon=5, include_ta=True):
    """
    Prepare features for machine learning models.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    lookback_periods: tuple
        Lookback periods for feature generation
    prediction_horizon: int
        Number of periods ahead to predict
    include_ta: bool
        Whether to include technical indicators as features

    Returns:
    --------
    tuple
        (X, y, feature_names, dates)
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    prices = data[symbol].copy()

    # Ensure no NaN values
    prices = prices.dropna()

    # Calculate returns
    returns = prices.pct_change().fillna(0)

    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).fillna(0)

    # Calculate target variable: n-day ahead return
    target = prices.pct_change(periods=prediction_horizon).shift(-prediction_horizon).fillna(0)

    # Feature set
    features = pd.DataFrame(index=prices.index)

    # 1. Price-based features
    for period in lookback_periods:
        # Price momentum
        features[f'return_{period}d'] = prices.pct_change(periods=period)
        features[f'log_return_{period}d'] = np.log(prices / prices.shift(period))

        # Moving averages
        features[f'sma_{period}d'] = prices.rolling(window=period).mean() / prices - 1
        features[f'ema_{period}d'] = prices.ewm(span=period, adjust=False).mean() / prices - 1

        # Volatility
        features[f'volatility_{period}d'] = returns.rolling(window=period).std()
        features[f'log_volatility_{period}d'] = log_returns.rolling(window=period).std()

        # Price distance from moving averages
        features[f'dist_sma_{period}d'] = prices / prices.rolling(window=period).mean() - 1

        # Rate of change
        features[f'roc_{period}d'] = (prices - prices.shift(period)) / prices.shift(period)

    # 2. Technical indicators (if requested)
    if include_ta:
        try:
            # RSI
            features['rsi_14'] = ta.RSI(prices.values, timeperiod=14)

            # MACD
            macd, macd_signal, macd_hist = ta.MACD(
                prices.values, fastperiod=12, slowperiod=26, signalperiod=9)
            features['macd'] = macd
            features['macd_signal'] = macd_signal
            features['macd_hist'] = macd_hist

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = ta.BBANDS(
                prices.values, timeperiod=20, nbdevup=2, nbdevdn=2)
            features['bb_upper'] = bb_upper
            features['bb_middle'] = bb_middle
            features['bb_lower'] = bb_lower
            features['bb_width'] = (bb_upper - bb_lower) / bb_middle
            features['bb_pctb'] = (prices.values - bb_lower) / (bb_upper - bb_lower)

            # Stochastic
            slowk, slowd = ta.STOCH(
                prices.values, prices.values, prices.values,
                fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3)
            features['stoch_k'] = slowk
            features['stoch_d'] = slowd

            # ADX
            adx = ta.ADX(
                prices.values, prices.values, prices.values, timeperiod=14)
            features['adx'] = adx

            # CCI
            cci = ta.CCI(
                prices.values, prices.values, prices.values, timeperiod=14)
            features['cci'] = cci

            # OBV (using pseudo-volume)
            pseudo_volume = np.ones_like(prices) * 1000
            obv = ta.OBV(prices.values, pseudo_volume)
            features['obv_change'] = (obv - np.roll(obv, 5)) / np.roll(obv, 5)

        except Exception as e:
            print(f"[WARNING] Error calculating technical indicators: {e}")

    # 3. Lagged returns
    for lag in range(1, 6):
        features[f'return_lag_{lag}'] = returns.shift(lag)
        features[f'log_return_lag_{lag}'] = log_returns.shift(lag)

    # 4. Rolling statistics of returns
    for period in [5, 10, 20]:
        features[f'return_mean_{period}d'] = returns.rolling(window=period).mean()
        features[f'return_std_{period}d'] = returns.rolling(window=period).std()
        features[f'return_skew_{period}d'] = returns.rolling(window=period).skew()
        features[f'return_kurt_{period}d'] = returns.rolling(window=period).kurt()

    # 5. Autocorrelation of returns
    for lag in [1, 5]:
        features[f'return_autocorr_{lag}'] = returns.rolling(window=20).apply(
            lambda x: x.autocorr(lag=lag) if len(x.dropna()) > lag else np.nan)

    # 6. Cross-correlations between different timeframes
    features['sma_cross_20_50'] = np.where(
        features['sma_20d'] > features['sma_50d'], 1, -1)
    features['sma_cross_10_20'] = np.where(
        features['sma_10d'] > features['sma_20d'], 1, -1)

    # 7. Mean reversion indicators
    features['dist_52w_high'] = prices / prices.rolling(window=252).max() - 1
    features['dist_52w_low'] = prices / prices.rolling(window=252).min() - 1

    # Fill NaN values
    features = features.fillna(0)

    # Remove rows with infinity
    features = features.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scale features using Z-score normalization
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Create training data (avoiding look-ahead bias)
    # Align features and target, remove prediction_horizon rows from the end
    X = scaled_features[:-prediction_horizon] if prediction_horizon > 0 else scaled_features
    y = target.values[:-prediction_horizon] if prediction_horizon > 0 else target.values

    # Store dates for reference
    dates = prices.index[:-prediction_horizon] if prediction_horizon > 0 else prices.index

    return X, y, features.columns.tolist(), dates


def train_xgboost_model(X, y, test_size=0.2, random_state=42):
    """
    Train an XGBoost model for price prediction.

    Parameters:
    -----------
    X: numpy.ndarray
        Feature matrix
    y: numpy.ndarray
        Target vector
    test_size: float
        Proportion of data to use for testing
    random_state: int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary with trained model and evaluation metrics
    """
    if len(X) < 100:
        print("[WARNING] Insufficient data for training (less than 100 samples)")
        return None

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state)

    # Create XGBoost model with optimized parameters
    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        min_child_weight=2,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=random_state
    )

    # Train model
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        eval_metric='rmse',
        early_stopping_rounds=10,
        verbose=False
    )

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Calculate direction accuracy
    train_dir_acc = np.mean((y_train > 0) == (y_pred_train > 0))
    test_dir_acc = np.mean((y_test > 0) == (y_pred_test > 0))

    # Calculate feature importance
    feature_importance = model.feature_importances_

    results = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_dir_acc': train_dir_acc,
        'test_dir_acc': test_dir_acc,
        'feature_importance': feature_importance
    }

    return results


def train_random_forest_model(X, y, test_size=0.2, random_state=42):
    """
    Train a Random Forest model for price prediction.

    Parameters:
    -----------
    X: numpy.ndarray
        Feature matrix
    y: numpy.ndarray
        Target vector
    test_size: float
        Proportion of data to use for testing
    random_state: int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary with trained model and evaluation metrics
    """
    if len(X) < 100:
        print("[WARNING] Insufficient data for training (less than 100 samples)")
        return None

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False, random_state=random_state)

    # Create Random Forest model with optimized parameters
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=random_state,
        n_jobs=-1
    )

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Calculate direction accuracy
    train_dir_acc = np.mean((y_train > 0) == (y_pred_train > 0))
    test_dir_acc = np.mean((y_test > 0) == (y_pred_test > 0))

    # Calculate feature importance
    feature_importance = model.feature_importances_

    results = {
        'model': model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_dir_acc': train_dir_acc,
        'test_dir_acc': test_dir_acc,
        'feature_importance': feature_importance
    }

    return results


def train_lstm_attention_model(data, symbol, sequence_length=20, prediction_horizon=5,
                               test_size=0.2, epochs=50, batch_size=32):
    """
    Train an LSTM model with attention mechanism for price prediction.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    sequence_length: int
        Number of time steps to use for each sequence
    prediction_horizon: int
        Number of periods ahead to predict
    test_size: float
        Proportion of data to use for testing
    epochs: int
        Number of training epochs
    batch_size: int
        Batch size for training

    Returns:
    --------
    dict
        Dictionary with trained model and evaluation metrics
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    # Prepare features
    X, y, feature_names, dates = prepare_features(
        data, symbol, prediction_horizon=prediction_horizon)

    if len(X) < sequence_length + 50:
        print("[WARNING] Insufficient data for LSTM training")
        return None

    # Prepare sequences for LSTM
    X_sequences = []
    y_sequences = []

    for i in range(len(X) - sequence_length):
        X_sequences.append(X[i:i + sequence_length])
        y_sequences.append(y[i + sequence_length])

    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)

    # Split data
    split_idx = int(len(X_sequences) * (1 - test_size))
    X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
    y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]

    # Define attention mechanism
    class AttentionLayer(tf.keras.layers.Layer):
        def __init__(self, **kwargs):
            super(AttentionLayer, self).__init__(**kwargs)

        def build(self, input_shape):
            self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                                     initializer="normal")
            self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                                     initializer="zeros")
            super(AttentionLayer, self).build(input_shape)

        def call(self, x):
            et = tf.keras.backend.squeeze(tf.keras.backend.tanh(
                tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
            at = tf.keras.backend.softmax(et)
            at = tf.keras.backend.expand_dims(at, axis=-1)
            output = x * at
            return tf.keras.backend.sum(output, axis=1)

        def compute_output_shape(self, input_shape):
            return (input_shape[0], input_shape[-1])

    # Build LSTM model with attention
    input_layer = Input(shape=(X_train.shape[1], X_train.shape[2]))
    lstm_layer = Bidirectional(LSTM(64, return_sequences=True))(input_layer)
    lstm_layer = Dropout(0.2)(lstm_layer)
    lstm_layer = Bidirectional(LSTM(32, return_sequences=True))(lstm_layer)
    attention_layer = AttentionLayer()(lstm_layer)
    dense_layer = Dense(16, activation='relu')(attention_layer)
    output_layer = Dense(1, activation='linear')(dense_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    # Compile model
    model.compile(optimizer='adam', loss='mse')

    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )

    # Make predictions
    y_pred_train = model.predict(X_train).flatten()
    y_pred_test = model.predict(X_test).flatten()

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Calculate direction accuracy
    train_dir_acc = np.mean((y_train > 0) == (y_pred_train > 0))
    test_dir_acc = np.mean((y_test > 0) == (y_pred_test > 0))

    # Get attention weights
    attention_model = Model(
        inputs=model.input,
        outputs=model.get_layer('attention_layer').output
    )

    # Store results
    results = {
        'model': model,
        'attention_model': attention_model,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_dir_acc': train_dir_acc,
        'test_dir_acc': test_dir_acc,
        'history': history.history
    }

    return results


def train_gaussian_process_model(X, y, test_size=0.2, random_state=42):
    """
    Train a Gaussian Process regression model for probabilistic price forecasting.

    Parameters:
    -----------
    X: numpy.ndarray
        Feature matrix
    y: numpy.ndarray
        Target vector
    test_size: float
        Proportion of data to use for testing
    random_state: int
        Random seed for reproducibility

    Returns:
    --------
    dict
        Dictionary with trained model and evaluation metrics
    """
    if len(X) < 100:
        print("[WARNING] Insufficient data for training (less than 100 samples)")
        return None

    # For GP models, we'll use a subset to avoid computational issues
    max_samples = 1000
    if len(X) > max_samples:
        indices = np.linspace(0, len(X) - 1, max_samples, dtype=int)
        X_subset = X[indices]
        y_subset = y[indices]
    else:
        X_subset = X
        y_subset = y

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_subset, y_subset, test_size=test_size, shuffle=False, random_state=random_state)

    # Reduce dimensionality for GP model
    pca = PCA(n_components=min(10, X_train.shape[1]))
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # Create Gaussian Process model
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-10,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=random_state
    )

    # Train model
    model.fit(X_train_pca, y_train)

    # Make predictions with confidence intervals
    y_pred_train, y_std_train = model.predict(X_train_pca, return_std=True)
    y_pred_test, y_std_test = model.predict(X_test_pca, return_std=True)

    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    # Calculate direction accuracy
    train_dir_acc = np.mean((y_train > 0) == (y_pred_train > 0))
    test_dir_acc = np.mean((y_test > 0) == (y_pred_test > 0))

    # Calculate calibration error (how well uncertainty estimates match errors)
    # For a well-calibrated model, about 95% of true values should fall within 2 standard deviations
    train_calibration = np.mean(np.abs(y_train - y_pred_train) <= 2 * y_std_train)
    test_calibration = np.mean(np.abs(y_test - y_pred_test) <= 2 * y_std_test)

    results = {
        'model': model,
        'pca': pca,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'train_dir_acc': train_dir_acc,
        'test_dir_acc': test_dir_acc,
        'train_calibration': train_calibration,
        'test_calibration': test_calibration,
        'prediction_std_train': y_std_train,
        'prediction_std_test': y_std_test
    }

    return results


def ensemble_prediction(data, symbol, prediction_horizon=5, test_size=0.2):
    """
    Create an ensemble of machine learning models for price prediction.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    prediction_horizon: int
        Number of periods ahead to predict
    test_size: float
        Proportion of data to use for testing

    Returns:
    --------
    dict
        Dictionary with ensemble predictions and evaluation metrics
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    print(f"[INFO] Training ensemble models for {symbol} with {prediction_horizon}-day horizon")

    # Prepare features
    X, y, feature_names, dates = prepare_features(
        data, symbol, prediction_horizon=prediction_horizon)

    if len(X) < 200:
        print("[WARNING] Insufficient data for ensemble training (less than 200 samples)")
        return None

    # Split data
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    dates_train, dates_test = dates[:split_idx], dates[split_idx:]

    # Train models
    models = {}
    predictions = {}

    # 1. XGBoost
    print("[INFO] Training XGBoost model...")
    xgb_results = train_xgboost_model(X, y, test_size=test_size)
    if xgb_results:
        models['xgboost'] = xgb_results['model']
        predictions['xgboost'] = {
            'train': models['xgboost'].predict(X_train),
            'test': models['xgboost'].predict(X_test)
        }
        print(
            f"[INFO] XGBoost test RMSE: {xgb_results['test_rmse']:.4f}, Direction accuracy: {xgb_results['test_dir_acc']:.2f}")

    # 2. Random Forest
    print("[INFO] Training Random Forest model...")
    rf_results = train_random_forest_model(X, y, test_size=test_size)
    if rf_results:
        models['random_forest'] = rf_results['model']
        predictions['random_forest'] = {
            'train': models['random_forest'].predict(X_train),
            'test': models['random_forest'].predict(X_test)
        }
        print(
            f"[INFO] Random Forest test RMSE: {rf_results['test_rmse']:.4f}, Direction accuracy: {rf_results['test_dir_acc']:.2f}")

    # 3. Gaussian Process for uncertainty estimation
    # (Only if dataset is not too large)
    if len(X) < 2000:
        print("[INFO] Training Gaussian Process model...")
        gp_results = train_gaussian_process_model(X, y, test_size=test_size)
        if gp_results:
            X_train_pca = gp_results['pca'].transform(X_train)
            X_test_pca = gp_results['pca'].transform(X_test)

            gp_pred_train, gp_std_train = gp_results['model'].predict(X_train_pca, return_std=True)
            gp_pred_test, gp_std_test = gp_results['model'].predict(X_test_pca, return_std=True)

            models['gaussian_process'] = gp_results['model']
            predictions['gaussian_process'] = {
                'train': gp_pred_train,
                'test': gp_pred_test,
                'train_std': gp_std_train,
                'test_std': gp_std_test
            }
            print(
                f"[INFO] Gaussian Process test RMSE: {gp_results['test_rmse']:.4f}, Direction accuracy: {gp_results['test_dir_acc']:.2f}")

    # 4. LSTM model (only if there's enough data)
    if len(X) >= 500:
        try:
            print("[INFO] Training LSTM model with attention...")
            lstm_results = train_lstm_attention_model(
                data, symbol, prediction_horizon=prediction_horizon, test_size=test_size)

            if lstm_results:
                models['lstm'] = lstm_results['model']

                # For LSTM, we need to handle predictions differently due to sequence input
                # We'll only store the test predictions for now
                predictions['lstm'] = {
                    'test': lstm_results['model'].predict(
                        X_test[-len(y_test):].reshape(-1, 1, X.shape[1])).flatten()
                }
                print(
                    f"[INFO] LSTM test RMSE: {lstm_results['test_rmse']:.4f}, Direction accuracy: {lstm_results['test_dir_acc']:.2f}")
        except Exception as e:
            print(f"[WARNING] Error training LSTM model: {e}")

    # Create ensemble predictions
    if not predictions:
        print("[WARNING] No models trained successfully")
        return None

    # Simple ensemble (average of predictions)
    ensemble_pred_train = np.zeros(len(y_train))
    ensemble_pred_test = np.zeros(len(y_test))

    model_count_train = 0
    model_count_test = 0

    for model_name, preds in predictions.items():
        if 'train' in preds and len(preds['train']) == len(y_train):
            ensemble_pred_train += preds['train']
            model_count_train += 1

        if 'test' in preds and len(preds['test']) == len(y_test):
            ensemble_pred_test += preds['test']
            model_count_test += 1

    if model_count_train > 0:
        ensemble_pred_train /= model_count_train

    if model_count_test > 0:
        ensemble_pred_test /= model_count_test

    # Calculate ensemble metrics
    ensemble_train_rmse = np.sqrt(mean_squared_error(y_train, ensemble_pred_train)) if model_count_train > 0 else np.nan
    ensemble_test_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred_test)) if model_count_test > 0 else np.nan

    ensemble_train_dir_acc = np.mean((y_train > 0) == (ensemble_pred_train > 0)) if model_count_train > 0 else np.nan
    ensemble_test_dir_acc = np.mean((y_test > 0) == (ensemble_pred_test > 0)) if model_count_test > 0 else np.nan

    print(f"[INFO] Ensemble test RMSE: {ensemble_test_rmse:.4f}, Direction accuracy: {ensemble_test_dir_acc:.2f}")

    # Prepare the final prediction with confidence interval
    # Use Gaussian Process uncertainty if available, otherwise use ensemble spread
    prediction_std_test = None
    if 'gaussian_process' in predictions and 'test_std' in predictions['gaussian_process']:
        prediction_std_test = predictions['gaussian_process']['test_std']
    else:
        # Calculate standard deviation across model predictions
        pred_array = np.array([preds['test'] for preds in predictions.values()
                               if 'test' in preds and len(preds['test']) == len(y_test)])
        if pred_array.shape[0] > 1:
            prediction_std_test = np.std(pred_array, axis=0)

    # Compile final results
    results = {
        'models': models,
        'predictions': predictions,
        'dates_test': dates_test,
        'y_test': y_test,
        'ensemble_pred_test': ensemble_pred_test,
        'prediction_std_test': prediction_std_test,
        'ensemble_test_rmse': ensemble_test_rmse,
        'ensemble_test_dir_acc': ensemble_test_dir_acc,
        'feature_names': feature_names
    }

    return results


#########################
# 5. Sentiment Analysis from Alternative Data
#########################

def analyze_news_sentiment(symbol, start_date=None, end_date=None, api_key=None):
    """
    Analyze news sentiment for a stock symbol.

    Note: This function uses a mock API for demonstration. In production,
    you would use a real news API with an API key.

    Parameters:
    -----------
    symbol: str
        Stock symbol to analyze
    start_date: str, optional
        Start date in YYYY-MM-DD format
    end_date: str, optional
        End date in YYYY-MM-DD format
    api_key: str, optional
        API key for the news service

    Returns:
    --------
    dict
        Dictionary with sentiment analysis results
    """
    print(f"[INFO] Analyzing news sentiment for {symbol}")

    # This is a mock function - in production, you would use a real API
    # For example, you might use News API, Alpha Vantage News, or Financial Times API

    # Create a mock response
    mock_dates = pd.date_range(start='2023-01-01', periods=30)
    mock_sentiment = np.random.normal(0, 0.5, len(mock_dates))
    mock_volume = np.random.randint(1, 20, len(mock_dates))

    # Create a sentiment dataframe
    sentiment_df = pd.DataFrame({
        'date': mock_dates,
        'sentiment_score': mock_sentiment,
        'news_volume': mock_volume
    })

    # Calculate rolling sentiment
    sentiment_df['rolling_sentiment'] = sentiment_df['sentiment_score'].rolling(window=7).mean()

    # Calculate sentiment momentum
    sentiment_df['sentiment_momentum'] = sentiment_df['rolling_sentiment'].diff(3)

    # Calculate volume-weighted sentiment
    sentiment_df['volume_weighted_sentiment'] = (
            sentiment_df['sentiment_score'] * sentiment_df['news_volume'] /
            sentiment_df['news_volume'].rolling(window=7).mean()
    )

    # Find sentiment extremes
    sentiment_z = (sentiment_df['rolling_sentiment'] -
                   sentiment_df['rolling_sentiment'].mean()) / sentiment_df['rolling_sentiment'].std()

    sentiment_df['sentiment_extreme'] = np.where(
        np.abs(sentiment_z) > 1.5, np.sign(sentiment_z), 0)

    # Calculate signals
    sentiment_df['sentiment_signal'] = np.where(
        sentiment_df['sentiment_extreme'] > 0, 1,
        np.where(sentiment_df['sentiment_extreme'] < 0, -1, 0)
    )

    # Return results
    results = {
        'sentiment_df': sentiment_df,
        'avg_sentiment': sentiment_df['sentiment_score'].mean(),
        'sentiment_volatility': sentiment_df['sentiment_score'].std(),
        'latest_sentiment': sentiment_df['sentiment_score'].iloc[-1],
        'latest_sentiment_momentum': sentiment_df['sentiment_momentum'].iloc[-1],
        'latest_signal': sentiment_df['sentiment_signal'].iloc[-1]
    }

    return results


def analyze_social_media_sentiment(symbol, lookback_days=30, api_key=None):
    """
    Analyze social media sentiment for a stock symbol.

    Note: This function uses mock data for demonstration. In production,
    you would use a real social media API with an API key.

    Parameters:
    -----------
    symbol: str
        Stock symbol to analyze
    lookback_days: int
        Number of days to look back
    api_key: str, optional
        API key for the social media service

    Returns:
    --------
    dict
        Dictionary with sentiment analysis results
    """
    print(f"[INFO] Analyzing social media sentiment for {symbol}")

    # This is a mock function - in production, you would use a real API
    # For example, you might use Twitter API, StockTwits API, or Reddit API

    # Create a mock response
    mock_dates = pd.date_range(end=pd.Timestamp.now(), periods=lookback_days)

    # Generate somewhat realistic sentiment data with trending patterns
    trend = np.cumsum(np.random.normal(0, 0.1, lookback_days))
    normalized_trend = trend / np.max(np.abs(trend)) * 0.5

    # Base sentiment with trend component
    mock_sentiment = normalized_trend + np.random.normal(0, 0.2, lookback_days)

    # Volume with weekly pattern (higher on weekdays, lower on weekends)
    base_volume = 100 + np.random.randint(0, 50, lookback_days)
    weekday_effect = np.array([0.7 if d.weekday() >= 5 else 1.2 for d in mock_dates])
    mock_volume = (base_volume * weekday_effect).astype(int)

    # Mentions with correlation to sentiment
    mock_mentions = (mock_volume * (0.8 + 0.4 * (mock_sentiment + 0.5))).astype(int)

    # Create a sentiment dataframe
    sentiment_df = pd.DataFrame({
        'date': mock_dates,
        'sentiment_score': mock_sentiment,
        'social_volume': mock_volume,
        'mentions': mock_mentions
    })

    # Calculate rolling metrics
    sentiment_df['rolling_sentiment'] = sentiment_df['sentiment_score'].rolling(window=7).mean()
    sentiment_df['rolling_volume'] = sentiment_df['social_volume'].rolling(window=7).mean()

    # Calculate volume-sentiment ratio (VSR)
    sentiment_df['volume_sentiment_ratio'] = (
            sentiment_df['social_volume'] / sentiment_df['social_volume'].rolling(window=7).mean() *
            sentiment_df['sentiment_score']
    )

    # Calculate sentiment divergence (divergence from volume trend)
    sentiment_df['sentiment_divergence'] = (
                                                   sentiment_df['sentiment_score'] -
                                                   sentiment_df['sentiment_score'].rolling(window=14).mean()
                                           ) * np.sign(
        sentiment_df['social_volume'] -
        sentiment_df['social_volume'].rolling(window=14).mean()
    )

    # Calculate sentiment acceleration
    sentiment_df['sentiment_acceleration'] = sentiment_df['rolling_sentiment'].diff().diff()

    # Calculate volume surprise
    sentiment_df['volume_surprise'] = (
            sentiment_df['social_volume'] /
            sentiment_df['social_volume'].rolling(window=14).mean() - 1
    )

    # Return results
    results = {
        'sentiment_df': sentiment_df,
        'avg_sentiment': sentiment_df['sentiment_score'].mean(),
        'sentiment_volatility': sentiment_df['sentiment_score'].std(),
        'latest_sentiment': sentiment_df['sentiment_score'].iloc[-1],
        'latest_volume': sentiment_df['social_volume'].iloc[-1],
        'sentiment_momentum': sentiment_df['rolling_sentiment'].diff(3).iloc[-1],
        'volume_surprise': sentiment_df['volume_surprise'].iloc[-1],
        'sentiment_divergence': sentiment_df['sentiment_divergence'].iloc[-1]
    }

    return results


def analyze_insider_transactions(symbol, lookback_days=90, api_key=None):
    """
    Analyze insider transactions for a stock symbol.

    Note: This function uses mock data for demonstration. In production,
    you would use a real API with an API key.

    Parameters:
    -----------
    symbol: str
        Stock symbol to analyze
    lookback_days: int
        Number of days to look back
    api_key: str, optional
        API key for the insider transactions service

    Returns:
    --------
    dict
        Dictionary with insider transaction analysis results
    """
    print(f"[INFO] Analyzing insider transactions for {symbol}")

    # This is a mock function - in production, you would use a real API
    # For example, you might use SEC Edgar API or a paid service

    # Create a mock response with some realistic patterns
    np.random.seed(sum(ord(c) for c in symbol))  # Use symbol as seed for reproducibility

    # Create dates for the lookback period
    mock_dates = pd.date_range(end=pd.Timestamp.now(), periods=lookback_days)

    # Simulate insider transactions (sparse events)
    num_transactions = np.random.randint(3, 15)  # Random number of transactions
    transaction_indices = np.sort(np.random.choice(range(lookback_days), num_transactions, replace=False))
    transaction_dates = mock_dates[transaction_indices]

    # Transaction types (buy or sell)
    transaction_types = np.random.choice(['BUY', 'SELL'], num_transactions, p=[0.3, 0.7])

    # Transaction values (higher for sells, lower for buys)
    transaction_values = []
    for t_type in transaction_types:
        if t_type == 'BUY':
            # Buys tend to be smaller
            transaction_values.append(np.random.randint(10000, 100000))
        else:
            # Sells tend to be larger
            transaction_values.append(np.random.randint(50000, 500000))

    # Create a dataframe for transactions
    transactions_df = pd.DataFrame({
        'date': transaction_dates,
        'type': transaction_types,
        'value': transaction_values,
        'insider': [f"Insider{i}" for i in range(1, num_transactions + 1)]
    })

    # Calculate aggregated metrics
    total_buy_value = transactions_df[transactions_df['type'] == 'BUY']['value'].sum()
    total_sell_value = transactions_df[transactions_df['type'] == 'SELL']['value'].sum()
    net_transaction_value = total_buy_value - total_sell_value

    buy_count = (transactions_df['type'] == 'BUY').sum()
    sell_count = (transactions_df['type'] == 'SELL').sum()

    # Calculate insider buy/sell ratio
    if sell_count == 0:
        buy_sell_ratio = float('inf')
    else:
        buy_sell_ratio = buy_count / sell_count

    # Calculate value-weighted buy/sell ratio
    if total_sell_value == 0:
        value_ratio = float('inf')
    else:
        value_ratio = total_buy_value / total_sell_value

    # Create a daily time series with cumulative net value
    daily_df = pd.DataFrame({'date': mock_dates})
    daily_df['net_value'] = 0

    for _, row in transactions_df.iterrows():
        idx = daily_df[daily_df['date'] == row['date']].index[0]
        value = row['value'] if row['type'] == 'BUY' else -row['value']
        daily_df.loc[idx:, 'net_value'] += value

    # Calculate 30-day and 90-day net transaction value
    last_30d_net = daily_df['net_value'].iloc[-1] - daily_df['net_value'].iloc[max(0, len(daily_df) - 30)]
    last_90d_net = daily_df['net_value'].iloc[-1] - daily_df['net_value'].iloc[0]

    # Return results
    results = {
        'transactions_df': transactions_df,
        'daily_df': daily_df,
        'total_buy_value': total_buy_value,
        'total_sell_value': total_sell_value,
        'net_transaction_value': net_transaction_value,
        'buy_count': buy_count,
        'sell_count': sell_count,
        'buy_sell_ratio': buy_sell_ratio,
        'value_ratio': value_ratio,
        'last_30d_net': last_30d_net,
        'last_90d_net': last_90d_net
    }

    # Calculate insider signal (-1 to 1 scale)
    if transactions_df.empty:
        results['insider_signal'] = 0
    else:
        # Base signal on net value relative to total transaction volume
        total_volume = total_buy_value + total_sell_value
        if total_volume > 0:
            results['insider_signal'] = net_transaction_value / total_volume
        else:
            results['insider_signal'] = 0

    return results


def alternative_data_integration(data, symbol, news_sentiment=None,
                                 social_sentiment=None, insider_data=None):
    """
    Integrate alternative data with price data.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    news_sentiment: dict, optional
        News sentiment analysis results
    social_sentiment: dict, optional
        Social media sentiment analysis results
    insider_data: dict, optional
        Insider transaction analysis results

    Returns:
    --------
    dict
        Dictionary with integrated analysis results
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    prices = data[symbol].copy()

    # Ensure we have a datetime index
    if not isinstance(prices.index, pd.DatetimeIndex):
        print("[WARNING] Price data index is not a DatetimeIndex, using integer index")
        dates = pd.date_range(end=pd.Timestamp.now(), periods=len(prices))
        price_df = pd.DataFrame({'price': prices.values, 'date': dates})
    else:
        price_df = pd.DataFrame({'price': prices.values, 'date': prices.index})

    # Calculate price returns for correlation analysis
    price_df['return'] = price_df['price'].pct_change()
    price_df['return_5d'] = price_df['price'].pct_change(periods=5)
    price_df['return_20d'] = price_df['price'].pct_change(periods=20)

    # Initialize the integrated features dataframe
    integrated_df = price_df.set_index('date')

    # Set of features to integrate
    integrated_features = {}

    # 1. Integrate news sentiment if available
    if news_sentiment is not None and 'sentiment_df' in news_sentiment:
        sentiment_df = news_sentiment['sentiment_df']

        # Align dates
        sentiment_df = sentiment_df.set_index('date')

        # Merge with price data
        for col in ['sentiment_score', 'rolling_sentiment', 'news_volume',
                    'sentiment_momentum', 'volume_weighted_sentiment']:
            if col in sentiment_df.columns:
                integrated_df[f'news_{col}'] = sentiment_df[col]

        # Fill missing values
        integrated_df = integrated_df.ffill()

        # Add to features dictionary
        integrated_features['news_sentiment'] = {
            'avg_sentiment': news_sentiment['avg_sentiment'],
            'sentiment_volatility': news_sentiment['sentiment_volatility'],
            'latest_sentiment': news_sentiment['latest_sentiment'],
            'latest_sentiment_momentum': news_sentiment['latest_sentiment_momentum'],
        }

    # 2. Integrate social media sentiment if available
    if social_sentiment is not None and 'sentiment_df' in social_sentiment:
        social_df = social_sentiment['sentiment_df']

        # Align dates
        social_df = social_df.set_index('date')

        # Merge with price data
        for col in ['sentiment_score', 'rolling_sentiment', 'social_volume',
                    'volume_sentiment_ratio', 'sentiment_divergence', 'sentiment_acceleration']:
            if col in social_df.columns:
                integrated_df[f'social_{col}'] = social_df[col]

        # Fill missing values
        integrated_df = integrated_df.ffill()

        # Add to features dictionary
        integrated_features['social_sentiment'] = {
            'avg_sentiment': social_sentiment['avg_sentiment'],
            'sentiment_volatility': social_sentiment['sentiment_volatility'],
            'latest_sentiment': social_sentiment['latest_sentiment'],
            'sentiment_momentum': social_sentiment['sentiment_momentum'],
            'volume_surprise': social_sentiment['volume_surprise'],
        }

    # 3. Integrate insider transaction data if available
    if insider_data is not None and 'daily_df' in insider_data:
        insider_df = insider_data['daily_df']

        # Align dates
        insider_df = insider_df.set_index('date')

        # Merge with price data
        if 'net_value' in insider_df.columns:
            integrated_df['insider_net_value'] = insider_df['net_value']

        # Calculate rolling insider metrics
        if 'insider_net_value' in integrated_df.columns:
            integrated_df['insider_net_30d'] = integrated_df['insider_net_value'].diff(30)
            integrated_df['insider_net_90d'] = integrated_df['insider_net_value'].diff(90)

        # Fill missing values
        integrated_df = integrated_df.ffill()

        # Add to features dictionary
        integrated_features['insider_data'] = {
            'buy_sell_ratio': insider_data['buy_sell_ratio'],
            'value_ratio': insider_data['value_ratio'],
            'insider_signal': insider_data['insider_signal'],
            'last_30d_net': insider_data['last_30d_net'],
            'last_90d_net': insider_data['last_90d_net'],
        }

    # Calculate correlations between alternative data and price returns
    correlations = {}

    for prefix, data_type in [('news_', 'News'), ('social_', 'Social Media'), ('insider_', 'Insider')]:
        # Find columns with this prefix
        cols = [col for col in integrated_df.columns if col.startswith(prefix)]

        if cols:
            # Calculate correlation with future returns
            for col in cols:
                for return_col in ['return', 'return_5d', 'return_20d']:
                    # Shift returns back to align with current alternative data
                    corr = integrated_df[col].corr(integrated_df[return_col].shift(-1))
                    correlations[f"{col}_vs_{return_col}"] = corr

    # Create a combined alternative data signal
    signal_components = []

    # Add news sentiment component if available
    if 'news_sentiment_score' in integrated_df.columns:
        # Normalize to [-1, 1] range
        news_z = (integrated_df['news_sentiment_score'] -
                  integrated_df['news_sentiment_score'].mean()) / integrated_df['news_sentiment_score'].std()
        news_signal = np.tanh(news_z)
        signal_components.append(news_signal)

    # Add social sentiment component if available
    if 'social_sentiment_score' in integrated_df.columns:
        # Normalize to [-1, 1] range
        social_z = (integrated_df['social_sentiment_score'] -
                    integrated_df['social_sentiment_score'].mean()) / integrated_df['social_sentiment_score'].std()
        social_signal = np.tanh(social_z)
        signal_components.append(social_signal)

    # Add insider signal component if available
    if 'insider_net_value' in integrated_df.columns:
        # Use insider net value change as signal
        insider_signal = np.tanh(integrated_df['insider_net_30d'] / 1000000)  # Scale to reasonable range
        signal_components.append(insider_signal)

    # Combine signals with equal weight
    if signal_components:
        integrated_df['alternative_data_signal'] = sum(signal_components) / len(signal_components)
    else:
        integrated_df['alternative_data_signal'] = 0

    # Return the integrated results
    results = {
        'integrated_df': integrated_df,
        'integrated_features': integrated_features,
        'correlations': correlations,
        'alternative_data_signal': integrated_df['alternative_data_signal'].iloc[-1] if len(integrated_df) > 0 else 0,
    }

    return results


#########################
# 6. Market Microstructure Indicators
#########################

def calculate_volume_delta(high, low, close, volume, ema_length=20):
    """
    Calculate Volume Delta analysis to track smart money flow.

    Parameters:
    -----------
    high: pandas Series
        High prices
    low: pandas Series
        Low prices
    close: pandas Series
        Close prices
    volume: pandas Series
        Volume data
    ema_length: int
        Length for the EMA calculations

    Returns:
    --------
    pandas DataFrame
        DataFrame with volume delta indicators
    """
    # Calculate typical price and previous close
    typical_price = (high + low + close) / 3
    prev_close = close.shift(1)

    # Avoid NaN in the first element
    prev_close.iloc[0] = close.iloc[0]

    # Calculate buy volume and sell volume based on price action
    total_volume = volume.copy()

    # Buying volume: proportional to close vs low range
    buy_volume = total_volume * (close - low) / (high - low)

    # Selling volume: proportional to high vs close range
    sell_volume = total_volume * (high - close) / (high - low)

    # Handle zero range
    zero_range = (high - low) == 0
    buy_volume[zero_range] = total_volume[zero_range] / 2
    sell_volume[zero_range] = total_volume[zero_range] / 2

    # Calculate delta as difference between buy and sell volume
    volume_delta = buy_volume - sell_volume

    # Calculate cumulative delta
    cumulative_delta = volume_delta.cumsum()

    # Calculate delta EMA
    delta_ema = volume_delta.ewm(span=ema_length, adjust=False).mean()

    # Calculate normalized delta
    # Normalize by the average volume
    avg_volume = total_volume.rolling(window=ema_length).mean()
    normalized_delta = volume_delta / avg_volume

    # Calculate divergences between price and volume delta
    close_change = close.pct_change()
    delta_change = volume_delta.pct_change()

    # Positive divergence: price down but delta up
    positive_divergence = (close_change < 0) & (delta_change > 0)

    # Negative divergence: price up but delta down
    negative_divergence = (close_change > 0) & (delta_change < 0)

    # Return all calculated values
    return pd.DataFrame({
        'close': close,
        'volume': total_volume,
        'buy_volume': buy_volume,
        'sell_volume': sell_volume,
        'volume_delta': volume_delta,
        'cumulative_delta': cumulative_delta,
        'delta_ema': delta_ema,
        'normalized_delta': normalized_delta,
        'positive_divergence': positive_divergence,
        'negative_divergence': negative_divergence
    })


def calculate_order_flow_imbalance(bid_price, ask_price, bid_size, ask_size,
                                   trades, trade_prices, trade_sizes, window=20):
    """
    Calculate Order Flow Imbalance metrics using bid-ask data.

    Note: This function simulates bid-ask data for demonstration.
    In production, you would use real bid-ask data from a market data provider.

    Parameters:
    -----------
    bid_price: pandas Series
        Bid prices (simulated)
    ask_price: pandas Series
        Ask prices (simulated)
    bid_size: pandas Series
        Bid sizes (simulated)
    ask_size: pandas Series
        Ask sizes (simulated)
    trades: pandas Series
        Number of trades (simulated)
    trade_prices: pandas Series
        Trade prices (simulated)
    trade_sizes: pandas Series
        Trade sizes (simulated)
    window: int
        Window size for calculations

    Returns:
    --------
    pandas DataFrame
        DataFrame with order flow imbalance metrics
    """
    # Calculate basic metrics
    spread = ask_price - bid_price
    mid_price = (ask_price + bid_price) / 2

    # Calculate order book imbalance
    book_imbalance = (bid_size - ask_size) / (bid_size + ask_size)

    # Calculate price impact (Kyle's lambda)
    # Higher values indicate lower liquidity
    price_changes = mid_price.diff().abs()
    volume_changes = (bid_size + ask_size).diff().abs()

    # Avoid division by zero
    volume_changes = volume_changes.replace(0, np.nan)

    # Calculate price impact
    kyle_lambda = price_changes / volume_changes
    kyle_lambda = kyle_lambda.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Calculate rolling average of lambda
    kyle_lambda_avg = kyle_lambda.rolling(window=window).mean()

    # Calculate trade imbalance
    # Determine aggressive buys/sells based on trade price relative to mid price
    aggressive_buys = (trade_prices >= mid_price).astype(int) * trade_sizes
    aggressive_sells = (trade_prices <= mid_price).astype(int) * trade_sizes

    # Calculate trade flow imbalance
    trade_flow = aggressive_buys.rolling(window=window).sum() - aggressive_sells.rolling(window=window).sum()
    normalized_trade_flow = trade_flow / trade_sizes.rolling(window=window).sum()

    # Calculate order flow toxicity (VPIN - Volume-synchronized Probability of Informed Trading)
    # Simplified version
    vpin = (aggressive_buys.rolling(window=window).sum().abs() -
            aggressive_sells.rolling(window=window).sum().abs()).abs() / trade_sizes.rolling(window=window).sum()

    # Calculate pressure on the order book
    # Positive: more buying pressure, Negative: more selling pressure
    pressure = book_imbalance * normalized_trade_flow

    # Return all calculated metrics
    return pd.DataFrame({
        'mid_price': mid_price,
        'spread': spread,
        'book_imbalance': book_imbalance,
        'kyle_lambda': kyle_lambda,
        'kyle_lambda_avg': kyle_lambda_avg,
        'trade_flow': trade_flow,
        'normalized_trade_flow': normalized_trade_flow,
        'vpin': vpin,
        'pressure': pressure
    })


def calculate_liquidity_vulnerability(data, symbol, window=20):
    """
    Create liquidity vulnerability indicators to identify potential liquidity gaps.

    Note: This function simulates microstructure data for demonstration.
    In production, you would use real market microstructure data.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    window: int
        Window size for calculations

    Returns:
    --------
    dict
        Dictionary with liquidity vulnerability indicators
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    # Get price data
    prices = data[symbol].copy()

    # Create simulated microstructure data
    # In practice, you would use actual microstructure data from a market data provider
    n = len(prices)
    dates = prices.index

    # Generate simulated bid-ask spread (typically 0.1% to 1% of price)
    # Spread tends to widen during volatile periods
    returns = prices.pct_change().fillna(0)
    volatility = returns.rolling(window=10).std()
    base_spread_pct = 0.001  # 0.1% base spread

    # Spread varies with volatility
    spread_pct = base_spread_pct + volatility * 5
    spread = prices * spread_pct

    # Generate bid and ask prices
    bid_price = prices - spread / 2
    ask_price = prices + spread / 2

    # Generate bid and ask sizes
    # In practice, these would be from the actual order book
    base_size = 1000  # Base size in shares

    # Size varies with inverse of volatility (less depth in volatile markets)
    size_factor = 1 / (1 + volatility * 10)
    size_factor = size_factor.fillna(1)

    bid_size = (base_size * size_factor * (1 + np.random.randn(n) * 0.2)).clip(lower=100)
    ask_size = (base_size * size_factor * (1 + np.random.randn(n) * 0.2)).clip(lower=100)

    # Generate trade data
    # Number of trades tends to increase with volatility
    trades = (base_size / 100 * (1 + volatility * 20)).clip(lower=10).astype(int)

    # Trade prices centered around mid price
    mid_price = (bid_price + ask_price) / 2
    trade_prices = mid_price + np.random.randn(n) * spread / 4

    # Trade sizes following a log-normal distribution
    trade_sizes = np.exp(np.random.randn(n) * 0.5 + np.log(base_size / 10))

    # Calculate order flow imbalance metrics
    ofi_metrics = calculate_order_flow_imbalance(
        bid_price, ask_price, bid_size, ask_size, trades, trade_prices, trade_sizes, window)

    # Calculate volume delta metrics
    # Generate simulated high and low prices
    daily_volatility = prices.pct_change().rolling(window=20).std().fillna(0.01)
    high = prices * (1 + daily_volatility * np.random.rand(n) * 0.5)
    low = prices * (1 - daily_volatility * np.random.rand(n) * 0.5)

    # Ensure high >= close >= low
    high = np.maximum(high, prices)
    low = np.minimum(low, prices)

    # Generate simulated volume
    volume = base_size * 10 * (1 + daily_volatility * 10)

    # Calculate volume delta
    vd_metrics = calculate_volume_delta(high, low, prices, volume)

    # Calculate additional liquidity vulnerability indicators

    # 1. Spread jumps (sudden increases in spread)
    spread_change = spread.pct_change().fillna(0)
    spread_jump = spread_change > spread_change.rolling(window=window).mean() + 2 * spread_change.rolling(
        window=window).std()

    # 2. Order book imbalance extremes
    book_imbalance_extreme = np.abs(ofi_metrics['book_imbalance']) > 0.7

    # 3. Price impact spikes
    lambda_spike = ofi_metrics['kyle_lambda'] > ofi_metrics['kyle_lambda'].rolling(window=window).mean() + 2 * \
                   ofi_metrics['kyle_lambda'].rolling(window=window).std()

    # 4. Volume depletion (falling bid/ask sizes)
    bid_size_change = pd.Series(bid_size).pct_change().fillna(0)
    ask_size_change = pd.Series(ask_size).pct_change().fillna(0)

    size_depletion = (bid_size_change < -0.3) | (ask_size_change < -0.3)

    # 5. One-sided trade flow
    one_sided_flow = np.abs(ofi_metrics['normalized_trade_flow']) > 0.7

    # 6. Price-volume divergence
    price_change = prices.pct_change().fillna(0)
    volume_change = pd.Series(volume).pct_change().fillna(0)

    price_vol_divergence = (
            ((price_change > 0) & (volume_change < 0)) |
            ((price_change < 0) & (volume_change > 0))
    )

    # 7. Composite Liquidity Vulnerability Indicator (CLVI)
    # Weighted sum of individual vulnerability indicators
    clvi = (
            0.2 * spread_jump.astype(int) +
            0.2 * book_imbalance_extreme.astype(int) +
            0.2 * lambda_spike.astype(int) +
            0.15 * size_depletion.astype(int) +
            0.15 * one_sided_flow.astype(int) +
            0.1 * price_vol_divergence.astype(int)
    )

    # Return all calculated indicators
    results = {
        'ofi_metrics': ofi_metrics,
        'vd_metrics': vd_metrics,
        'spread_jump': spread_jump,
        'book_imbalance_extreme': book_imbalance_extreme,
        'lambda_spike': lambda_spike,
        'size_depletion': size_depletion,
        'one_sided_flow': one_sided_flow,
        'price_vol_divergence': price_vol_divergence,
        'clvi': clvi
    }

    return results


def run_market_microstructure_analysis(data, symbol, window=20):
    """
    Run comprehensive market microstructure analysis.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    window: int
        Window size for calculations

    Returns:
    --------
    dict
        Dictionary with market microstructure analysis results
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    # Get price data
    prices = data[symbol].copy()

    print(f"[INFO] Running market microstructure analysis for {symbol}")

    # Calculate volume delta indicators
    try:
        # Generate simulated high, low, and volume data
        n = len(prices)
        daily_volatility = prices.pct_change().rolling(window=20).std().fillna(0.01)

        high = prices * (1 + daily_volatility * np.random.rand(n) * 0.5)
        low = prices * (1 - daily_volatility * np.random.rand(n) * 0.5)

        # Ensure high >= close >= low
        high = pd.Series(np.maximum(high, prices), index=prices.index)
        low = pd.Series(np.minimum(low, prices), index=prices.index)

        # Generate simulated volume
        volume = pd.Series(1000 * (1 + daily_volatility * 10), index=prices.index)

        # Calculate volume delta
        vd_metrics = calculate_volume_delta(high, low, prices, volume)

        print(f"[INFO] Calculated volume delta metrics for {symbol}")
    except Exception as e:
        print(f"[WARNING] Error calculating volume delta metrics: {e}")
        vd_metrics = None

    # Calculate liquidity vulnerability indicators
    try:
        lv_results = calculate_liquidity_vulnerability(data, symbol, window)
        print(f"[INFO] Calculated liquidity vulnerability indicators for {symbol}")
    except Exception as e:
        print(f"[WARNING] Error calculating liquidity vulnerability indicators: {e}")
        lv_results = None

    # Combine all results
    results = {
        'volume_delta': vd_metrics,
        'liquidity_vulnerability': lv_results
    }

    # Extract key insights if results are available
    insights = {}

    if vd_metrics is not None:
        # Last values for key metrics
        insights['volume_delta_last'] = vd_metrics['volume_delta'].iloc[-1]
        insights['cumulative_delta_last'] = vd_metrics['cumulative_delta'].iloc[-1]
        insights['delta_ema_last'] = vd_metrics['delta_ema'].iloc[-1]

        # Divergences
        insights['positive_divergence_count'] = vd_metrics['positive_divergence'].sum()
        insights['negative_divergence_count'] = vd_metrics['negative_divergence'].sum()

        # Recent trend
        delta_trend = vd_metrics['delta_ema'].iloc[-5:].mean() - vd_metrics['delta_ema'].iloc[-10:-5].mean()
        insights['delta_trend'] = delta_trend

        # Signal from volume delta
        if delta_trend > 0 and insights['delta_ema_last'] > 0:
            insights['volume_delta_signal'] = "Bullish"
        elif delta_trend < 0 and insights['delta_ema_last'] < 0:
            insights['volume_delta_signal'] = "Bearish"
        else:
            insights['volume_delta_signal'] = "Neutral"

    if lv_results is not None and 'clvi' in lv_results:
        # Latest CLVI value
        insights['clvi_last'] = lv_results['clvi'].iloc[-1]

        # Average CLVI over last 5 periods
        insights['clvi_avg_5'] = lv_results['clvi'].iloc[-5:].mean()

        # CLVI threshold breaches
        insights['high_vulnerability_count'] = (lv_results['clvi'] > 0.5).sum()

        # Liquidity signal
        if insights['clvi_last'] > 0.7:
            insights['liquidity_signal'] = "High Vulnerability"
        elif insights['clvi_last'] > 0.4:
            insights['liquidity_signal'] = "Moderate Vulnerability"
        else:
            insights['liquidity_signal'] = "Low Vulnerability"

    results['insights'] = insights

    return results


#########################
# 7. Multi-Fractal Market Analysis
#########################

def calculate_hurst_exponent(time_series, max_lag=20, min_lag=2):
    """
    Calculate Hurst exponent across multiple timeframes to identify fractal patterns.

    Parameters:
    -----------
    time_series: pandas Series or numpy array
        Input time series data
    max_lag: int
        Maximum lag for R/S analysis
    min_lag: int
        Minimum lag for R/S analysis

    Returns:
    --------
    float
        Hurst exponent
    """
    if isinstance(time_series, pd.Series):
        time_series = time_series.values

    # Check for NaN values
    if np.isnan(time_series).any():
        time_series = pd.Series(time_series).fillna(method='ffill').values

    # Check if we have enough data
    if len(time_series) < max_lag * 4:
        print(f"[WARNING] Not enough data for Hurst calculation: {len(time_series)}")
        return 0.5  # Return neutral value

    # Calculate returns
    returns = np.diff(np.log(time_series))

    # Create a range of lag values
    lags = range(min_lag, max_lag + 1)

    # Calculate R/S values for each lag
    rs_values = []

    for lag in lags:
        # Number of subseries
        n_chunks = int(len(returns) / lag)

        if n_chunks < 1:
            continue

        # Split returns into subseries
        chunks = [returns[i:i + lag] for i in range(0, n_chunks * lag, lag)]

        # Calculate R/S for each subseries
        rs_array = []

        for chunk in chunks:
            if len(chunk) < lag:
                continue

            # Calculate standard deviation
            std = np.std(chunk)
            if std == 0:
                continue

            # Calculate mean-adjusted series
            mean_adj = chunk - np.mean(chunk)

            # Calculate cumulative sum
            cum_sum = np.cumsum(mean_adj)

            # Calculate range
            r = np.max(cum_sum) - np.min(cum_sum)

            # Calculate R/S value
            rs = r / std

            rs_array.append(rs)

        if rs_array:
            # Average R/S value for this lag
            rs_values.append(np.mean(rs_array))
        else:
            rs_values.append(np.nan)

    # Clean up the results
    rs_values = np.array([x for x in rs_values if not np.isnan(x)])
    lags = np.array([lags[i] for i in range(len(rs_values))])

    if len(lags) < 4:
        print("[WARNING] Not enough valid R/S values for Hurst calculation")
        return 0.5  # Return neutral value

    # Log-log regression to find Hurst exponent
    x = np.log10(lags)
    y = np.log10(rs_values)

    # Linear regression
    slope, _, _, _, _ = stats.linregress(x, y)

    # Hurst exponent is the slope of the line
    h = slope

    return h


def calculate_multifractal_spectrum(time_series, q_range=(-5, 5), q_steps=21, scale_range=(4, 64), scale_steps=16):
    """
    Calculate the multifractal spectrum to analyze fractal dimensions across scales.

    Parameters:
    -----------
    time_series: pandas Series or numpy array
        Input time series data
    q_range: tuple
        Range of q values for multifractal analysis
    q_steps: int
        Number of q values to use
    scale_range: tuple
        Range of scales (window sizes) to use
    scale_steps: int
        Number of scale steps to use

    Returns:
    --------
    dict
        Dictionary with multifractal spectrum results
    """
    if isinstance(time_series, pd.Series):
        time_series = time_series.values

    # Check for NaN values
    if np.isnan(time_series).any():
        time_series = pd.Series(time_series).fillna(method='ffill').values

    # Check if we have enough data
    if len(time_series) < scale_range[1] * 4:
        print(f"[WARNING] Not enough data for multifractal analysis: {len(time_series)}")
        return None

    # Generate q values
    q_values = np.linspace(q_range[0], q_range[1], q_steps)

    # Generate scales
    scales = np.logspace(np.log10(scale_range[0]), np.log10(scale_range[1]), scale_steps).astype(int)

    # Calculate fluctuation function for each q and scale
    f_q_s = np.zeros((len(q_values), len(scales)))

    # Calculate returns
    returns = np.diff(np.log(time_series))

    # Calculate mean for each scale
    for j, scale in enumerate(scales):
        # Number of segments
        n_segments = int(len(returns) / scale)

        if n_segments < 4:  # Need enough segments
            continue

        # Split into segments
        segments = np.array_split(returns[:n_segments * scale], n_segments)

        # Calculate fluctuation for each segment
        fluctuations = []

        for segment in segments:
            # Profile (cumulative sum)
            profile = np.cumsum(segment - np.mean(segment))

            # Polynomial fit (detrending)
            x = np.arange(len(profile))
            coeffs = np.polyfit(x, profile, 1)
            trend = np.polyval(coeffs, x)

            # Detrended profile
            detrended = profile - trend

            # Fluctuation (standard deviation)
            fluctuation = np.std(detrended)

            fluctuations.append(fluctuation)

        # Calculate F(q, s) for each q
        for i, q in enumerate(q_values):
            if q == 0:
                # For q=0, use logarithmic average
                f_q_s[i, j] = np.exp(0.5 * np.mean(np.log(np.array(fluctuations) ** 2)))
            else:
                # For q≠0, use regular formula
                f_q_s[i, j] = (np.mean(np.array(fluctuations) ** q)) ** (1 / q)

    # Calculate Hurst exponent for each q
    h_q = np.zeros(len(q_values))
    r_squared = np.zeros(len(q_values))

    for i, q in enumerate(q_values):
        # Log-log regression
        y = np.log10(f_q_s[i, :])
        x = np.log10(scales)

        # Remove NaN values
        mask = ~np.isnan(y)
        if np.sum(mask) < 4:
            h_q[i] = np.nan
            r_squared[i] = np.nan
            continue

        # Linear regression
        slope, intercept, r_value, _, _ = stats.linregress(x[mask], y[mask])

        h_q[i] = slope
        r_squared[i] = r_value ** 2

    # Calculate the multifractal spectrum
    tau_q = q_values * h_q - 1

    # Calculate the singularity spectrum
    alpha = np.gradient(tau_q, q_values)
    f_alpha = q_values * alpha - tau_q

    # Calculate width of the multifractal spectrum
    if not np.isnan(alpha).all():
        alpha_range = np.nanmax(alpha) - np.nanmin(alpha)
    else:
        alpha_range = np.nan

    # Return results
    results = {
        'q_values': q_values,
        'h_q': h_q,
        'r_squared': r_squared,
        'tau_q': tau_q,
        'alpha': alpha,
        'f_alpha': f_alpha,
        'alpha_range': alpha_range
    }

    return results


def calculate_fractal_dimension(time_series, max_step=20):
    """
    Calculate the fractal dimension of a time series using the box-counting method.

    Parameters:
    -----------
    time_series: pandas Series or numpy array
        Input time series data
    max_step: int
        Maximum step size for box counting

    Returns:
    --------
    float
        Fractal dimension
    """
    if isinstance(time_series, pd.Series):
        time_series = time_series.values

    # Check for NaN values
    if np.isnan(time_series).any():
        time_series = pd.Series(time_series).fillna(method='ffill').values

    # Normalize the series to [0, 1] range
    normalized = (time_series - np.min(time_series)) / (np.max(time_series) - np.min(time_series))

    # Set up the steps for box counting
    steps = range(1, max_step + 1)

    # Count boxes for each step size
    box_counts = []

    for step in steps:
        # Create grid
        x_grid = np.arange(0, len(normalized), step)
        y_grid = np.arange(0, 1, step / len(normalized))

        # Count filled boxes
        count = 0

        for i in range(len(x_grid) - 1):
            for j in range(len(y_grid) - 1):
                # Check if box contains part of the time series
                x_min, x_max = x_grid[i], x_grid[i + 1]
                y_min, y_max = y_grid[j], y_grid[j + 1]

                # Get values in this x range
                values_in_range = normalized[int(x_min):int(x_max)]

                if len(values_in_range) > 0:
                    if np.any((values_in_range >= y_min) & (values_in_range < y_max)):
                        count += 1

        box_counts.append(count)

    # Log-log regression to find fractal dimension
    x = np.log(1 / np.array(steps))
    y = np.log(np.array(box_counts))

    # Remove NaN or infinite values
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return 1.0  # Return null result

    # Linear regression
    slope, _, _, _, _ = stats.linregress(x[mask], y[mask])

    return slope


def detect_self_similar_patterns(time_series, window=20, n_patterns=3, min_pattern_size=5, max_pattern_size=20):
    """
    Detect self-similar patterns in the time series.

    Parameters:
    -----------
    time_series: pandas Series or numpy array
        Input time series data
    window: int
        Rolling window for pattern detection
    n_patterns: int
        Number of patterns to detect
    min_pattern_size: int
        Minimum pattern size
    max_pattern_size: int
        Maximum pattern size

    Returns:
    --------
    dict
        Dictionary with detected patterns and their similarity scores
    """
    if isinstance(time_series, pd.Series):
        ts_values = time_series.values
        dates = time_series.index
    else:
        ts_values = time_series
        dates = np.arange(len(time_series))

    # Normalize the time series
    normalized = (ts_values - np.mean(ts_values)) / np.std(ts_values)

    # Calculate returns for additional patterns
    if len(normalized) > 1:
        returns = np.diff(normalized)
        returns = np.insert(returns, 0, 0)  # Add a 0 at the beginning to maintain length
    else:
        returns = np.zeros_like(normalized)

    # Initialize results
    pattern_results = []

    # Try different pattern sizes
    for pattern_size in range(min_pattern_size, min(max_pattern_size + 1, len(normalized) // 3)):
        # Calculate rolling windows of the specified size
        patterns = []

        for i in range(len(normalized) - pattern_size + 1):
            pattern = normalized[i:i + pattern_size]
            patterns.append((i, pattern))

        # Calculate similarity matrix (correlation)
        n_patterns_found = len(patterns)
        similarity_matrix = np.zeros((n_patterns_found, n_patterns_found))

        for i in range(n_patterns_found):
            for j in range(i + 1, n_patterns_found):
                corr = np.corrcoef(patterns[i][1], patterns[j][1])[0, 1]
                similarity_matrix[i, j] = corr
                similarity_matrix[j, i] = corr

        # Find patterns with highest average similarity
        avg_similarity = np.sum(similarity_matrix, axis=1) / (n_patterns_found - 1)

        # Get top patterns
        top_indices = np.argsort(avg_similarity)[-n_patterns:]

        for idx in top_indices:
            pattern_idx, pattern_values = patterns[idx]

            # Calculate average similarity with other patterns
            avg_sim = avg_similarity[idx]

            # Calculate pattern strength (average similarity weighted by size)
            pattern_strength = avg_sim * pattern_size / max_pattern_size

            # Store pattern details
            pattern_details = {
                'start_idx': pattern_idx,
                'end_idx': pattern_idx + pattern_size,
                'start_date': dates[pattern_idx],
                'end_date': dates[pattern_idx + pattern_size - 1],
                'pattern_size': pattern_size,
                'pattern_values': pattern_values,
                'avg_similarity': avg_sim,
                'pattern_strength': pattern_strength
            }

            pattern_results.append(pattern_details)

    # Sort patterns by strength
    pattern_results = sorted(pattern_results, key=lambda x: x['pattern_strength'], reverse=True)

    # Keep only the top n_patterns
    top_patterns = pattern_results[:n_patterns]

    # De-duplicate overlapping patterns
    final_patterns = []

    for pattern in top_patterns:
        # Check for overlap with already selected patterns
        overlapping = False

        for selected in final_patterns:
            # Check if pattern overlaps with selected pattern
            if (pattern['start_idx'] < selected['end_idx'] and
                    pattern['end_idx'] > selected['start_idx']):
                overlapping = True
                break

        if not overlapping:
            final_patterns.append(pattern)

        if len(final_patterns) >= n_patterns:
            break

    # Calculate future projection for each pattern
    for pattern in final_patterns:
        # Extract pattern values
        pattern_vals = pattern['pattern_values']

        # Next n values as projection (where n is pattern size)
        pattern_size = pattern['pattern_size']
        end_idx = pattern['end_idx']

        # Check if we have enough data for projection
        if end_idx + pattern_size <= len(normalized):
            future_vals = normalized[end_idx:end_idx + pattern_size]

            # Calculate correlation with actual future
            future_corr = np.corrcoef(pattern_vals, future_vals)[0, 1]

            pattern['future_correlation'] = future_corr
        else:
            pattern['future_correlation'] = None

    return {
        'patterns': final_patterns,
        'n_patterns_found': len(final_patterns)
    }


def run_multifractal_analysis(data, symbol, timeframes=(1, 5, 20, 60)):
    """
    Run comprehensive multifractal analysis across multiple timeframes.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    timeframes: tuple
        Timeframes (in days) to analyze

    Returns:
    --------
    dict
        Dictionary with multifractal analysis results across timeframes
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    # Get price data
    prices = data[symbol].copy()

    print(f"[INFO] Running multifractal analysis for {symbol}")

    results = {}

    # Calculate for each timeframe
    for tf in timeframes:
        # Resample the data (simple moving average)
        if tf == 1:
            # Daily data (no resampling)
            tf_prices = prices.copy()
            tf_label = 'daily'
        else:
            # Resampled data
            tf_prices = prices.rolling(window=tf).mean().dropna()
            tf_label = f'{tf}d'

        print(f"[INFO] Analyzing {tf_label} timeframe ({len(tf_prices)} data points)")

        # Skip if not enough data
        if len(tf_prices) < 50:
            print(f"[WARNING] Not enough data for {tf_label} timeframe. Skipping.")
            continue

        tf_results = {}

        # 1. Calculate Hurst exponent
        try:
            h = calculate_hurst_exponent(tf_prices)

            # Interpret Hurst exponent
            if h < 0.4:
                regime = "Strong Mean Reversion"
            elif h < 0.45:
                regime = "Mean Reversion"
            elif h < 0.55:
                regime = "Random Walk"
            elif h < 0.65:
                regime = "Trending"
            else:
                regime = "Strong Trending"

            tf_results['hurst'] = h
            tf_results['regime'] = regime

            print(f"[INFO] {tf_label} Hurst: {h:.3f} - {regime}")
        except Exception as e:
            print(f"[WARNING] Error calculating Hurst exponent for {tf_label}: {e}")
            tf_results['hurst'] = None
            tf_results['regime'] = None

        # 2. Calculate multifractal spectrum
        try:
            mf_spectrum = calculate_multifractal_spectrum(tf_prices)

            if mf_spectrum is not None:
                tf_results['multifractal_spectrum'] = mf_spectrum

                # Calculate multifractality strength (width of spectrum)
                alpha_range = mf_spectrum['alpha_range']

                if not np.isnan(alpha_range):
                    if alpha_range > 0.5:
                        mf_strength = "Strong"
                    elif alpha_range > 0.3:
                        mf_strength = "Moderate"
                    else:
                        mf_strength = "Weak"

                    tf_results['multifractality_strength'] = mf_strength
                    tf_results['alpha_range'] = alpha_range

                    print(f"[INFO] {tf_label} Multifractality: {mf_strength} (α range: {alpha_range:.3f})")
        except Exception as e:
            print(f"[WARNING] Error calculating multifractal spectrum for {tf_label}: {e}")
            tf_results['multifractal_spectrum'] = None

        # 3. Calculate fractal dimension
        try:
            fd = calculate_fractal_dimension(tf_prices)
            tf_results['fractal_dimension'] = fd

            print(f"[INFO] {tf_label} Fractal Dimension: {fd:.3f}")
        except Exception as e:
            print(f"[WARNING] Error calculating fractal dimension for {tf_label}: {e}")
            tf_results['fractal_dimension'] = None

        # 4. Detect self-similar patterns
        try:
            patterns = detect_self_similar_patterns(tf_prices)

            tf_results['self_similar_patterns'] = patterns

            if patterns['n_patterns_found'] > 0:
                # Get average pattern strength
                avg_strength = np.mean([p['pattern_strength'] for p in patterns['patterns']])
                tf_results['avg_pattern_strength'] = avg_strength

                print(
                    f"[INFO] {tf_label} Found {patterns['n_patterns_found']} self-similar patterns (avg strength: {avg_strength:.3f})")
            else:
                print(f"[INFO] {tf_label} No self-similar patterns found")
        except Exception as e:
            print(f"[WARNING] Error detecting self-similar patterns for {tf_label}: {e}")
            tf_results['self_similar_patterns'] = None

        # Store results for this timeframe
        results[tf_label] = tf_results

    # Calculate cross-timeframe metrics

    # 1. Hurst correlation across timeframes
    hurst_values = [results[tf]['hurst'] for tf in results if
                    'hurst' in results[tf] and results[tf]['hurst'] is not None]

    if len(hurst_values) > 1:
        avg_hurst = np.mean(hurst_values)
        hurst_range = np.max(hurst_values) - np.min(hurst_values)

        results['cross_timeframe'] = {
            'avg_hurst': avg_hurst,
            'hurst_range': hurst_range
        }

        # Interpret cross-timeframe consistency
        if hurst_range < 0.1:
            results['cross_timeframe']['fractal_consistency'] = "High"
        elif hurst_range < 0.2:
            results['cross_timeframe']['fractal_consistency'] = "Moderate"
        else:
            results['cross_timeframe']['fractal_consistency'] = "Low"

        print(f"[INFO] Cross-timeframe Hurst average: {avg_hurst:.3f}, range: {hurst_range:.3f}")
        print(f"[INFO] Fractal consistency: {results['cross_timeframe']['fractal_consistency']}")

    return results


#########################
# 8. Extreme Value Theory for Tail Risk
#########################

def fit_gpd_model(returns, threshold_method='percentile', threshold_value=0.95):
    """
    Fit a Generalized Pareto Distribution to returns tails.

    Parameters:
    -----------
    returns: pandas Series or numpy array
        Return series
    threshold_method: str
        Method to select threshold ('percentile', 'value')
    threshold_value: float
        Threshold parameter (percentile or absolute value)

    Returns:
    --------
    dict
        Dictionary with GPD model parameters
    """
    if isinstance(returns, pd.Series):
        returns = returns.values

    # Separate positive and negative returns
    positive_returns = returns[returns > 0]
    negative_returns = -returns[returns < 0]  # Convert to positive values

    results = {}

    # Function to fit GPD to one tail
    def fit_one_tail(tail_data, tail_name):
        # Select threshold
        if threshold_method == 'percentile':
            threshold = np.percentile(tail_data, threshold_value * 100)
        else:
            threshold = threshold_value

        # Get exceedances over threshold
        exceedances = tail_data[tail_data > threshold] - threshold

        if len(exceedances) < 10:
            print(f"[WARNING] Not enough exceedances for {tail_name} tail. Skipping GPD fit.")
            return None

        try:
            # Fit GPD
            shape, loc, scale = genpareto.fit(exceedances)

            # Calculate goodness of fit
            # Kolmogorov-Smirnov test
            ks_stat, ks_pvalue = stats.kstest(
                exceedances,
                lambda x: genpareto.cdf(x, shape, loc, scale)
            )

            # Return parameters
            return {
                'shape': shape,
                'loc': loc,
                'scale': scale,
                'threshold': threshold,
                'n_exceedances': len(exceedances),
                'exceedance_pct': len(exceedances) / len(tail_data),
                'ks_stat': ks_stat,
                'ks_pvalue': ks_pvalue
            }
        except Exception as e:
            print(f"[WARNING] Error fitting GPD to {tail_name} tail: {e}")
            return None

    # Fit positive tail
    results['positive_tail'] = fit_one_tail(positive_returns, 'positive')

    # Fit negative tail
    results['negative_tail'] = fit_one_tail(negative_returns, 'negative')

    return results


def calculate_expected_shortfall(returns, confidence_levels=(0.95, 0.99), method='empirical'):
    """
    Calculate Expected Shortfall (Conditional Value at Risk) at different confidence levels.

    Parameters:
    -----------
    returns: pandas Series or numpy array
        Return series
    confidence_levels: tuple
        Confidence levels for ES calculation
    method: str
        Method to use ('empirical', 'gpd')

    Returns:
    --------
    dict
        Dictionary with Expected Shortfall values
    """
    if isinstance(returns, pd.Series):
        returns_arr = returns.values
    else:
        returns_arr = returns

    results = {}

    # Calculate empirical ES
    def calc_empirical_es(ret, level):
        var = np.percentile(ret, (1 - level) * 100)
        es = ret[ret <= var].mean()
        return es

    # Calculate ES using GPD model
    def calc_gpd_es(ret, level, tail_params):
        if tail_params is None:
            return None

        threshold = tail_params['threshold']
        shape = tail_params['shape']
        scale = tail_params['scale']

        # Probability of exceeding threshold
        p_exceed = len(ret[ret > threshold]) / len(ret)

        # Calculate VaR from GPD
        var = threshold + (scale / shape) * ((1 - level) / p_exceed) ** (-shape) - 1

        # Calculate ES from GPD
        if shape < 1:
            es = var / (1 - shape) + (scale - shape * threshold) / (1 - shape)
        else:
            # Shape >= 1 means infinite ES
            es = float('inf')

        return es

    # Calculate ES for each confidence level
    for level in confidence_levels:
        key = f'es_{int(level * 100)}'

        if method == 'empirical':
            # Empirical ES
            results[key] = calc_empirical_es(returns_arr, level)
        elif method == 'gpd':
            # Fit GPD model
            gpd_model = fit_gpd_model(returns_arr)

            # Use negative tail for ES
            negative_tail = gpd_model['negative_tail']

            # Calculate ES using GPD
            results[key] = calc_gpd_es(-returns_arr, level, negative_tail)
        else:
            raise ValueError(f"Unknown method: {method}")

    return results


def calculate_tail_risk_metrics(returns, window=252):
    """
    Calculate comprehensive tail risk metrics.

    Parameters:
    -----------
    returns: pandas Series
        Return series
    window: int
        Rolling window for calculations

    Returns:
    --------
    dict
        Dictionary with tail risk metrics
    """
    # Check for NaN values
    returns = returns.fillna(0)

    # Initialize results dictionary
    results = {}

    # Calculate rolling skewness and kurtosis
    results['rolling_skew'] = returns.rolling(window=window).skew()
    results['rolling_kurt'] = returns.rolling(window=window).kurt()

    # Calculate rolling max drawdown
    rolling_max = returns.rolling(window=window).cummax()
    rolling_drawdown = returns / rolling_max - 1
    results['rolling_max_drawdown'] = rolling_drawdown.rolling(window=window).min()

    # Calculate VaR at different confidence levels
    for level in [0.95, 0.99]:
        var_key = f'var_{int(level * 100)}'
        results[var_key] = returns.rolling(window=window).quantile(1 - level)

    # Calculate Conditional VaR (Expected Shortfall)
    def rolling_cvar(x, level=0.95):
        var = np.percentile(x, (1 - level) * 100)
        return x[x <= var].mean() if any(x <= var) else var

    for level in [0.95, 0.99]:
        cvar_key = f'cvar_{int(level * 100)}'
        results[cvar_key] = returns.rolling(window=window).apply(
            lambda x: rolling_cvar(x, level=level), raw=True)

    # Calculate tail ratio (ratio of positive to negative tail)
    def tail_ratio(x, quantile=0.05):
        """Ratio of positive to negative tail means"""
        upper_tail = x > np.percentile(x, (1 - quantile) * 100)
        lower_tail = x < np.percentile(x, quantile * 100)

        upper_mean = x[upper_tail].mean() if any(upper_tail) else 0
        lower_mean = abs(x[lower_tail].mean()) if any(lower_tail) else 0

        return upper_mean / lower_mean if lower_mean != 0 else float('inf')

    results['tail_ratio'] = returns.rolling(window=window).apply(
        lambda x: tail_ratio(x), raw=True)

    # Calculate crash risk index (combination of skew, kurt, and tail ratio)
    # Low values indicate higher crash risk
    results['crash_risk_index'] = results['rolling_skew'] - (
            results['rolling_kurt'] - 3) / 10 + results['tail_ratio'] / 2

    return results


def detect_tail_risk_regimes(returns, window=252):
    """
    Detect tail risk regimes using extreme value theory.

    Parameters:
    -----------
    returns: pandas Series
        Return series
    window: int
        Rolling window for regime detection

    Returns:
    --------
    dict
        Dictionary with tail risk regime indicators
    """
    # Check for NaN values
    returns = returns.fillna(0)

    # Initialize results
    results = {}

    # Calculate tail risk metrics
    tail_metrics = calculate_tail_risk_metrics(returns, window=window)

    # Extract key metrics
    skew = tail_metrics['rolling_skew']
    kurt = tail_metrics['rolling_kurt']
    cvar_95 = tail_metrics['cvar_95']
    crash_risk = tail_metrics['crash_risk_index']

    # Calculate z-scores for regime detection
    skew_z = (skew - skew.rolling(window).mean()) / skew.rolling(window).std()
    kurt_z = (kurt - kurt.rolling(window).mean()) / kurt.rolling(window).std()
    cvar_z = (cvar_95 - cvar_95.rolling(window).mean()) / cvar_95.rolling(window).std()
    crash_z = (crash_risk - crash_risk.rolling(window).mean()) / crash_risk.rolling(window).std()

    # Detect extreme tail risk regime
    # Extreme negative skew, high kurtosis, low crash risk index
    extreme_tail_risk = (skew_z < -1.5) & (kurt_z > 1.5) & (crash_z < -1.5)

    # Detect high tail risk regime
    high_tail_risk = (skew_z < -1.0) & (kurt_z > 1.0) & (crash_z < -1.0)

    # Detect normal tail risk regime
    normal_tail_risk = ~high_tail_risk & ~extreme_tail_risk

    # Calculate tail risk score (0-100)
    # Higher score = higher tail risk
    tail_risk_score = (50 - skew_z * 10 + kurt_z * 5 - crash_z * 10).clip(0, 100)

    # Store results
    results['tail_risk_metrics'] = tail_metrics
    results['regime_indicators'] = {
        'extreme_tail_risk': extreme_tail_risk,
        'high_tail_risk': high_tail_risk,
        'normal_tail_risk': normal_tail_risk
    }
    results['tail_risk_score'] = tail_risk_score

    # Calculate regime durations
    def calc_regime_duration(regime_series):
        duration = pd.Series(0, index=regime_series.index)
        count = 0

        for i, val in enumerate(regime_series):
            if val:
                count += 1
            else:
                count = 0

            duration.iloc[i] = count

        return duration

    results['regime_durations'] = {
        'extreme_tail_risk': calc_regime_duration(extreme_tail_risk),
        'high_tail_risk': calc_regime_duration(high_tail_risk)
    }

    # Get current regime
    if extreme_tail_risk.iloc[-1]:
        current_regime = "Extreme Tail Risk"
    elif high_tail_risk.iloc[-1]:
        current_regime = "High Tail Risk"
    else:
        current_regime = "Normal Tail Risk"

    results['current_regime'] = current_regime
    results['current_tail_risk_score'] = tail_risk_score.iloc[-1]

    return results


def run_tail_risk_analysis(data, symbol, window=252):
    """
    Run comprehensive tail risk analysis.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    window: int
        Rolling window for analysis

    Returns:
    --------
    dict
        Dictionary with tail risk analysis results
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    # Get price data
    prices = data[symbol].copy()

    print(f"[INFO] Running tail risk analysis for {symbol}")

    # Calculate returns
    returns = prices.pct_change().dropna()

    if len(returns) < window:
        print(f"[WARNING] Not enough data for tail risk analysis: {len(returns)} < {window}")
        return None

    results = {}

    # 1. Fit GPD model
    try:
        gpd_model = fit_gpd_model(returns)
        results['gpd_model'] = gpd_model

        # Extract key parameters
        if gpd_model['negative_tail'] is not None:
            negative_shape = gpd_model['negative_tail']['shape']

            # Interpret shape parameter (xi)
            if negative_shape > 0.3:
                tail_type = "Very Heavy Tail"
                tail_desc = "Extreme events much more likely than normal distribution"
            elif negative_shape > 0.1:
                tail_type = "Heavy Tail"
                tail_desc = "Fat tails with significant outlier risk"
            elif negative_shape > -0.1:
                tail_type = "Moderate Tail"
                tail_desc = "Similar to exponential tail decay"
            else:
                tail_type = "Thin Tail"
                tail_desc = "Bounded tail with finite maximum loss"

            print(f"[INFO] Negative tail type: {tail_type} (shape={negative_shape:.3f})")
            print(f"[INFO] {tail_desc}")

            results['tail_type'] = tail_type
            results['tail_description'] = tail_desc
    except Exception as e:
        print(f"[WARNING] Error fitting GPD model: {e}")

    # 2. Calculate Expected Shortfall
    try:
        es_empirical = calculate_expected_shortfall(returns, method='empirical')

        results['expected_shortfall'] = es_empirical

        print(f"[INFO] 95% Expected Shortfall: {es_empirical['es_95']:.3%}")
        print(f"[INFO] 99% Expected Shortfall: {es_empirical['es_99']:.3%}")
    except Exception as e:
        print(f"[WARNING] Error calculating Expected Shortfall: {e}")

    # 3. Calculate tail risk metrics
    try:
        tail_metrics = calculate_tail_risk_metrics(returns, window=window)

        # Extract latest values
        latest_metrics = {key: val.iloc[-1] for key, val in tail_metrics.items()}

        results['tail_metrics'] = latest_metrics
        results['tail_metrics_timeseries'] = tail_metrics

        print(f"[INFO] Latest skewness: {latest_metrics['rolling_skew']:.3f}")
        print(f"[INFO] Latest kurtosis: {latest_metrics['rolling_kurt']:.3f}")
        print(f"[INFO] Latest 95% CVaR: {latest_metrics['cvar_95']:.3%}")
    except Exception as e:
        print(f"[WARNING] Error calculating tail risk metrics: {e}")

    # 4. Detect tail risk regimes
    try:
        risk_regimes = detect_tail_risk_regimes(returns, window=window)

        results['risk_regimes'] = risk_regimes

        print(f"[INFO] Current tail risk regime: {risk_regimes['current_regime']}")
        print(f"[INFO] Current tail risk score: {risk_regimes['current_tail_risk_score']:.2f}")
    except Exception as e:
        print(f"[WARNING] Error detecting tail risk regimes: {e}")

    # 5. Calculate early warning indicators

    # 5.1 Volatility of volatility
    returns_std = returns.rolling(window=30).std()
    vol_of_vol = returns_std.rolling(window=window).std() / returns_std.rolling(window=window).mean()

    # 5.2 Jump intensity (frequency of large returns)
    jump_threshold = 3 * returns.rolling(window=window).std()
    jump_intensity = returns.abs().rolling(window=window).apply(
        lambda x: np.sum(x > jump_threshold.loc[x.index[0]])) / window

    # 5.3 Tail correlation with market (if available)
    # (Placeholder - in practice, you would use market index returns)

    # 5.4 Combine early warning indicators
    early_warning = 0.5 * vol_of_vol + 0.5 * jump_intensity
    early_warning_z = (early_warning - early_warning.rolling(window).mean()) / early_warning.rolling(window).std()

    # Store early warning indicators
    results['early_warning'] = {
        'vol_of_vol': vol_of_vol.iloc[-1],
        'jump_intensity': jump_intensity.iloc[-1],
        'early_warning_indicator': early_warning.iloc[-1],
        'early_warning_z': early_warning_z.iloc[-1]
    }

    # Interpret early warning level
    if early_warning_z.iloc[-1] > 2:
        early_warning_level = "Critical"
    elif early_warning_z.iloc[-1] > 1:
        early_warning_level = "High"
    elif early_warning_z.iloc[-1] > 0:
        early_warning_level = "Elevated"
    else:
        early_warning_level = "Normal"

    results['early_warning']['level'] = early_warning_level

    print(f"[INFO] Early warning level: {early_warning_level}")

    return results


#########################
# 9. Wavelet Analysis for Time-Frequency Decomposition
#########################

def calculate_wavelet_transform(time_series, wavelet='morl', scales=None, sampling_period=1):
    """
    Apply continuous wavelet transform to identify dominant frequencies.

    Parameters:
    -----------
    time_series: pandas Series or numpy array
        Input time series data
    wavelet: str
        Wavelet type ('morl', 'mexh', 'cmor', etc.)
    scales: numpy array, optional
        Scales for wavelet transform
    sampling_period: float
        Sampling period of time series

    Returns:
    --------
    dict
        Dictionary with wavelet transform results
    """
    if isinstance(time_series, pd.Series):
        time_series_values = time_series.values
        dates = time_series.index
    else:
        time_series_values = time_series
        dates = np.arange(len(time_series))

    # Check for NaN values
    if np.isnan(time_series_values).any():
        time_series_values = pd.Series(time_series_values).fillna(method='ffill').values

    # Generate scales if not provided
    if scales is None:
        scales = np.arange(1, min(128, len(time_series_values) // 2))

    # Apply continuous wavelet transform
    [coefficients, frequencies] = pywt.cwt(time_series_values, scales, wavelet, sampling_period)

    # Calculate power spectrum
    power = (np.abs(coefficients)) ** 2

    # Calculate global wavelet spectrum (average power over time)
    global_ws = power.mean(axis=1)

    # Find dominant frequencies
    dom_freq_idx = np.argmax(global_ws)
    dom_freq = frequencies[dom_freq_idx]
    dom_period = 1 / dom_freq if dom_freq != 0 else float('inf')

    # Calculate energy distribution across frequencies
    energy_dist = global_ws / global_ws.sum()

    # Return wavelet transform results
    results = {
        'coefficients': coefficients,
        'frequencies': frequencies,
        'power': power,
        'global_wavelet_spectrum': global_ws,
        'dominant_frequency': dom_freq,
        'dominant_period': dom_period,
        'energy_distribution': energy_dist,
        'scales': scales
    }

    return results


def calculate_cross_wavelet_transform(x, y, wavelet='morl', scales=None, sampling_period=1):
    """
    Apply cross-wavelet transform to detect leading/lagging relationships.

    Parameters:
    -----------
    x: pandas Series or numpy array
        First time series
    y: pandas Series or numpy array
        Second time series
    wavelet: str
        Wavelet type ('morl', 'mexh', 'cmor', etc.)
    scales: numpy array, optional
        Scales for wavelet transform
    sampling_period: float
        Sampling period of time series

    Returns:
    --------
    dict
        Dictionary with cross-wavelet transform results
    """
    # Check input type
    if isinstance(x, pd.Series):
        x_values = x.values
    else:
        x_values = x

    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y

    # Ensure same length
    min_len = min(len(x_values), len(y_values))
    x_values = x_values[:min_len]
    y_values = y_values[:min_len]

    # Check for NaN values
    if np.isnan(x_values).any():
        x_values = pd.Series(x_values).fillna(method='ffill').values

    if np.isnan(y_values).any():
        y_values = pd.Series(y_values).fillna(method='ffill').values

    # Generate scales if not provided
    if scales is None:
        scales = np.arange(1, min(64, min_len // 4))

    # Calculate wavelet transforms of x and y
    [coeffs_x, frequencies] = pywt.cwt(x_values, scales, wavelet, sampling_period)
    [coeffs_y, _] = pywt.cwt(y_values, scales, wavelet, sampling_period)

    # Cross-wavelet transform: complex conjugate of X * Y
    cross_wavelet = coeffs_x * np.conj(coeffs_y)

    # Cross-wavelet power
    cross_power = np.abs(cross_wavelet) ** 2

    # Phase difference
    phase_diff = np.angle(cross_wavelet)

    # Mean phase difference across time
    mean_phase_diff = np.nanmean(phase_diff, axis=1)

    # Interpret phase difference
    # If mean_phase_diff is positive, X leads Y
    # If mean_phase_diff is negative, Y leads X
    lead_lag = np.zeros_like(mean_phase_diff)
    lead_lag[mean_phase_diff > 0] = 1  # X leads Y
    lead_lag[mean_phase_diff < 0] = -1  # Y leads X

    # Calculate global cross-wavelet power
    global_cross_power = cross_power.mean(axis=1)

    # Find dominant relationship (scale with highest cross power)
    dom_scale_idx = np.argmax(global_cross_power)
    dom_scale = scales[dom_scale_idx]
    dom_freq = frequencies[dom_scale_idx]
    dom_period = 1 / dom_freq if dom_freq != 0 else float('inf')
    dom_phase = mean_phase_diff[dom_scale_idx]

    # Calculate lead/lag in time units
    # Phase difference in radians / (2*pi*frequency)
    time_lag = dom_phase / (2 * np.pi * dom_freq) if dom_freq != 0 else 0

    # Return cross-wavelet transform results
    results = {
        'cross_wavelet': cross_wavelet,
        'cross_power': cross_power,
        'phase_diff': phase_diff,
        'mean_phase_diff': mean_phase_diff,
        'lead_lag': lead_lag,
        'global_cross_power': global_cross_power,
        'dominant_scale': dom_scale,
        'dominant_frequency': dom_freq,
        'dominant_period': dom_period,
        'dominant_phase': dom_phase,
        'time_lag': time_lag,
        'frequencies': frequencies,
        'scales': scales
    }

    return results


def calculate_wavelet_coherence(x, y, wavelet='morl', scales=None, sampling_period=1):
    """
    Calculate wavelet coherence to measure correlation across timeframes.

    Parameters:
    -----------
    x: pandas Series or numpy array
        First time series
    y: pandas Series or numpy array
        Second time series
    wavelet: str
        Wavelet type ('morl', 'mexh', 'cmor', etc.)
    scales: numpy array, optional
        Scales for wavelet transform
    sampling_period: float
        Sampling period of time series

    Returns:
    --------
    dict
        Dictionary with wavelet coherence results
    """
    # Check input type
    if isinstance(x, pd.Series):
        x_values = x.values
    else:
        x_values = x

    if isinstance(y, pd.Series):
        y_values = y.values
    else:
        y_values = y

    # Ensure same length
    min_len = min(len(x_values), len(y_values))
    x_values = x_values[:min_len]
    y_values = y_values[:min_len]

    # Check for NaN values
    if np.isnan(x_values).any():
        x_values = pd.Series(x_values).fillna(method='ffill').values

    if np.isnan(y_values).any():
        y_values = pd.Series(y_values).fillna(method='ffill').values

    # Generate scales if not provided
    if scales is None:
        scales = np.arange(1, min(64, min_len // 4))

    # Calculate wavelet transforms of x and y
    [coeffs_x, frequencies] = pywt.cwt(x_values, scales, wavelet, sampling_period)
    [coeffs_y, _] = pywt.cwt(y_values, scales, wavelet, sampling_period)

    # Cross-wavelet transform
    cross_wavelet = coeffs_x * np.conj(coeffs_y)

    # Auto-wavelet powers
    power_x = np.abs(coeffs_x) ** 2
    power_y = np.abs(coeffs_y) ** 2

    # Smooth the powers (using Gaussian filter along both dimensions)
    smoothed_power_x = np.zeros_like(power_x)
    smoothed_power_y = np.zeros_like(power_y)
    smoothed_cross = np.zeros_like(cross_wavelet, dtype=complex)

    for i in range(power_x.shape[0]):
        smoothed_power_x[i, :] = gaussian_filter1d(power_x[i, :], sigma=2)
        smoothed_power_y[i, :] = gaussian_filter1d(power_y[i, :], sigma=2)
        smoothed_cross[i, :] = gaussian_filter1d(cross_wavelet[i, :].real, sigma=2) + 1j * gaussian_filter1d(
            cross_wavelet[i, :].imag, sigma=2)

    # Calculate wavelet coherence
    coherence = np.abs(smoothed_cross) ** 2 / (smoothed_power_x * smoothed_power_y)

    # Phase difference
    phase_diff = np.angle(smoothed_cross)

    # Average coherence across time for each scale
    mean_coherence = np.mean(coherence, axis=1)

    # Find scale with highest coherence
    max_coherence_idx = np.argmax(mean_coherence)
    max_coherence_scale = scales[max_coherence_idx]
    max_coherence_freq = frequencies[max_coherence_idx]
    max_coherence_period = 1 / max_coherence_freq if max_coherence_freq != 0 else float('inf')

    # Return wavelet coherence results
    results = {
        'coherence': coherence,
        'phase_diff': phase_diff,
        'mean_coherence': mean_coherence,
        'max_coherence_scale': max_coherence_scale,
        'max_coherence_frequency': max_coherence_freq,
        'max_coherence_period': max_coherence_period,
        'frequencies': frequencies,
        'scales': scales
    }

    return results


def run_wavelet_analysis(data, symbol, reference_symbol=None, max_scale=None):
    """
    Run comprehensive wavelet analysis for a time series.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    reference_symbol: str, optional
        Reference symbol for cross-wavelet analysis
    max_scale: int, optional
        Maximum scale for wavelet transform

    Returns:
    --------
    dict
        Dictionary with wavelet analysis results
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    # Get price data
    prices = data[symbol].copy()

    print(f"[INFO] Running wavelet analysis for {symbol}")

    if len(prices) < 64:
        print(f"[WARNING] Not enough data for wavelet analysis: {len(prices)} < 64")
        return None

    # Calculate returns
    returns = prices.pct_change().fillna(0)

    # Set maximum scale
    if max_scale is None:
        max_scale = min(64, len(returns) // 4)

    # Generate scales (power of 2 for efficiency)
    scales = np.arange(1, max_scale)

    results = {}

    # 1. Calculate wavelet transform
    try:
        cwt_results = calculate_wavelet_transform(returns, scales=scales)
        results['wavelet_transform'] = cwt_results

        # Extract key insights
        dom_period = cwt_results['dominant_period']
        dom_freq = cwt_results['dominant_frequency']

        print(f"[INFO] Dominant cycle: {dom_period:.2f} days (frequency: {dom_freq:.6f})")
    except Exception as e:
        print(f"[WARNING] Error calculating wavelet transform: {e}")

    # 2. Calculate cross-wavelet transform if reference symbol provided
    if reference_symbol is not None and reference_symbol in data.columns:
        try:
            ref_prices = data[reference_symbol].copy()
            ref_returns = ref_prices.pct_change().fillna(0)

            # Ensure same length
            min_len = min(len(returns), len(ref_returns))
            returns_aligned = returns.iloc[-min_len:]
            ref_returns_aligned = ref_returns.iloc[-min_len:]

            xwt_results = calculate_cross_wavelet_transform(
                returns_aligned, ref_returns_aligned, scales=scales)

            results['cross_wavelet_transform'] = xwt_results

            # Extract key insights
            time_lag = xwt_results['time_lag']
            dom_period = xwt_results['dominant_period']

            lead_lag = "leads" if time_lag > 0 else "lags" if time_lag < 0 else "coincident with"

            print(
                f"[INFO] {symbol} {lead_lag} {reference_symbol} by {abs(time_lag):.2f} days at cycle of {dom_period:.2f} days")

            # Calculate wavelet coherence
            wct_results = calculate_wavelet_coherence(
                returns_aligned, ref_returns_aligned, scales=scales)

            results['wavelet_coherence'] = wct_results

            # Extract key insights
            max_coherence = wct_results['mean_coherence'].max()
            max_coherence_period = wct_results['max_coherence_period']

            print(f"[INFO] Maximum coherence: {max_coherence:.3f} at period of {max_coherence_period:.2f} days")
        except Exception as e:
            print(f"[WARNING] Error calculating cross-wavelet analysis: {e}")

    # 3. Identify market regimes based on wavelet energy
    try:
        # Get power spectrum
        power = results['wavelet_transform']['power']
        frequencies = results['wavelet_transform']['frequencies']

        # Split frequencies into bands
        # High frequency (short-term): >1/10 cycles per day
        # Medium frequency (medium-term): 1/10 to 1/30 cycles per day
        # Low frequency (long-term): <1/30 cycles per day
        high_freq_mask = frequencies > 1 / 10
        medium_freq_mask = (frequencies <= 1 / 10) & (frequencies > 1 / 30)
        low_freq_mask = frequencies <= 1 / 30

        # Calculate energy in each band over time
        high_freq_energy = power[high_freq_mask, :].sum(axis=0) if any(high_freq_mask) else np.zeros(power.shape[1])
        medium_freq_energy = power[medium_freq_mask, :].sum(axis=0) if any(medium_freq_mask) else np.zeros(
            power.shape[1])
        low_freq_energy = power[low_freq_mask, :].sum(axis=0) if any(low_freq_mask) else np.zeros(power.shape[1])

        # Calculate total energy
        total_energy = high_freq_energy + medium_freq_energy + low_freq_energy
        total_energy = np.where(total_energy == 0, 1, total_energy)  # Avoid division by zero

        # Calculate energy ratios
        high_freq_ratio = high_freq_energy / total_energy
        medium_freq_ratio = medium_freq_energy / total_energy
        low_freq_ratio = low_freq_energy / total_energy

        # Identify dominant regime
        dominant_regime = np.zeros(power.shape[1])

        for i in range(power.shape[1]):
            if high_freq_ratio[i] > 0.5:
                dominant_regime[i] = 1  # High frequency dominated
            elif medium_freq_ratio[i] > 0.5:
                dominant_regime[i] = 2  # Medium frequency dominated
            elif low_freq_ratio[i] > 0.5:
                dominant_regime[i] = 3  # Low frequency dominated
            else:
                dominant_regime[i] = 0  # Mixed regime

        # Create a DataFrame with energy ratios
        energy_df = pd.DataFrame({
            'high_freq_ratio': high_freq_ratio,
            'medium_freq_ratio': medium_freq_ratio,
            'low_freq_ratio': low_freq_ratio,
            'dominant_regime': dominant_regime
        }, index=returns.index[-len(dominant_regime):])

        # Add regime labels
        regime_labels = {
            0: "Mixed",
            1: "Short-term",
            2: "Medium-term",
            3: "Long-term"
        }

        energy_df['regime_label'] = energy_df['dominant_regime'].map(regime_labels)

        results['energy_ratios'] = energy_df

        # Get current regime
        current_regime = regime_labels[int(dominant_regime[-1])]
        current_high = high_freq_ratio[-1]
        current_medium = medium_freq_ratio[-1]
        current_low = low_freq_ratio[-1]

        print(f"[INFO] Current regime: {current_regime}")
        print(
            f"[INFO] Current energy distribution - Short-term: {current_high:.2f}, Medium-term: {current_medium:.2f}, Long-term: {current_low:.2f}")
    except Exception as e:
        print(f"[WARNING] Error calculating wavelet energy regimes: {e}")

    return results


#########################
# 10. Bayesian Regime Detection and Adaptation
#########################

def detect_changepoints_bayesian(returns, model='online', max_points=1000):
    """
    Implement Bayesian changepoint detection to identify market regime shifts.

    Parameters:
    -----------
    returns: pandas Series
        Return series
    model: str
        Model type ('offline', 'online')
    max_points: int
        Maximum number of points to use (for computational efficiency)

    Returns:
    --------
    dict
        Dictionary with changepoint detection results
    """
    # Check for NaN values
    returns = returns.fillna(0)

    # For computational efficiency, use a subset if the series is too long
    if len(returns) > max_points:
        # Use the most recent points
        returns_subset = returns.iloc[-max_points:]
    else:
        returns_subset = returns.copy()

    dates = returns_subset.index
    values = returns_subset.values

    # Initialize results dictionary
    results = {}

    print(f"[INFO] Running Bayesian changepoint detection on {len(values)} points")

    # Define offline changepoint detection model
    def offline_changepoint_detection():
        try:
            with pm.Model() as model:
                # Number of potential changepoints - more for longer series
                n_changepoints = min(20, len(values) // 50)

                # Prior on changepoint locations - uniform across time
                changepoint_loc = pm.DiscreteUniform(
                    'changepoint_loc',
                    lower=10,
                    upper=len(values) - 10,
                    shape=n_changepoints
                )

                # Sort the changepoints
                changepoint_loc_sorted = tt.sort(changepoint_loc)

                # Prior on segment means
                segment_mean = pm.Normal('segment_mean', mu=0, sd=0.01, shape=n_changepoints + 1)

                # Prior on segment standard deviations (half-normal to ensure positive)
                segment_std = pm.HalfNormal('segment_std', sd=0.01, shape=n_changepoints + 1)

                # Create index variable for segment assignment
                idx = tt.arange(len(values))

                # Function to assign data points to segments
                def get_segment_idx(idx, changepoint_loc_sorted):
                    segment_idx = tt.zeros_like(idx)

                    for i in range(n_changepoints):
                        segment_idx = tt.switch(
                            idx >= changepoint_loc_sorted[i],
                            i + 1,
                            segment_idx
                        )

                    return segment_idx

                # Get segment index for each data point
                segment_idx = get_segment_idx(idx, changepoint_loc_sorted)

                # Get mean and standard deviation for each data point
                idx_means = segment_mean[segment_idx]
                idx_stds = segment_std[segment_idx]

                # Likelihood
                returns_obs = pm.Normal(
                    'returns_obs',
                    mu=idx_means,
                    sd=idx_stds,
                    observed=values
                )

                # Inference
                trace = pm.sample(
                    1000,
                    tune=500,
                    chains=2,
                    cores=1,
                    return_inferencedata=False
                )

            # Extract results
            changepoint_samples = trace['changepoint_loc_sorted']

            # Calculate changepoint probabilities
            changepoint_probs = np.zeros(len(values))

            for sample in changepoint_samples:
                for cp in sample:
                    if 0 <= cp < len(changepoint_probs):
                        changepoint_probs[cp] += 1

            # Normalize
            changepoint_probs = changepoint_probs / len(changepoint_samples)

            # Detect significant changepoints (probability > threshold)
            threshold = 0.2
            significant_cps = np.where(changepoint_probs > threshold)[0]

            # Get segment means
            segment_means = trace['segment_mean'].mean(axis=0)

            # Get dates for significant changepoints
            if len(significant_cps) > 0:
                cp_dates = [dates[cp] for cp in significant_cps if cp < len(dates)]
            else:
                cp_dates = []

            return {
                'changepoint_probabilities': changepoint_probs,
                'significant_changepoints': significant_cps,
                'changepoint_dates': cp_dates,
                'segment_means': segment_means,
                'trace': trace
            }
        except Exception as e:
            print(f"[WARNING] Error in offline changepoint detection: {e}")
            return {
                'changepoint_probabilities': np.zeros(len(values)),
                'significant_changepoints': [],
                'changepoint_dates': [],
                'segment_means': [],
                'trace': None
            }

    # Define online changepoint detection using Bayesian Online Changepoint Detection (BOCD)
    def online_changepoint_detection():
        try:
            # Simplified BOCD implementation
            # Parameters
            hazard = 1 / 200  # Expected run length (1/hazard)
            mu0 = 0  # Prior mean
            kappa0 = 1  # Prior precision
            alpha0 = 1  # Prior shape
            beta0 = 1  # Prior rate

            # Initialize variables
            R = np.zeros((len(values), len(values)))  # Run length distribution
            R[0, 0] = 1  # Start with run length 0

            mu = np.zeros((len(values), len(values)))
            mu[0, 0] = mu0

            kappa = np.zeros((len(values), len(values)))
            kappa[0, 0] = kappa0

            alpha = np.zeros((len(values), len(values)))
            alpha[0, 0] = alpha0

            beta = np.zeros((len(values), len(values)))
            beta[0, 0] = beta0

            # Run BOCD algorithm
            for t in range(1, len(values)):
                # Predictive probabilities
                pred_probs = np.zeros(t)

                for i in range(t):
                    # Calculate predictive probability
                    alpha_i = alpha[t - 1, i]
                    beta_i = beta[t - 1, i]
                    kappa_i = kappa[t - 1, i]
                    mu_i = mu[t - 1, i]

                    # Student-t distribution parameters
                    nu = 2 * alpha_i
                    sigma2 = beta_i * (kappa_i + 1) / (alpha_i * kappa_i)

                    # Calculate predictive probability using Student-t distribution
                    x = values[t]
                    pred_probs[i] = stats.t.pdf(x, df=nu, loc=mu_i, scale=np.sqrt(sigma2))

                # Calculate growth probabilities
                growth_probs = pred_probs * R[t - 1, :t] * (1 - hazard)

                # Calculate changepoint probability
                cp_prob = np.sum(pred_probs * R[t - 1, :t] * hazard)

                # Update run length distribution
                R[t, 1:t + 1] = growth_probs
                R[t, 0] = cp_prob

                # Normalize
                R[t, :t + 1] = R[t, :t + 1] / np.sum(R[t, :t + 1])

                # Update posterior parameters
                for i in range(t + 1):
                    if i == 0:  # Changepoint
                        mu[t, i] = mu0
                        kappa[t, i] = kappa0
                        alpha[t, i] = alpha0
                        beta[t, i] = beta0
                    else:  # No changepoint
                        muT = mu[t - 1, i - 1]
                        kappaT = kappa[t - 1, i - 1]
                        alphaT = alpha[t - 1, i - 1]
                        betaT = beta[t - 1, i - 1]

                        # Update parameters
                        kappa[t, i] = kappaT + 1
                        mu[t, i] = (kappaT * muT + values[t]) / kappa[t, i]
                        alpha[t, i] = alphaT + 0.5
                        beta[t, i] = betaT + 0.5 * kappaT * (values[t] - muT) ** 2 / kappa[t, i]

            # Extract changepoint probabilities
            changepoint_probs = R[:, 0]

            # Detect significant changepoints (probability > threshold)
            threshold = 0.2
            significant_cps = np.where(changepoint_probs > threshold)[0]

            # Get most likely run length at each time
            max_run_lengths = np.argmax(R, axis=1)

            # Get dates for significant changepoints
            if len(significant_cps) > 0:
                cp_dates = [dates[cp] for cp in significant_cps if cp < len(dates)]
            else:
                cp_dates = []

            return {
                'changepoint_probabilities': changepoint_probs,
                'significant_changepoints': significant_cps,
                'changepoint_dates': cp_dates,
                'max_run_lengths': max_run_lengths,
                'run_length_dist': R
            }
        except Exception as e:
            print(f"[WARNING] Error in online changepoint detection: {e}")
            return {
                'changepoint_probabilities': np.zeros(len(values)),
                'significant_changepoints': [],
                'changepoint_dates': [],
                'max_run_lengths': np.zeros(len(values)),
                'run_length_dist': np.zeros((len(values), len(values)))
            }

    # Run selected model
    if model == 'offline':
        cp_results = offline_changepoint_detection()
    else:  # model == 'online'
        cp_results = online_changepoint_detection()

    # Extract and store results
    results['changepoint_detection'] = cp_results

    # Create a DataFrame with changepoint probabilities
    cp_df = pd.DataFrame({
        'changepoint_probability': cp_results['changepoint_probabilities']
    }, index=dates)

    results['changepoint_df'] = cp_df

    # Extract significant changepoints
    significant_cps = cp_results['significant_changepoints']

    if len(significant_cps) > 0:
        print(f"[INFO] Detected {len(significant_cps)} significant regime changes")

        # Get the most recent changepoint
        most_recent_cp = significant_cps[-1]

        if most_recent_cp < len(dates):
            most_recent_date = dates[most_recent_cp]
            days_since_change = (dates[-1] - most_recent_date).days

            print(
                f"[INFO] Most recent regime change: {most_recent_date.strftime('%Y-%m-%d')} ({days_since_change} days ago)")

            results['most_recent_changepoint'] = {
                'index': most_recent_cp,
                'date': most_recent_date,
                'days_since_change': days_since_change
            }
        else:
            print("[WARNING] Most recent changepoint index out of bounds")
    else:
        print("[INFO] No significant regime changes detected")

    return results


def estimate_regime_parameters(returns, changepoints=None, min_regime_length=30):
    """
    Estimate parameters for each regime.

    Parameters:
    -----------
    returns: pandas Series
        Return series
    changepoints: list, optional
        List of changepoint indices
    min_regime_length: int
        Minimum regime length

    Returns:
    --------
    dict
        Dictionary with regime parameters
    """
    if changepoints is None or len(changepoints) == 0:
        # If no changepoints provided, treat the entire series as one regime
        regimes = [{'start': 0, 'end': len(returns) - 1}]
    else:
        # Create regimes from changepoints
        regimes = []

        # First regime: from start to first changepoint
        regimes.append({'start': 0, 'end': changepoints[0] - 1})

        # Middle regimes
        for i in range(len(changepoints) - 1):
            regimes.append({'start': changepoints[i], 'end': changepoints[i + 1] - 1})

        # Last regime: from last changepoint to end
        regimes.append({'start': changepoints[-1], 'end': len(returns) - 1})

    # Estimate parameters for each regime
    regime_params = []

    for i, regime in enumerate(regimes):
        start, end = regime['start'], regime['end']

        # Skip regimes that are too short
        if end - start + 1 < min_regime_length:
            continue

        # Extract regime data
        regime_returns = returns.iloc[start:end + 1]

        # Calculate parameters
        mean = regime_returns.mean()
        std = regime_returns.std()
        sharpe = mean / std if std > 0 else 0
        skew = stats.skew(regime_returns)
        kurt = stats.kurtosis(regime_returns)

        # Calculate VaR and CVaR
        var_95 = np.percentile(regime_returns, 5)
        cvar_95 = regime_returns[regime_returns <= var_95].mean()

        # Calculate autocorrelation
        acf_1 = regime_returns.autocorr(lag=1)

        # Store parameters
        regime_params.append({
            'regime': i + 1,
            'start_idx': start,
            'end_idx': end,
            'start_date': returns.index[start],
            'end_date': returns.index[end],
            'length': end - start + 1,
            'mean': mean,
            'std': std,
            'sharpe': sharpe,
            'skew': skew,
            'kurt': kurt,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'acf_1': acf_1
        })

    return regime_params


def classify_market_regimes(regime_params):
    """
    Classify market regimes based on their parameters.

    Parameters:
    -----------
    regime_params: list
        List of regime parameters

    Returns:
    --------
    list
        List of regime classifications
    """
    regime_classifications = []

    for regime in regime_params:
        # Extract parameters
        mean = regime['mean']
        std = regime['std']
        sharpe = regime['sharpe']
        skew = regime['skew']
        kurt = regime['kurt']

        # Classify based on return and volatility
        if mean > 0.001 and std < 0.01:
            regime_type = "Low Volatility Bull"
        elif mean > 0.001 and std >= 0.01:
            regime_type = "High Volatility Bull"
        elif mean < -0.001 and std < 0.01:
            regime_type = "Low Volatility Bear"
        elif mean < -0.001 and std >= 0.01:
            regime_type = "High Volatility Bear"
        else:
            regime_type = "Sideways/Neutral"

        # Add sub-classification based on higher moments
        if skew < -0.5 and kurt > 3:
            sub_type = "Crash Risk"
        elif skew > 0.5 and kurt > 3:
            sub_type = "Rally Potential"
        elif abs(skew) <= 0.5 and kurt <= 3:
            sub_type = "Normal Distribution"
        else:
            sub_type = "Mixed"

        # Add momentum classification based on autocorrelation
        acf_1 = regime['acf_1']

        if acf_1 > 0.2:
            momentum = "Strong Trend"
        elif acf_1 > 0:
            momentum = "Weak Trend"
        elif acf_1 > -0.2:
            momentum = "Weak Mean Reversion"
        else:
            momentum = "Strong Mean Reversion"

        # Store classification
        regime_classifications.append({
            'regime': regime['regime'],
            'start_date': regime['start_date'],
            'end_date': regime['end_date'],
            'regime_type': regime_type,
            'sub_type': sub_type,
            'momentum': momentum,
            'parameters': regime
        })

    return regime_classifications


def create_regime_adaptive_model(returns, regime_classifications, current_regime):
    """
    Create dynamic model weightings based on regime probability.

    Parameters:
    -----------
    returns: pandas Series
        Return series
    regime_classifications: list
        List of regime classifications
    current_regime: dict
        Current regime information

    Returns:
    --------
    dict
        Dictionary with adaptive model parameters
    """
    # Extract current regime type
    current_type = current_regime['regime_type']
    current_sub_type = current_regime['sub_type']
    current_momentum = current_regime['momentum']

    # Define model weights for different regime types
    model_weights = {
        # Format: regime_type -> model_weights
        "Low Volatility Bull": {
            "momentum": 0.5,
            "mean_reversion": 0.2,
            "fundamental": 0.2,
            "machine_learning": 0.1
        },
        "High Volatility Bull": {
            "momentum": 0.3,
            "mean_reversion": 0.3,
            "fundamental": 0.1,
            "machine_learning": 0.3
        },
        "Low Volatility Bear": {
            "momentum": 0.4,
            "mean_reversion": 0.3,
            "fundamental": 0.1,
            "machine_learning": 0.2
        },
        "High Volatility Bear": {
            "momentum": 0.2,
            "mean_reversion": 0.4,
            "fundamental": 0.1,
            "machine_learning": 0.3
        },
        "Sideways/Neutral": {
            "momentum": 0.2,
            "mean_reversion": 0.5,
            "fundamental": 0.2,
            "machine_learning": 0.1
        }
    }

    # Get base weights for current regime type
    if current_type in model_weights:
        weights = model_weights[current_type].copy()
    else:
        # Default weights if regime type not found
        weights = {
            "momentum": 0.25,
            "mean_reversion": 0.25,
            "fundamental": 0.25,
            "machine_learning": 0.25
        }

    # Adjust weights based on sub-type
    if current_sub_type == "Crash Risk":
        # Increase mean reversion and ML for crash risk
        weights["mean_reversion"] += 0.1
        weights["machine_learning"] += 0.1
        weights["momentum"] -= 0.2
    elif current_sub_type == "Rally Potential":
        # Increase momentum for rally potential
        weights["momentum"] += 0.1
        weights["machine_learning"] += 0.1
        weights["mean_reversion"] -= 0.2

    # Adjust weights based on momentum
    if current_momentum == "Strong Trend":
        weights["momentum"] += 0.1
        weights["mean_reversion"] -= 0.1
    elif current_momentum == "Strong Mean Reversion":
        weights["momentum"] -= 0.1
        weights["mean_reversion"] += 0.1

    # Normalize weights to sum to 1
    total = sum(weights.values())
    for k in weights:
        weights[k] = weights[k] / total

    # Define hyperparameters for each model type
    hyperparams = {
        "momentum": {
            "lookback_period": 20 if "Bull" in current_type else 10,
            "smoothing_factor": 0.1 if "High Volatility" in current_type else 0.2
        },
        "mean_reversion": {
            "zscore_threshold": 2.0 if "High Volatility" in current_type else 1.5,
            "half_life": 10 if "Strong Mean Reversion" in current_momentum else 20
        },
        "fundamental": {
            "value_weight": 0.7 if "Bear" in current_type else 0.3,
            "growth_weight": 0.3 if "Bear" in current_type else 0.7
        },
        "machine_learning": {
            "ensemble_weights": {
                "xgboost": 0.4 if "High Volatility" in current_type else 0.3,
                "random_forest": 0.3,
                "neural_network": 0.3 if "High Volatility" in current_type else 0.4
            }
        }
    }

    # Store adaptive model parameters
    adaptive_model = {
        'weights': weights,
        'hyperparams': hyperparams,
        'current_regime': current_regime
    }

    return adaptive_model


def run_bayesian_regime_analysis(data, symbol, model='online', window=252):
    """
    Run Bayesian regime detection and create adaptive models.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    model: str
        Model type for changepoint detection ('offline', 'online')
    window: int
        Window size for analysis

    Returns:
    --------
    dict
        Dictionary with Bayesian regime analysis results
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    # Get price data
    prices = data[symbol].copy()

    print(f"[INFO] Running Bayesian regime analysis for {symbol}")

    # Calculate returns
    returns = prices.pct_change().dropna()

    if len(returns) < window:
        print(f"[WARNING] Not enough data for Bayesian regime analysis: {len(returns)} < {window}")
        return None

    # 1. Detect changepoints
    try:
        cp_results = detect_changepoints_bayesian(returns, model=model)

        if cp_results is None:
            print("[WARNING] Changepoint detection failed")
            return None
    except Exception as e:
        print(f"[WARNING] Error in changepoint detection: {e}")
        return None

    # 2. Estimate regime parameters
    try:
        # Get significant changepoints
        significant_cps = cp_results['changepoint_detection']['significant_changepoints']

        # Estimate parameters for each regime
        regime_params = estimate_regime_parameters(returns, significant_cps)

        print(f"[INFO] Estimated parameters for {len(regime_params)} regimes")
    except Exception as e:
        print(f"[WARNING] Error estimating regime parameters: {e}")
        return None

    # 3. Classify regimes
    try:
        regime_classifications = classify_market_regimes(regime_params)

        # Get current regime
        current_regime = regime_classifications[-1]

        print(
            f"[INFO] Current regime: {current_regime['regime_type']} - {current_regime['sub_type']} ({current_regime['momentum']})")
    except Exception as e:
        print(f"[WARNING] Error classifying regimes: {e}")
        return None

    # 4. Create adaptive model
    try:
        adaptive_model = create_regime_adaptive_model(returns, regime_classifications, current_regime)

        # Print model weights
        weights = adaptive_model['weights']
        print(f"[INFO] Adaptive model weights:")
        for model_type, weight in weights.items():
            print(f"  - {model_type}: {weight:.2f}")
    except Exception as e:
        print(f"[WARNING] Error creating adaptive model: {e}")
        return None

    # Compile results
    results = {
        'changepoint_detection': cp_results,
        'regime_params': regime_params,
        'regime_classifications': regime_classifications,
        'current_regime': current_regime,
        'adaptive_model': adaptive_model
    }

    return results


#########################
# 11. Advanced Risk Factor Decomposition
#########################

def perform_risk_factor_pca(returns, min_history=252):
    """
    Implement principal component analysis on returns to identify risk factors.

    Parameters:
    -----------
    returns: pandas DataFrame
        DataFrame containing return series for multiple assets
    min_history: int
        Minimum history required for analysis

    Returns:
    --------
    dict
        Dictionary with PCA results
    """
    # Remove columns with too many NaN values
    valid_cols = []

    for col in returns.columns:
        if returns[col].count() >= min_history:
            valid_cols.append(col)

    if len(valid_cols) < 5:
        print(f"[WARNING] Not enough valid columns for PCA: {len(valid_cols)} < 5")
        return None

    # Use valid columns
    returns_valid = returns[valid_cols].copy()

    # Fill NaN values (simple forward fill then backward fill)
    returns_filled = returns_valid.fillna(method='ffill').fillna(method='bfill')

    # Standardize returns
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_filled)

    # Apply PCA
    n_components = min(10, len(valid_cols))
    pca = PCA(n_components=n_components)
    pca_results = pca.fit_transform(returns_scaled)

    # Create DataFrame with PCA results
    pca_df = pd.DataFrame(
        pca_results,
        index=returns_filled.index,
        columns=[f'PC{i + 1}' for i in range(n_components)]
    )

    # Get component loadings (correlation between each asset and each principal component)
    loadings = pd.DataFrame(
        pca.components_.T,
        index=valid_cols,
        columns=[f'PC{i + 1}' for i in range(n_components)]
    )

    # Get explained variance
    explained_variance = pca.explained_variance_ratio_

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(explained_variance)

    # Interpret principal components
    pc_interpretation = interpret_principal_components(loadings)

    # Return results
    results = {
        'pca_results': pca_df,
        'loadings': loadings,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'interpretation': pc_interpretation
    }

    return results


def interpret_principal_components(loadings, threshold=0.3):
    """
    Interpret principal components based on their loadings.

    Parameters:
    -----------
    loadings: pandas DataFrame
        DataFrame with component loadings
    threshold: float
        Threshold for significant loadings

    Returns:
    --------
    list
        List of dictionaries with interpretations
    """
    interpretations = []

    for pc in loadings.columns:
        # Get significant positive and negative loadings
        pos_loadings = loadings[pc][loadings[pc] > threshold].sort_values(ascending=False)
        neg_loadings = loadings[pc][loadings[pc] < -threshold].sort_values(ascending=True)

        # Count sectors
        pos_sectors = []
        neg_sectors = []

        # For stocks with sector information, count sector representation
        # This is a simplified approach. In practice, you would use actual sector data.

        # Create interpretation dictionary
        interpretation = {
            'component': pc,
            'positive_assets': pos_loadings.index.tolist(),
            'negative_assets': neg_loadings.index.tolist()
        }

        # Interpret based on loadings
        if len(pos_loadings) > 0 and len(neg_loadings) > 0:
            # Common patterns for the first few PCs
            if pc == 'PC1':
                interpretation['label'] = "Market Factor"
                interpretation['description'] = "Overall market direction affecting most assets"
            elif pc == 'PC2':
                interpretation['label'] = "Value vs Growth"
                interpretation['description'] = "Contrast between value and growth assets"
            elif pc == 'PC3':
                interpretation['label'] = "Size Factor"
                interpretation['description'] = "Contrast between large-cap and small-cap assets"
            elif pc == 'PC4':
                interpretation['label'] = "Sector Rotation"
                interpretation['description'] = "Sector-specific moves affecting certain asset groups"
            else:
                interpretation['label'] = f"Factor {pc}"
                interpretation['description'] = "Mixed factor with various influences"
        else:
            interpretation['label'] = f"Minor Factor {pc}"
            interpretation['description'] = "Factor with limited influence across assets"

        interpretations.append(interpretation)

    return interpretations


def calculate_factor_exposures(returns, factor_returns, min_periods=60):
    """
    Calculate asset exposures to risk factors using regression.

    Parameters:
    -----------
    returns: pandas DataFrame
        DataFrame containing asset returns
    factor_returns: pandas DataFrame
        DataFrame containing factor returns
    min_periods: int
        Minimum number of periods for regression

    Returns:
    --------
    dict
        Dictionary with factor exposures
    """
    # Initialize results dictionary
    results = {}

    # Calculate exposures for each asset
    for asset in returns.columns:
        # Get asset returns
        asset_returns = returns[asset].dropna()

        # Skip assets with too few data points
        if len(asset_returns) < min_periods:
            continue

        # Align factor returns with asset returns
        aligned_data = pd.concat([asset_returns, factor_returns], axis=1).dropna()

        if len(aligned_data) < min_periods:
            continue

        # Prepare data for regression
        y = aligned_data[asset].values
        X = aligned_data[factor_returns.columns].values

        # Add constant for intercept
        X = sm.add_constant(X)

        # Run regression
        try:
            model = sm.OLS(y, X).fit()

            # Extract coefficients
            alpha = model.params[0]
            betas = model.params[1:]

            # Calculate metrics
            r_squared = model.rsquared
            specific_risk = np.sqrt(np.var(model.resid))
            factor_risk = np.sqrt(np.var(y) - np.var(model.resid))

            # Store results
            results[asset] = {
                'alpha': alpha,
                'betas': dict(zip(factor_returns.columns, betas)),
                'r_squared': r_squared,
                'specific_risk': specific_risk,
                'factor_risk': factor_risk,
                'total_risk': np.sqrt(np.var(y)),
                'factor_contribution': factor_risk / np.sqrt(np.var(y)) if np.var(y) > 0 else 0
            }
        except:
            # Skip assets with regression errors
            continue

    return results


def analyze_eigenportfolios(returns, min_history=252):
    """
    Analyze eigenportfolios to identify market driving forces.

    Parameters:
    -----------
    returns: pandas DataFrame
        DataFrame containing return series for multiple assets
    min_history: int
        Minimum history required for analysis

    Returns:
    --------
    dict
        Dictionary with eigenportfolio analysis
    """
    # Remove columns with too many NaN values
    valid_cols = []

    for col in returns.columns:
        if returns[col].count() >= min_history:
            valid_cols.append(col)

    if len(valid_cols) < 5:
        print(f"[WARNING] Not enough valid columns for eigenportfolio analysis: {len(valid_cols)} < 5")
        return None

    # Use valid columns
    returns_valid = returns[valid_cols].copy()

    # Fill NaN values (simple forward fill then backward fill)
    returns_filled = returns_valid.fillna(method='ffill').fillna(method='bfill')

    # Calculate returns covariance matrix
    cov_matrix = returns_filled.cov()

    # Calculate eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Create eigenportfolios
    eigenportfolios = pd.DataFrame(
        eigenvectors,
        index=valid_cols,
        columns=[f'Eigen{i + 1}' for i in range(len(valid_cols))]
    )

    # Calculate eigenportfolio returns
    eigen_returns = pd.DataFrame(index=returns_filled.index)

    for i in range(min(10, len(valid_cols))):
        weights = eigenportfolios[f'Eigen{i + 1}']

        # Normalize weights to sum to 1
        weights = weights / weights.abs().sum()

        # Calculate portfolio returns
        eigen_returns[f'Eigen{i + 1}'] = (returns_filled * weights).sum(axis=1)

    # Calculate percentage of variance explained
    total_variance = np.sum(eigenvalues)
    explained_variance = eigenvalues / total_variance if total_variance > 0 else np.zeros_like(eigenvalues)

    # Calculate cumulative variance explained
    cumulative_variance = np.cumsum(explained_variance)

    # Calculate risk concentration
    risk_concentration = (eigenvalues[0] / total_variance) if total_variance > 0 else 0

    # Calculate effective rank
    effective_rank = (np.sum(eigenvalues) ** 2) / np.sum(eigenvalues ** 2) if np.sum(eigenvalues ** 2) > 0 else 0

    # Return results
    results = {
        'eigenvalues': eigenvalues,
        'eigenportfolios': eigenportfolios,
        'eigen_returns': eigen_returns,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'risk_concentration': risk_concentration,
        'effective_rank': effective_rank
    }

    return results


def run_risk_factor_analysis(data, min_history=252):
    """
    Run comprehensive risk factor decomposition analysis.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data for multiple assets
    min_history: int
        Minimum history required for analysis

    Returns:
    --------
    dict
        Dictionary with risk factor analysis results
    """
    print("[INFO] Running risk factor decomposition analysis")

    # Check if we have enough assets
    if len(data.columns) < 5:
        print(f"[WARNING] Not enough assets for risk factor analysis: {len(data.columns)} < 5")
        return None

    # Calculate returns
    returns = data.pct_change().dropna(how='all')

    # Check if we have enough history
    if len(returns) < min_history:
        print(f"[WARNING] Not enough history for risk factor analysis: {len(returns)} < {min_history}")
        return None

    # Initialize results dictionary
    results = {}

    # 1. Perform PCA to identify risk factors
    try:
        pca_results = perform_risk_factor_pca(returns, min_history=min_history)

        if pca_results is not None:
            results['pca'] = pca_results

            # Extract key insights
            n_components = len(pca_results['explained_variance'])
            total_var_explained = pca_results['cumulative_variance'][-1]

            print(
                f"[INFO] Identified {n_components} principal components explaining {total_var_explained:.2%} of variance")

            # Print top components
            for i in range(min(3, n_components)):
                pc_label = pca_results['interpretation'][i]['label']
                pc_variance = pca_results['explained_variance'][i]
                print(f"[INFO] {pc_label} explains {pc_variance:.2%} of variance")
    except Exception as e:
        print(f"[WARNING] Error in PCA analysis: {e}")

    # 2. Calculate factor exposures
    try:
        if 'pca' in results:
            # Use PCA results as factors
            factor_returns = results['pca']['pca_results']

            # Calculate exposures
            exposures = calculate_factor_exposures(returns, factor_returns)

            if exposures:
                results['factor_exposures'] = exposures

                # Calculate average factor contributions
                avg_contributions = {}

                for asset, metrics in exposures.items():
                    for factor, beta in metrics['betas'].items():
                        if factor not in avg_contributions:
                            avg_contributions[factor] = []

                        avg_contributions[factor].append(abs(beta))

                # Calculate average absolute exposure to each factor
                avg_abs_exposures = {factor: np.mean(betas) for factor, betas in avg_contributions.items()}

                results['average_exposures'] = avg_abs_exposures

                print(f"[INFO] Calculated factor exposures for {len(exposures)} assets")
    except Exception as e:
        print(f"[WARNING] Error calculating factor exposures: {e}")

    # 3. Analyze eigenportfolios
    try:
        eigen_results = analyze_eigenportfolios(returns, min_history=min_history)

        if eigen_results is not None:
            results['eigenportfolios'] = eigen_results

            # Extract key insights
            risk_concentration = eigen_results['risk_concentration']
            effective_rank = eigen_results['effective_rank']

            print(f"[INFO] Risk concentration: {risk_concentration:.2%}")
            print(f"[INFO] Effective rank: {effective_rank:.1f}")
    except Exception as e:
        print(f"[WARNING] Error in eigenportfolio analysis: {e}")

    return results


#########################
# 12. Non-Parametric Market Inefficiency Metrics
#########################

def calculate_approximate_entropy(time_series, m=2, r=0.2, normalize=True):
    """
    Calculate approximate entropy to quantify predictability.
    
    Parameters:
    -----------
    time_series: pandas Series or numpy array
        Input time series data
    m: int
        Embedding dimension
    r: float
        Tolerance (as a fraction of standard deviation)
    normalize: bool
        Whether to normalize the series first
        
    Returns:
    --------
    float
        Approximate entropy value
    """
    # Convert to numpy array if needed
    if isinstance(time_series, pd.Series):
        time_series = time_series.values
    
    # Remove NaN values
    time_series = time_series[~np.isnan(time_series)]
    
    N = len(time_series)
    
    if N < m + 1:
        return None
    
    # Normalize if requested
    if normalize:
        time_series = (time_series - np.mean(time_series)) / np.std(time_series)
    
    # Calculate absolute tolerance
    r_abs = r * np.std(time_series)
    
    # Function to count matches
    def count_matches(window, template, tolerance):
        return np.sum(np.max(np.abs(window - template), axis=1) <= tolerance)
    
    # Create embedding vectors
    def create_embedding(ts, dim):
        return np.array([ts[i:i+dim] for i in range(N - dim + 1)])
    
    # Create embeddings of dimension m and m+1
    embedding_m = create_embedding(time_series, m)
    embedding_m1 = create_embedding(time_series, m + 1)
    
    # Calculate Phi(m) - the negative log of the conditional probability
    # For embedding dimension m
    phi_m = 0
    for i in range(N - m + 1):
        template = embedding_m[i]
        
        # Count matches
        matches = count_matches(embedding_m, template, r_abs)
        
        # Calculate probability
        if matches > 0:
            phi_m += np.log(matches / (N - m + 1))
    
    phi_m /= (N - m + 1)
    
    # Calculate Phi(m+1) for embedding dimension m+1
    phi_m1 = 0
    for i in range(N - m):
        template = embedding_m1[i]
        
        # Count matches
        matches = count_matches(embedding_m1, template, r_abs)
        
        # Calculate probability
        if matches > 0:
            phi_m1 += np.log(matches / (N - m))
    
    phi_m1 /= (N - m)
    
    # Calculate approximate entropy
    apen = phi_m - phi_m1
    
    return apen


def calculate_transfer_entropy(source, target, k=1, normalize=True):
    """
    Calculate transfer entropy to detect information flow between stocks.
    
    Parameters:
    -----------
    source: pandas Series or numpy array
        Source time series
    target: pandas Series or numpy array
        Target time series
    k: int
        History length
    normalize: bool
        Whether to normalize the series first
        
    Returns:
    --------
    float
        Transfer entropy value
    """
    # Convert to numpy arrays if needed
    if isinstance(source, pd.Series):
        source = source.values
    
    if isinstance(target, pd.Series):
        target = target.values
    
    # Align lengths
    min_len = min(len(source), len(target))
    source = source[:min_len]
    target = target[:min_len]
    
    # Remove NaN values
    valid_idx = ~(np.isnan(source) | np.isnan(target))
    source = source[valid_idx]
    target = target[valid_idx]
    
    if len(source) < k + 2:
        return None
    
    # Normalize if requested
    if normalize:
        source = (source - np.mean(source)) / np.std(source)
        target = (target - np.mean(target)) / np.std(target)
    
    # Discretize data (simple median split)
    source_binary = (source > np.median(source)).astype(int)
    target_binary = (target > np.median(target)).astype(int)
    
    # Function to calculate entropy using scipy
    def entropy_func(hist_vals):
        # Calculate probability distribution
        bin_count = np.bincount(hist_vals)
        # Avoid division by zero
        if len(bin_count) == 0 or np.sum(bin_count) == 0:
            return 0
        prob_dist = bin_count / np.sum(bin_count)
        
        # Filter out zeros
        prob_dist = prob_dist[prob_dist > 0]
        
        # Calculate entropy using scipy
        return scipy_entropy(prob_dist)
    
    # Function to calculate conditional entropy
    def conditional_entropy(x_vals, y_vals):
        if len(x_vals) != len(y_vals):
            return None
            
        # Create joint distribution
        joint_vals = np.array([x_vals, y_vals]).T
        
        max_x = np.max(x_vals) + 1
        max_y = np.max(y_vals) + 1
        
        # Convert to indices (ravel_multi_index could fail if values too large)
        if max_x * max_y < 1000:  # Safe size for joint distribution
            # Create joint indices
            joint_idx = np.ravel_multi_index(joint_vals.T, [max_x, max_y])
            
            # Calculate joint entropy
            joint_entropy = entropy_func(joint_idx)
            
            # Calculate entropy of y
            y_entropy = entropy_func(y_vals)
            
            # Calculate conditional entropy
            return joint_entropy - y_entropy
        else:
            # Manual approach for larger distributions
            joint_counts = {}
            y_counts = {}
            
            # Count joint occurrences
            for i in range(len(x_vals)):
                joint_key = (x_vals[i], y_vals[i])
                y_key = y_vals[i]
                
                if joint_key in joint_counts:
                    joint_counts[joint_key] += 1
                else:
                    joint_counts[joint_key] = 1
                    
                if y_key in y_counts:
                    y_counts[y_key] += 1
                else:
                    y_counts[y_key] = 1
            
            # Calculate entropies
            total = len(x_vals)
            
            joint_entropy = 0
            for count in joint_counts.values():
                p = count / total
                joint_entropy -= p * np.log2(p)
                
            y_entropy = 0
            for count in y_counts.values():
                p = count / total
                y_entropy -= p * np.log2(p)
                
            return joint_entropy - y_entropy
    
    # Create time-shifted sequences
    target_past = target_binary[:-1]
    target_future = target_binary[1:]
    source_past = source_binary[:-1]
    
    # Calculate transfer entropy components
    
    # H(target_future | target_past)
    h_target_future_given_target_past = conditional_entropy(target_past, target_future)
    
    # H(target_future | target_past, source_past)
    # Combine target_past and source_past
    joint_past = np.zeros(len(target_past), dtype=int)
    for i in range(len(target_past)):
        joint_past[i] = target_past[i] * 2 + source_past[i]
    
    h_target_future_given_joint_past = conditional_entropy(joint_past, target_future)
    
    # Calculate transfer entropy
    if h_target_future_given_target_past is None or h_target_future_given_joint_past is None:
        return None
    
    transfer_entropy = h_target_future_given_target_past - h_target_future_given_joint_past
    
    return transfer_entropy


def calculate_market_efficiency_coefficient(time_series, window=10):
    """
    Calculate Market Efficiency Coefficient (MEC).
    
    Parameters:
    -----------
    time_series: pandas Series
        Input price time series
    window: int
        Rolling window for variance calculation
        
    Returns:
    --------
    float
        Market Efficiency Coefficient
    """
    # Calculate log returns
    log_returns = np.log(time_series / time_series.shift(1)).dropna()
    
    # Calculate variance of k-period returns
    var_k = np.var(np.log(time_series / time_series.shift(window)).dropna())
    
    # Calculate sum of variances of 1-period returns
    var_1_sum = window * np.var(log_returns)
    
    # Calculate MEC
    if var_1_sum == 0:
        return None
    
    mec = var_k / var_1_sum
    
    return mec


def run_market_inefficiency_analysis(data, symbol, windows=(1, 5, 20, 60)):
    """
    Run comprehensive market inefficiency analysis.
    
    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    windows: tuple
        Time windows for analysis (in days)
        
    Returns:
    --------
    dict
        Dictionary with market inefficiency analysis results
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")
    
    # Get price data
    prices = data[symbol].copy()
    
    print(f"[INFO] Running market inefficiency analysis for {symbol}")
    
    # Check if we have enough data
    min_required = max(windows) * 2 + 10
    
    if len(prices) < min_required:
        print(f"[WARNING] Not enough data for market inefficiency analysis: {len(prices)} < {min_required}")
        return None
    
    # Calculate returns
    returns = prices.pct_change().dropna()
    
    # Initialize results dictionary
    results = {}
    
    # 1. Calculate approximate entropy for different embedding dimensions
    try:
        apen_values = {}
        
        for m in [2, 3, 4]:
            apen = calculate_approximate_entropy(returns, m=m)
            
            if apen is not None:
                apen_values[f'ApEn(m={m})'] = apen
        
        if apen_values:
            results['approximate_entropy'] = apen_values
            
            # Interpret results
            mean_apen = np.mean(list(apen_values.values()))
            
            if mean_apen < 0.3:
                apen_interp = "High predictability (low complexity)"
            elif mean_apen < 0.6:
                apen_interp = "Moderate predictability"
            else:
                apen_interp = "Low predictability (high complexity)"
            
            results['apen_interpretation'] = apen_interp
            
            print(f"[INFO] Approximate Entropy: {mean_apen:.4f} - {apen_interp}")
    except Exception as e:
        print(f"[WARNING] Error calculating approximate entropy: {e}")
    
    # 2. Calculate transfer entropy with market index or sector
    # (For demonstration, we'll use a synthetic benchmark)
    try:
        # Create synthetic benchmark (weighted average of all columns)
        if len(data.columns) > 1:
            # Use other columns as benchmark
            other_cols = [col for col in data.columns if col != symbol]
            
            # Use up to 5 other columns
            benchmark_cols = other_cols[:min(5, len(other_cols))]
            
            if benchmark_cols:
                benchmark = data[benchmark_cols].mean(axis=1)
                benchmark_returns = benchmark.pct_change().dropna()
                
                # Align returns
                aligned_data = pd.concat([returns, benchmark_returns], axis=1).dropna()
                
                if len(aligned_data) >= 60:
                    # Calculate transfer entropy in both directions
                    te_to_market = calculate_transfer_entropy(aligned_data.iloc[:, 0], aligned_data.iloc[:, 1])
                    te_from_market = calculate_transfer_entropy(aligned_data.iloc[:, 1], aligned_data.iloc[:, 0])
                    
                    if te_to_market is not None and te_from_market is not None:
                        results['transfer_entropy'] = {
                            'to_market': te_to_market,
                            'from_market': te_from_market,
                            'net_flow': te_to_market - te_from_market
                        }
                        
                        # Interpret results
                        net_flow = te_to_market - te_from_market
                        
                        if net_flow > 0.05:
                            te_interp = f"{symbol} leads the market (information source)"
                        elif net_flow < -0.05:
                            te_interp = f"{symbol} follows the market (information receiver)"
                        else:
                            te_interp = "Balanced information flow with the market"
                        
                        results['te_interpretation'] = te_interp
                        
                        print(f"[INFO] Transfer Entropy - Net flow: {net_flow:.4f} - {te_interp}")
    except Exception as e:
        print(f"[WARNING] Error calculating transfer entropy: {e}")
    
    # 3. Calculate Market Efficiency Coefficient for different windows
    try:
        mec_values = {}
        
        for window in windows:
            mec = calculate_market_efficiency_coefficient(prices, window=window)
            
            if mec is not None:
                mec_values[f'MEC(w={window})'] = mec
        
        if mec_values:
            results['market_efficiency_coefficient'] = mec_values
            
            # Calculate average MEC
            mean_mec = np.mean(list(mec_values.values()))
            
            # Interpret results
            if mean_mec < 0.7:
                mec_interp = "Strong mean reversion (inefficient market)"
            elif mean_mec < 0.9:
                mec_interp = "Moderate mean reversion"
            elif mean_mec < 1.1:
                mec_interp = "Random walk (efficient market)"
            else:
                mec_interp = "Momentum effects (inefficient market)"
            
            results['mec_interpretation'] = mec_interp
            
            print(f"[INFO] Market Efficiency Coefficient: {mean_mec:.4f} - {mec_interp}")
    except Exception as e:
        print(f"[WARNING] Error calculating Market Efficiency Coefficient: {e}")
    
    # 4. Calculate combined inefficiency score
    try:
        if 'approximate_entropy' in results and 'market_efficiency_coefficient' in results:
            # Extract metrics
            mean_apen = np.mean(list(results['approximate_entropy'].values()))
            mean_mec = np.mean(list(results['market_efficiency_coefficient'].values()))
            
            # Normalize metrics
            norm_apen = 1 - mean_apen  # Lower ApEn means higher predictability
            
            # Normalize MEC (1.0 is efficient, deviation means inefficient)
            norm_mec = 1 - abs(mean_mec - 1)
            
            # Calculate inefficiency score (0 = efficient, 1 = inefficient)
            inefficiency_score = 1 - (norm_apen * 0.5 + norm_mec * 0.5)
            
            results['inefficiency_score'] = inefficiency_score
            
            # Interpret results
            if inefficiency_score < 0.3:
                score_interp = "Highly efficient market"
            elif inefficiency_score < 0.5:
                score_interp = "Moderately efficient market"
            elif inefficiency_score < 0.7:
                score_interp = "Moderately inefficient market"
            else:
                score_interp = "Highly inefficient market"
            
            results['inefficiency_interpretation'] = score_interp
            
            print(f"[INFO] Market Inefficiency Score: {inefficiency_score:.4f} - {score_interp}")
    except Exception as e:
        print(f"[WARNING] Error calculating inefficiency score: {e}")
    
    return results


#########################
# Black-Scholes Option Pricing Model
#########################

def black_scholes(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option price using Black-Scholes model.

    Parameters:
    -----------
    S: float
        Current stock price
    K: float
        Strike price
    T: float
        Time to expiration (in years)
    r: float
        Risk-free interest rate (annualized)
    sigma: float
        Volatility (annualized)
    option_type: str
        Option type ('call' or 'put')

    Returns:
    --------
    float
        Option price
    """
    from scipy.stats import norm

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Calculate option price
    if option_type.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price


def calculate_implied_volatility(option_price, S, K, T, r, option_type='call', precision=0.0001, max_iterations=100):
    """
    Calculate implied volatility using Newton-Raphson method.

    Parameters:
    -----------
    option_price: float
        Market price of the option
    S: float
        Current stock price
    K: float
        Strike price
    T: float
        Time to expiration (in years)
    r: float
        Risk-free interest rate (annualized)
    option_type: str
        Option type ('call' or 'put')
    precision: float
        Desired precision
    max_iterations: int
        Maximum number of iterations

    Returns:
    --------
    float
        Implied volatility
    """
    from scipy.stats import norm

    # Initial guess for volatility
    sigma = 0.2

    for i in range(max_iterations):
        # Calculate option price
        price = black_scholes(S, K, T, r, sigma, option_type)

        # Check if we're within desired precision
        if abs(price - option_price) < precision:
            return sigma

        # Calculate vega (derivative of price with respect to volatility)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)

        # Update volatility using Newton-Raphson
        if vega == 0:
            # Avoid division by zero
            sigma = sigma * 1.5
        else:
            sigma = sigma - (price - option_price) / vega

        # Ensure volatility is positive
        sigma = max(0.001, sigma)

    # If we reached max iterations, return the last sigma
    return sigma


def calculate_option_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Calculate option Greeks (delta, gamma, theta, vega, rho).

    Parameters:
    -----------
    S: float
        Current stock price
    K: float
        Strike price
    T: float
        Time to expiration (in years)
    r: float
        Risk-free interest rate (annualized)
    sigma: float
        Volatility (annualized)
    option_type: str
        Option type ('call' or 'put')

    Returns:
    --------
    dict
        Dictionary with option Greeks
    """
    from scipy.stats import norm

    # Calculate d1 and d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Common term for theta calculation
    common_theta_term = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    # Calculate Greeks for call option
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
        theta = common_theta_term - r * K * np.exp(-r * T) * norm.cdf(d2)
        rho = K * T * np.exp(-r * T) * norm.cdf(d2)
    else:  # put option
        delta = norm.cdf(d1) - 1
        theta = common_theta_term + r * K * np.exp(-r * T) * norm.cdf(-d2)
        rho = -K * T * np.exp(-r * T) * norm.cdf(-d2)

    # Greeks that are the same for calls and puts
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * np.sqrt(T) * norm.pdf(d1)

    # Return Greeks
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta / 365,  # Convert to daily theta
        'vega': vega / 100,  # Convert to 1% volatility change
        'rho': rho / 100  # Convert to 1% interest rate change
    }


def derive_implied_stock_price(options_data, current_stock_price, risk_free_rate):
    """
    Derive the implied stock price from a set of options using a weighted average approach.

    Parameters:
    -----------
    options_data: list of dict
        List of option data dictionaries, each containing:
        - 'option_type': str ('call' or 'put')
        - 'strike': float (strike price)
        - 'expiry': float (time to expiration in years)
        - 'price': float (market price of option)
        - 'volume': float (trading volume, for weighting)
    current_stock_price: float
        Current stock price
    risk_free_rate: float
        Risk-free interest rate (annualized)

    Returns:
    --------
    float
        Implied stock price
    """
    # Calculate implied volatility for each option
    for option in options_data:
        option['implied_vol'] = calculate_implied_volatility(
            option['price'],
            current_stock_price,
            option['strike'],
            option['expiry'],
            risk_free_rate,
            option['option_type']
        )

    # Filter out options with extreme implied volatilities
    filtered_options = [opt for opt in options_data if 0.05 <= opt['implied_vol'] <= 2.0]

    if not filtered_options:
        return current_stock_price  # Return current price if no valid options

    # Calculate implied stock price for each option using put-call parity
    for option in filtered_options:
        if option['option_type'].lower() == 'call':
            # For calls: S = C + K * exp(-r * T) - (put value)
            # Approximate put value using Black-Scholes
            put_value = black_scholes(
                current_stock_price,
                option['strike'],
                option['expiry'],
                risk_free_rate,
                option['implied_vol'],
                'put'
            )
            option['implied_stock'] = option['price'] + option['strike'] * np.exp(
                -risk_free_rate * option['expiry']) - put_value
        else:  # put option
            # For puts: S = P - K * exp(-r * T) + (call value)
            # Approximate call value using Black-Scholes
            call_value = black_scholes(
                current_stock_price,
                option['strike'],
                option['expiry'],
                risk_free_rate,
                option['implied_vol'],
                'call'
            )
            option['implied_stock'] = -option['price'] + option['strike'] * np.exp(
                -risk_free_rate * option['expiry']) + call_value

    # Calculate weights based on volume and closeness to current price
    total_weight = 0
    weighted_sum = 0

    for option in filtered_options:
        # Weight by volume and inverse of time to expiration (prefer closer expirations)
        weight = option['volume'] / (1 + option['expiry'])

        # Add weight based on closeness to the money
        moneyness = abs(np.log(option['strike'] / current_stock_price))
        weight *= np.exp(-moneyness * 2)  # Exponential decay for far-from-money options

        weighted_sum += option['implied_stock'] * weight
        total_weight += weight

    # Calculate weighted average implied stock price
    if total_weight > 0:
        implied_stock_price = weighted_sum / total_weight
    else:
        implied_stock_price = current_stock_price

    return implied_stock_price


#########################
# Integration with Existing Codebase
#########################

def integrate_with_existing_analysis(data, symbol, existing_analysis=None):
    """
    Integrate new quantitative functions with existing analysis.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str
        Stock symbol to analyze
    existing_analysis: dict, optional
        Existing analysis results

    Returns:
    --------
    dict
        Integrated analysis results
    """
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    # Get price data
    prices = data[symbol].copy()

    print(f"[INFO] Integrating advanced quantitative analysis for {symbol}")

    # Initialize results with existing analysis if provided
    if existing_analysis is not None:
        results = existing_analysis.copy()
    else:
        results = {}

    # Run selected analyses

    # 1. Calculate multi-fractal metrics for trend strength
    try:
        mf_results = run_multifractal_analysis(data, symbol)

        if mf_results is not None:
            results['multifractal'] = mf_results

            # Extract key metrics
            if 'daily' in mf_results and 'hurst' in mf_results['daily']:
                hurst = mf_results['daily']['hurst']
                fractal_dim = mf_results['daily'].get('fractal_dimension')

                # Add to results
                if 'metrics' not in results:
                    results['metrics'] = {}

                results['metrics']['hurst_exponent'] = hurst
                results['metrics']['fractal_dimension'] = fractal_dim
    except Exception as e:
        print(f"[WARNING] Error in multifractal analysis: {e}")

    # 2. Calculate market inefficiency metrics
    try:
        inefficiency_results = run_market_inefficiency_analysis(data, symbol)

        if inefficiency_results is not None:
            results['inefficiency'] = inefficiency_results

            # Extract key metrics
            if 'inefficiency_score' in inefficiency_results:
                score = inefficiency_results['inefficiency_score']

                # Add to results
                if 'metrics' not in results:
                    results['metrics'] = {}

                results['metrics']['inefficiency_score'] = score
    except Exception as e:
        print(f"[WARNING] Error in market inefficiency analysis: {e}")

    # 3. Detect regime changes
    try:
        regime_results = run_bayesian_regime_analysis(data, symbol)

        if regime_results is not None:
            results['regime'] = regime_results

            # Extract regime information
            if 'current_regime' in regime_results:
                current_regime = regime_results['current_regime']

                # Add to results
                if 'metrics' not in results:
                    results['metrics'] = {}

                results['metrics']['current_regime'] = current_regime
    except Exception as e:
        print(f"[WARNING] Error in regime analysis: {e}")

    # 4. Analyze market microstructure
    try:
        microstructure_results = run_market_microstructure_analysis(data, symbol)

        if microstructure_results is not None:
            results['microstructure'] = microstructure_results

            # Extract key metrics
            if 'insights' in microstructure_results:
                insights = microstructure_results['insights']

                # Add to results
                if 'metrics' not in results:
                    results['metrics'] = {}

                results['metrics']['microstructure_insights'] = insights
    except Exception as e:
        print(f"[WARNING] Error in microstructure analysis: {e}")

    # 5. Perform tail risk analysis
    try:
        tail_risk_results = run_tail_risk_analysis(data, symbol)

        if tail_risk_results is not None:
            results['tail_risk'] = tail_risk_results

            # Extract key metrics
            if 'tail_metrics' in tail_risk_results:
                metrics = tail_risk_results['tail_metrics']

                # Add to results
                if 'metrics' not in results:
                    results['metrics'] = {}

                results['metrics']['tail_risk_metrics'] = metrics
    except Exception as e:
        print(f"[WARNING] Error in tail risk analysis: {e}")

    # Merge all metrics into a combined signal
    try:
        if 'metrics' in results:
            metrics = results['metrics']

            # Define range of alpha values (0-1)
            # For sigma calculation as in original codebase
            signals = []

            # 1. Hurst exponent signal (closer to 1 = trending = higher signal)
            if 'hurst_exponent' in metrics:
                hurst = metrics['hurst_exponent']
                hurst_signal = (hurst - 0.4) / 0.6  # Normalize 0.4-1.0 to 0-1
                hurst_signal = max(0, min(1, hurst_signal))
                signals.append(hurst_signal)

            # 2. Inefficiency score signal (higher inefficiency = higher opportunity = higher signal)
            if 'inefficiency_score' in metrics:
                ineff_score = metrics['inefficiency_score']
                ineff_signal = ineff_score
                signals.append(ineff_signal)

            # 3. Regime signal
            if 'current_regime' in metrics:
                regime = metrics['current_regime']

                if 'regime_type' in regime:
                    regime_type = regime['regime_type']

                    # Map regime type to signal
                    if regime_type == "Low Volatility Bull":
                        regime_signal = 0.8
                    elif regime_type == "High Volatility Bull":
                        regime_signal = 0.6
                    elif regime_type == "Sideways/Neutral":
                        regime_signal = 0.5
                    elif regime_type == "Low Volatility Bear":
                        regime_signal = 0.4
                    elif regime_type == "High Volatility Bear":
                        regime_signal = 0.2
                    else:
                        regime_signal = 0.5

                    signals.append(regime_signal)

            # 4. Microstructure signal
            if 'microstructure_insights' in metrics and 'volume_delta_signal' in metrics['microstructure_insights']:
                ms_signal = metrics['microstructure_insights']['volume_delta_signal']

                # Map signal to value
                if ms_signal == "Bullish":
                    ms_value = 0.7
                elif ms_signal == "Bearish":
                    ms_value = 0.3
                else:
                    ms_value = 0.5

                signals.append(ms_value)

            # 5. Tail risk signal (inverse relationship)
            if 'tail_risk_metrics' in metrics and 'crash_risk_index' in metrics['tail_risk_metrics']:
                crash_risk = metrics['tail_risk_metrics']['crash_risk_index']
                # Normalize and invert (higher crash risk = lower signal)
                tailrisk_signal = 1 - (crash_risk + 1) / 2 if -1 <= crash_risk <= 1 else 0.5
                signals.append(tailrisk_signal)

            # Calculate combined signal (average)
            if signals:
                combined_signal = sum(signals) / len(signals)

                # Map to original sigma scale (0-1)
                results['combined_sigma'] = combined_signal

                # Add interpretation
                if combined_signal > 0.8:
                    interp = "STRONG BUY"
                elif combined_signal > 0.6:
                    interp = "BUY"
                elif combined_signal > 0.4:
                    interp = "HOLD"
                elif combined_signal > 0.2:
                    interp = "SELL"
                else:
                    interp = "STRONG SELL"

                results['combined_recommendation'] = interp
    except Exception as e:
        print(f"[WARNING] Error calculating combined signal: {e}")

    return results


# Main function to run integrated analysis
def run_advanced_quantitative_analysis(data, symbol=None, analyses=None):
    """
    Run selected advanced quantitative analyses.

    Parameters:
    -----------
    data: pandas DataFrame
        DataFrame containing price data
    symbol: str, optional
        Stock symbol to analyze (if None, analyze all columns)
    analyses: list, optional
        List of analyses to run (if None, run all)

    Returns:
    --------
    dict
        Analysis results
    """
    # Set default analyses
    if analyses is None:
        analyses = [
            'multifractal',
            'tail_risk',
            'wavelet',
            'regime',
            'inefficiency',
            'microstructure'
        ]

    # If no symbol provided, analyze all columns
    if symbol is None:
        results = {}

        for col in data.columns:
            print(f"\n[INFO] Running advanced quantitative analysis for {col}")

            # Run analysis for each column
            column_results = run_advanced_quantitative_analysis(data, col, analyses)

            # Store results
            results[col] = column_results

        return results

    # Check if symbol is valid
    if symbol not in data.columns:
        raise ValueError(f"Symbol {symbol} not found in data")

    print(f"\n[INFO] Running advanced quantitative analysis for {symbol}")

    # Initialize results dictionary
    results = {}

    # Run selected analyses

    # 1. Multi-fractal analysis
    if 'multifractal' in analyses:
        print("\n[INFO] Running multi-fractal analysis...")

        try:
            mf_results = run_multifractal_analysis(data, symbol)

            if mf_results is not None:
                results['multifractal'] = mf_results
                print("[INFO] Multi-fractal analysis completed")
            else:
                print("[WARNING] Multi-fractal analysis returned no results")
        except Exception as e:
            print(f"[ERROR] Error in multi-fractal analysis: {e}")

    # 2. Tail risk analysis
    if 'tail_risk' in analyses:
        print("\n[INFO] Running tail risk analysis...")

        try:
            tail_risk_results = run_tail_risk_analysis(data, symbol)

            if tail_risk_results is not None:
                results['tail_risk'] = tail_risk_results
                print("[INFO] Tail risk analysis completed")
            else:
                print("[WARNING] Tail risk analysis returned no results")
        except Exception as e:
            print(f"[ERROR] Error in tail risk analysis: {e}")

    # 3. Wavelet analysis
    if 'wavelet' in analyses:
        print("\n[INFO] Running wavelet analysis...")

        try:
            wavelet_results = run_wavelet_analysis(data, symbol)

            if wavelet_results is not None:
                results['wavelet'] = wavelet_results
                print("[INFO] Wavelet analysis completed")
            else:
                print("[WARNING] Wavelet analysis returned no results")
        except Exception as e:
            print(f"[ERROR] Error in wavelet analysis: {e}")

    # 4. Bayesian regime analysis
    if 'regime' in analyses:
        print("\n[INFO] Running Bayesian regime analysis...")

        try:
            regime_results = run_bayesian_regime_analysis(data, symbol)

            if regime_results is not None:
                results['regime'] = regime_results
                print("[INFO] Bayesian regime analysis completed")
            else:
                print("[WARNING] Bayesian regime analysis returned no results")
        except Exception as e:
            print(f"[ERROR] Error in Bayesian regime analysis: {e}")

    # 5. Market inefficiency analysis
    if 'inefficiency' in analyses:
        print("\n[INFO] Running market inefficiency analysis...")

        try:
            inefficiency_results = run_market_inefficiency_analysis(data, symbol)

            if inefficiency_results is not None:
                results['inefficiency'] = inefficiency_results
                print("[INFO] Market inefficiency analysis completed")
            else:
                print("[WARNING] Market inefficiency analysis returned no results")
        except Exception as e:
            print(f"[ERROR] Error in market inefficiency analysis: {e}")

    # 6. Market microstructure analysis
    if 'microstructure' in analyses:
        print("\n[INFO] Running market microstructure analysis...")

        try:
            microstructure_results = run_market_microstructure_analysis(data, symbol)

            if microstructure_results is not None:
                results['microstructure'] = microstructure_results
                print("[INFO] Market microstructure analysis completed")
            else:
                print("[WARNING] Market microstructure analysis returned no results")
        except Exception as e:
            print(f"[ERROR] Error in market microstructure analysis: {e}")

    # 7. Integrate with existing analysis (calculate combined signal)
    try:
        integrated_results = integrate_with_existing_analysis(data, symbol, results)

        if integrated_results is not None:
            results = integrated_results

            if 'combined_sigma' in results:
                sigma = results['combined_sigma']
                recommendation = results['combined_recommendation']

                print(f"\n[INFO] Combined signal (sigma): {sigma:.4f}")
                print(f"[INFO] Recommendation: {recommendation}")
        else:
            print("[WARNING] Integration returned no results")
    except Exception as e:
        print(f"[ERROR] Error integrating results: {e}")

    return results