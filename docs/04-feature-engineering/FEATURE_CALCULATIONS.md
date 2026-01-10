# Feature Calculations (explicit feature names)

This document enumerates every feature name produced by the indicator pipeline. Each feature is listed as it appears in the dataset (no parameterized expressions). When an indicator is produced for multiple periods the concrete feature names are listed individually.

If you need the concrete list for a specific run, export the consolidated parquet file and run:

```python
df.columns.sort()
```

Refer to the implementation for details:
- `src/data_collector/indicator_pipeline/trend_indicators.py`
- `src/data_collector/indicator_pipeline/momentum_indicators.py`
- `src/data_collector/indicator_pipeline/volatility_indicators.py`
- `src/data_collector/indicator_pipeline/volume_indicators.py`
- `src/data_collector/indicator_pipeline/feature_calculator.py`


## Trend indicators
- SMA_5: $SMA_5(t)=mean(close_{t-4..t})$
- SMA_10: $SMA_{10}(t)=mean(close_{t-9..t})$
- SMA_20: $SMA_{20}(t)=mean(close_{t-19..t})$
- SMA_50: $SMA_{50}(t)=mean(close_{t-49..t})$
- SMA_100: $SMA_{100}(t)=mean(close_{t-99..t})$
- SMA_200: $SMA_{200}(t)=mean(close_{t-199..t})$

- EMA_5: Exponential moving average of close over 5 days (ta.trend.ema_indicator).
- EMA_10: Exponential moving average of close over 10 days (ta.trend.ema_indicator).
- EMA_20: Exponential moving average of close over 20 days (ta.trend.ema_indicator).
- EMA_50: Exponential moving average of close over 50 days (ta.trend.ema_indicator).
- EMA_100: Exponential moving average of close over 100 days (ta.trend.ema_indicator).
- EMA_200: Exponential moving average of close over 200 days (ta.trend.ema_indicator).

- MACD: $EMA_{fast}(close)-EMA_{slow}(close)$ (params from feature_config.MACD_PARAMS).
- MACD_Signal: $EMA_{signal}(MACD)$.
- MACD_Histogram: $MACD - MACD\_Signal$.
- MACD_Above_Signal: $1[MACD > MACD\_Signal]$.
- MACD_Crossover: $diff(1[MACD > MACD\_Signal])$ (typically +1 on cross up, -1 on cross down).

- Ichimoku_Tenkan: Conversion line (ta.trend.ichimoku_conversion_line); standard form: $0.5*(max(high,tenkan)+min(low,tenkan))$.
- Ichimoku_Kijun: Base line (ta.trend.ichimoku_base_line); standard form: $0.5*(max(high,kijun)+min(low,kijun))$.
- Ichimoku_Senkou_A: Lead span A (ta.trend.ichimoku_a) shifted forward by displacement: $shift(displacement, 0.5*(Tenkan+Kijun))$.
- Ichimoku_Senkou_B: Lead span B (ta.trend.ichimoku_b) shifted forward by displacement: $shift(displacement, 0.5*(max(high,senkou\_b)+min(low,senkou\_b)))$.
- Ichimoku_Chikou: Lagging span: $close.shift(displacement)$.
- Ichimoku_Tenkan_Above_Kijun: $1[Tenkan > Kijun]$.
- Ichimoku_Price_Above_Cloud: $1[close > max(Senkou\_A, Senkou\_B)]$.
- Ichimoku_Price_Below_Cloud: $1[close < min(Senkou\_A, Senkou\_B)]$.
- Ichimoku_Cloud_Green: $1[Senkou\_A > Senkou\_B]$.
- Ichimoku_Cloud_Thickness: $abs(Senkou\_A - Senkou\_B)$.

## Momentum indicators
- RSI_14: RSI on close with window=14 (ta.momentum.rsi); standard form: $RSI=100-100/(1+RS)$.
- RSI_14_Overbought: $1[RSI\_14 > 70]$.
- RSI_14_Oversold: $1[RSI\_14 < 30]$.
- RSI_14_Neutral: $1[30 \le RSI\_14 \le 70]$.

- RSI_21: RSI on close with window=21 (ta.momentum.rsi).
- RSI_21_Overbought: $1[RSI\_21 > 70]$.
- RSI_21_Oversold: $1[RSI\_21 < 30]$.
- RSI_21_Neutral: $1[30 \le RSI\_21 \le 70]$.

- Stoch_K: Stochastic %K (ta.momentum.stoch) using (high, low, close) with window=k_period and smooth_window=d_period; standard form: $100*(close-L_k)/(H_k-L_k)$ with smoothing.
- Stoch_D: Stochastic signal (ta.momentum.stoch_signal) using same parameters.
- Stoch_Overbought: $1[Stoch\_K > 80 \wedge Stoch\_D > 80]$.
- Stoch_Oversold: $1[Stoch\_K < 20 \wedge Stoch\_D < 20]$.
- Stoch_K_Above_D: $1[Stoch\_K > Stoch\_D]$.
- Stoch_Crossover: $diff(1[Stoch\_K > Stoch\_D])$.

- ROC_10: Rate of change on close (ta.momentum.roc) with window=10; standard form: $(close/close.shift(10)-1)*100$.
- ROC_10_Positive: $1[ROC\_10 > 0]$.
- ROC_10_Strong_Positive: $1[ROC\_10 > 5]$.
- ROC_10_Strong_Negative: $1[ROC\_10 < -5]$.

- ROC_20: $(close/close.shift(20)-1)*100$.
- ROC_20_Positive: $1[ROC\_20 > 0]$.
- ROC_20_Strong_Positive: $1[ROC\_20 > 5]$.
- ROC_20_Strong_Negative: $1[ROC\_20 < -5]$.

- ROC_30: $(close/close.shift(30)-1)*100$.
- ROC_30_Positive: $1[ROC\_30 > 0]$.
- ROC_30_Strong_Positive: $1[ROC\_30 > 5]$.
- ROC_30_Strong_Negative: $1[ROC\_30 < -5]$.

- Williams_R_14: Williams %R (ta.momentum.williams_r) with lbp=14; standard form: $-100*(H_{14}-close)/(H_{14}-L_{14})$.
- Williams_R_14_Overbought: $1[Williams\_R\_14 > -20]$.
- Williams_R_14_Oversold: $1[Williams\_R\_14 < -80]$.
- Williams_R_14_Neutral: $1[-80 \le Williams\_R\_14 \le -20]$.

- Williams_R_21: $-100*(H_{21}-close)/(H_{21}-L_{21})$.
- Williams_R_21_Overbought: $1[Williams\_R\_21 > -20]$.
- Williams_R_21_Oversold: $1[Williams\_R\_21 < -80]$.
- Williams_R_21_Neutral: $1[-80 \le Williams\_R\_21 \le -20]$.

## Volatility indicators
- BB_Lower: Bollinger lower band (ta.volatility.bollinger_lband) using close, window=period, window_dev=std; standard form: $SMA - std*STD(close,period)$.
- BB_Middle: Bollinger middle band (ta.volatility.bollinger_mavg): $SMA(close,period)$.
- BB_Upper: Bollinger upper band (ta.volatility.bollinger_hband): $SMA + std*STD(close,period)$.
- BB_Bandwidth: Bollinger width percent (ta.volatility.bollinger_wband); commonly $(BB\_Upper-BB\_Lower)/BB\_Middle*100$.
- BB_Percent: Bollinger %B (ta.volatility.bollinger_pband): $(close-BB\_Lower)/(BB\_Upper-BB\_Lower)$.
- BB_Width: $BB\_Upper - BB\_Lower$.
- BB_Above_Upper: $1[close > BB\_Upper]$.
- BB_Below_Lower: $1[close < BB\_Lower]$.
- BB_Squeeze: $1[BB\_Width < rolling\_mean(BB\_Width,20)]$.
- BB_Position: $(close-BB\_Lower)/(BB\_Upper-BB\_Lower)$.

- ATR: Average true range (ta.volatility.average_true_range) with window=ATR_PERIOD.
- ATR_Percent: $(ATR/close)*100$.
- ATR_High_Volatility: $1[ATR\_Percent > rolling\_quantile(ATR\_Percent, lookback, high\_pct)]$.
- ATR_Low_Volatility: $1[ATR\_Percent < rolling\_quantile(ATR\_Percent, lookback, low\_pct)]$.
- ATR_Upper_Band: $close + 2*ATR$.
- ATR_Lower_Band: $close - 2*ATR$.

- Volatility_Std_10: $std(close.pct\_change(),10)$.
- Volatility_Std_20: $std(close.pct\_change(),20)$.
- Volatility_Std_30: $std(close.pct\_change(),30)$.

- Volatility_Annualized_10: $Volatility\_Std\_10*sqrt(252)$.
- Volatility_Annualized_20: $Volatility\_Std\_20*sqrt(252)$.
- Volatility_Annualized_30: $Volatility\_Std\_30*sqrt(252)$.

- HL_Volatility_10: $mean(((high-low)/close),10)$.
- HL_Volatility_20: $mean(((high-low)/close),20)$.
- HL_Volatility_30: $mean(((high-low)/close),30)$.

- Parkinson_Vol_10: $sqrt((1/(4*ln(2))) * mean(ln(high/low)^2,10))$.
- Parkinson_Vol_20: $sqrt((1/(4*ln(2))) * mean(ln(high/low)^2,20))$.
- Parkinson_Vol_30: $sqrt((1/(4*ln(2))) * mean(ln(high/low)^2,30))$.

- Vol_Regime_High: $1[Volatility\_Std\_10 > rolling\_quantile(Volatility\_Std\_10,50,0.8)]$.
- Vol_Regime_Low: $1[Volatility\_Std\_10 < rolling\_quantile(Volatility\_Std\_10,50,0.2)]$.
- Vol_Trend_Rising: $1[Volatility\_Std\_10 > Volatility\_Std\_30]$.
- Vol_Clustering: $std(std(close.pct\_change(),20),10)$.
- Vol_Clustering_High: $1[Vol\_Clustering > rolling\_quantile(Vol\_Clustering,50,0.7)]$.

## Volume indicators
- OBV: On-balance volume (ta.volume.on_balance_volume); cumulative signed volume by close direction.
- OBV_SMA_10: $mean(OBV,10)$.
- OBV_SMA_20: $mean(OBV,20)$.
- OBV_Above_SMA_10: $1[OBV > OBV\_SMA\_10]$.
- OBV_Above_SMA_20: $1[OBV > OBV\_SMA\_20]$.
- OBV_Rising: $1[OBV > OBV.shift(1)]$.
- OBV_Falling: $1[OBV < OBV.shift(1)]$.
- OBV_Momentum_5: $OBV - OBV.shift(5)$.
- OBV_Momentum_10: $OBV - OBV.shift(10)$.
- OBV_Price_Divergence: $1[(close.pct\_change(5)>0 \wedge OBV.pct\_change(5)<0) \vee (close.pct\_change(5)<0 \wedge OBV.pct\_change(5)>0)]$.

- VPT: Volume price trend (ta.volume.volume_price_trend); cumulative sum of $volume*(close-close.shift(1))/close.shift(1)$.
- VPT_SMA_10: $mean(VPT,10)$.
- VPT_SMA_20: $mean(VPT,20)$.
- VPT_Above_SMA_10: $1[VPT > VPT\_SMA\_10]$.
- VPT_Above_SMA_20: $1[VPT > VPT\_SMA\_20]$.
- VPT_Rising: $1[VPT > VPT.shift(1)]$.
- VPT_Falling: $1[VPT < VPT.shift(1)]$.
- VPT_Momentum_5: $VPT - VPT.shift(5)$.
- VPT_Momentum_10: $VPT - VPT.shift(10)$.
- VPT_ROC_5: $VPT.pct\_change(5)*100$ (inf/-inf sanitized to NaN).
- VPT_ROC_10: $VPT.pct\_change(10)*100$ (inf/-inf sanitized to NaN).

- AD_Line: Accumulation/distribution line (ta.volume.acc_dist_index); cumulative money flow volume.
- AD_SMA_10: $mean(AD\_Line,10)$.
- AD_SMA_20: $mean(AD\_Line,20)$.
- AD_Above_SMA_10: $1[AD\_Line > AD\_SMA\_10]$.
- AD_Above_SMA_20: $1[AD\_Line > AD\_SMA\_20]$.
- AD_Rising: $1[AD\_Line > AD\_Line.shift(1)]$.
- AD_Falling: $1[AD\_Line < AD\_Line.shift(1)]$.
- AD_Momentum_5: $AD\_Line - AD\_Line.shift(5)$.
- AD_Momentum_10: $AD\_Line - AD\_Line.shift(10)$.
- AD_Oscillator: $AD\_Line - EMA(AD\_Line, span=10)$.
- AD_Oscillator_Above_Zero: $1[AD\_Oscillator > 0]$.

- Volume_MA_10: $mean(volume,10)$.
- Volume_MA_20: $mean(volume,20)$.
- Volume_MA_50: $mean(volume,50)$.

- Volume_Ratio_10: $volume/Volume\_MA\_10$.
- Volume_Ratio_20: $volume/Volume\_MA\_20$.
- Volume_Ratio_50: $volume/Volume\_MA\_50$.

- High_Volume_10: $1[volume > 1.5*Volume\_MA\_10]$.
- High_Volume_20: $1[volume > 1.5*Volume\_MA\_20]$.
- High_Volume_50: $1[volume > 1.5*Volume\_MA\_50]$.

- Low_Volume_10: $1[volume < 0.5*Volume\_MA\_10]$.
- Low_Volume_20: $1[volume < 0.5*Volume\_MA\_20]$.
- Low_Volume_50: $1[volume < 0.5*Volume\_MA\_50]$.

- Volume_Trend_10: Rolling(10) apply: 1 if last(volume_window) > first(volume_window) else 0.
- Volume_Trend_20: Rolling(20) apply: 1 if last(volume_window) > first(volume_window) else 0.
- Volume_Trend_50: Rolling(50) apply: 1 if last(volume_window) > first(volume_window) else 0.

- PV_Correlation_10: $corr(close.pct\_change(), volume.pct\_change(), window=10)$.
- PV_Correlation_20: $corr(close.pct\_change(), volume.pct\_change(), window=20)$.

- Volume_Confirms_Uptrend: $1[(close.pct\_change()>0) \wedge (volume > mean(volume,10))]$.
- Volume_Confirms_Downtrend: $1[(close.pct\_change()<0) \wedge (volume > mean(volume,10))]$.
- Volume_Divergence: $1[(close.pct\_change()>0 \wedge volume.pct\_change()<0) \vee (close.pct\_change()<0 \wedge volume.pct\_change()>0)]$.

- Volume_Percentile_50: Rolling(50): $(last-min)/(max-min)$ else 0.5.
- Volume_Regime_High: $1[volume > mean(volume,50) + 2*std(volume,50)]$.
- Volume_Regime_Low: $1[volume < mean(volume,50) - std(volume,50)]$.

- MFI: Money flow index (ta.volume.money_flow_index) with window=14.
- MFI_Overbought: $1[MFI > 80]$.
- MFI_Oversold: $1[MFI < 20]$.
- MFI_Neutral: $1[20 \le MFI \le 80]$.
- MFI_Rising: $1[MFI > MFI.shift(1)]$.
- MFI_Falling: $1[MFI < MFI.shift(1)]$.
- MFI_Momentum_5: $MFI - MFI.shift(5)$.
- MFI_Momentum_10: $MFI - MFI.shift(10)$.
- MFI_Price_Divergence: $1[(close.pct\_change(5)>0 \wedge MFI.pct\_change(5)<0) \vee (close.pct\_change(5)<0 \wedge MFI.pct\_change(5)>0)]$.

## Price & basic derived features
- Price_High_Low_Ratio: $high/low$.
- Price_Close_Open_Ratio: $close/open$.
- Price_Range: $high-low$.
- Price_Range_Pct: $(high-low)/close$ (close=0 treated as NaN).
- Body_Size: $abs(close-open)$.
- Upper_Shadow: $high-max(close,open)$.
- Lower_Shadow: $min(close,open)-low$.
- Body_Shadow_Ratio: $Body\_Size/(Upper\_Shadow+Lower\_Shadow+1e-8)$.

- Return_1D: $close.pct\_change(1)$.
- Return_5D: $close.pct\_change(5)$.
- Return_10D: $close.pct\_change(10)$.
- Return_20D: $close.pct\_change(20)$.
- Return_30D: $close.pct\_change(30)$.

- Log_Return_1D: $ln(close/close.shift(1))$.
- Log_Return_5D: $ln(close/close.shift(5))$.
- Log_Return_10D: $ln(close/close.shift(10))$.
- Log_Return_20D: $ln(close/close.shift(20))$.
- Log_Return_30D: $ln(close/close.shift(30))$.

- Volume_Price_Ratio: $volume/close$.
- Volume_Range_Ratio: $volume/Price\_Range$ (Price_Range=0 treated as NaN).
- Gap_Up: $1[open > high.shift(1)]$.
- Gap_Down: $1[open < low.shift(1)]$.
- Gap_Size: $open - close.shift(1)$.

- Future_High_5D: $high.shift(-5)$ (future trading rows, computed after date sort/alignment).
- Future_High_10D: $high.shift(-10)$.
- Future_High_20D: $high.shift(-20)$.
- Future_Close_5D: $close.shift(-5)$.
- Future_Close_10D: $close.shift(-10)$.
- Future_Close_20D: $close.shift(-20)$.

- dayofweek: $date.dayofweek$ (Monday=0) derived from the working date series.

## Dividend-related features
- dividend_amount: Sum of cash_amount on the mapped ex-dividend trading date (nearest trading date within +/- 3 days).
- dividend_yield: $dividend\_amount/close$ on that mapped date (NaN if close missing/0).
- is_ex_dividend: 1 on mapped ex-dividend trading dates, else 0.
- days_since_last_dividend: For each date, $(current\_date - prev\_ex\_date).days$ (NaN before first).
- days_to_next_dividend: For each date, $(next\_ex\_date - current\_date).days$ (NaN after last).

## Metadata and monitoring fields
- adjusted_close
- ticker
- feature_version
- calculation_date
- quality_score
- warnings
- total_features
- file_size_mb
- record_count
- feature_categories


