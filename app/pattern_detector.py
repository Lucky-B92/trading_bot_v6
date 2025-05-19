import numpy as np
import pandas as pd
from .ml_trainer import predict_with_model

def detect_patterns(df):
    """Detecta padrões de reversão e volume sem TA-Lib."""
    # Calcular EMAs adicionais
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()

    # Calcular volume médio
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']

    # Definir suporte como o mínimo dos últimos 20 períodos
    support = df['low'].rolling(20).min().iloc[-1]
    current_price = df['close'].iloc[-1]

    # Detectar padrão Morning Star (3 candles)
    morning_star = 0
    if len(df) >= 3:
        c1 = df.iloc[-3]
        c2 = df.iloc[-2]
        c3 = df.iloc[-1]

        cond1 = c1['close'] < c1['open']  # candle vermelho
        cond2 = abs(c2['close'] - c2['open']) < 0.3 * (c2['high'] - c2['low'])  # doji ou candle pequeno
        cond3 = c3['close'] > c3['open'] and c3['close'] > ((c1['open'] + c1['close']) / 2)  # candle verde que fecha acima da metade do c1

        if cond1 and cond2 and cond3:
            morning_star = 1

    # Detectar pivôs de fundo
    def find_local_pivots(df, window=3):
        pivots = []
        for i in range(window, len(df) - window):
            is_pivot = True
            for j in range(1, window + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_pivot = False
                    break
            if is_pivot:
                pivots.append(i)
        return pivots

    pivot_indices = find_local_pivots(df)

    double_bottom_detected = 0
    if len(pivot_indices) >= 2:
        last_pivot = df['low'].iloc[pivot_indices[-1]]
        prev_pivot = df['low'].iloc[pivot_indices[-2]]
        diff = abs(last_pivot - prev_pivot) / prev_pivot

        if diff <= 0.02 and df['close'].iloc[-1] > df['close'].iloc[pivot_indices[-1]]:
            double_bottom_detected = 1

    patterns = {
        'doji': ((df['high'] - df['low']) <= (0.01 * df['close'])) &
                (abs(df['close'] - df['open']) <= (0.01 * df['close'])),
        'hammer': ((df['close'] > df['open']) &
                   ((df['close'] - df['low']) > 2 * (df['high'] - df['close'])) &
                   ((df['open'] - df['low']) > 2 * (df['high'] - df['open']))),
        'engulfing': ((df['close'].shift(1) < df['open'].shift(1)) &
                      (df['close'] > df['open']) &
                      (df['close'] > df['open'].shift(1)) &
                      (df['open'] < df['close'].shift(1))),
        'ema_cross': df['ema8'] > df['ema21'],
        'macd_bullish': df['macd'] > df['macd_signal'],
        'rsi_ok': (df['rsi'] > 40) & (df['rsi'] < 70),
        'above_support': df['close'] > support,
        'volume_spike': df['volume_ratio'] > 1.2,
        'bullish_engulfing': ((df['close'].shift(1) < df['open'].shift(1)) &
                              (df['close'] > df['open']) &
                              (df['close'] > df['open'].shift(1)) &
                              (df['open'] < df['close'].shift(1))),
        'morning_star': morning_star,
        'double_bottom': double_bottom_detected
    }

    # Ajustar os índices para garantir que sejam idênticos
    last_index = df.index[-1]
    patterns = {k: v.reindex([last_index]).fillna(0).iloc[0] if isinstance(v, pd.Series) else v for k, v in patterns.items()}

    return patterns



def detect_market_regime(df):  # Já está correto
    sma_50 = df['close'].rolling(50).mean()
    sma_200 = df['close'].rolling(200).mean()
    if sma_50.iloc[-1] > sma_200.iloc[-1]:
        return 'bull'
    elif sma_50.iloc[-1] < sma_200.iloc[-1]:
        return 'bear'
    return 'sideways'