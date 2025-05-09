import numpy as np
import pandas as pd
from .ml_trainer import predict_with_model

def detect_patterns(df):
    """Implementação alternativa sem TA-Lib"""
    # Calcular EMAs adicionais
    df['ema8'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema21'] = df['close'].ewm(span=21, adjust=False).mean()
    
    # Calcular volume médio
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma']
    
    # Definir suporte como o mínimo dos últimos 20 períodos
    support = df['low'].rolling(20).min().iloc[-1]
    current_price = df['close'].iloc[-1]
    
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
                             (df['open'] < df['close'].shift(1)))
    }

    # Ajustar os índices para garantir que sejam idênticos
    last_index = df.index[-1]
    patterns = {k: v.reindex([last_index]).fillna(0).iloc[0] for k, v in patterns.items()}

    return patterns


def detect_market_regime(df):  # Já está correto
    sma_50 = df['close'].rolling(50).mean()
    sma_200 = df['close'].rolling(200).mean()
    if sma_50.iloc[-1] > sma_200.iloc[-1]:
        return 'bull'
    elif sma_50.iloc[-1] < sma_200.iloc[-1]:
        return 'bear'
    return 'sideways'