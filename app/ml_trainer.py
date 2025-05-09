import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib
from .config import config
from .database import log_message

class MLTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.load_model()
    
    def load_model(self):
        try:
            self.model = joblib.load(config.ML_MODEL_PATH)
            self.scaler = joblib.load(config.SCALER_PATH)
            self.is_scaler_fitted = True
        except:
            log_message("Modelo não encontrado, criando e treinando novo modelo", 'warning')
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.scaler = StandardScaler()
            self.is_scaler_fitted = False
            
            # Treina com dados dummy iniciais
            dummy_X = np.random.rand(10, 5)  # 10 amostras, 5 features
            dummy_y = np.random.randint(0, 2, 10)  # Classes binárias
            self.train_model(dummy_X, dummy_y)
    
    def prepare_features(self, df):
        # Garantir que temos colunas necessárias
        required_cols = ['close', 'high', 'low', 'volume']
        if not all(col in df.columns for col in required_cols):
            raise ValueError("DataFrame está faltando colunas necessárias")
        
        # Calcular indicadores
        df = df.copy()
        
        # EMAs
        df['ema8'] = df['close'].ewm(span=config.EMA_SHORT_PERIOD, adjust=False).mean()
        df['ema21'] = df['close'].ewm(span=config.EMA_LONG_PERIOD, adjust=False).mean()
        df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
        df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
        
        # ATR (para volatilidade)
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        # OBV
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).cumsum()
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(14).mean()
        avg_loss = loss.rolling(14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Volume
        df['volume_ma'] = df['volume'].rolling(config.VOLUME_MA_PERIOD).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Suporte
        support = df['low'].rolling(config.SUPPORT_PERIOD).min().iloc[-1]
        
        # Criar DataFrame de features
        features = pd.DataFrame({
            'rsi': df['rsi'],
            'macd': df['macd'],
            'macd_signal': df['macd_signal'],
            'ema_cross': (df['ema8'] > df['ema21']).astype(int),
            'rsi_ok': ((df['rsi'] > config.RSI_BUY_MIN) & (df['rsi'] < config.RSI_BUY_MAX)).astype(int),
            'volume_spike': (df['volume_ratio'] > config.VOLUME_SPIKE_THRESHOLD).astype(int),
            'bullish_engulfing': ((df['close'].shift(1) < df['open'].shift(1)) & 
                                (df['close'] > df['open']) & 
                                (df['close'] > df['open'].shift(1)) & 
                                (df['open'] < df['close'].shift(1))).astype(int),
            'above_support': (df['close'] > support).astype(int),
            'atr_strength': (df['atr'] > df['atr'].rolling(50).mean()).astype(int),
        })
        
        # Ajustar os índices para garantir que sejam idênticos
        last_index = df.index[-1]
        features = features.reindex([last_index]).fillna(0)

        return features

    
    # ... (restante do código permanece igual)
    
    def train_model(self, X, y):
        try:
            # Dividir dados
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Converter para numpy array se for DataFrame
            if hasattr(X_train, 'values'):
                X_train = X_train.values
                X_test = X_test.values
            
            # Normalizar features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Treinar modelo
            self.model.fit(X_train_scaled, y_train)
            
            # Avaliar
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled)
            
            train_acc = accuracy_score(y_train, train_pred)
            test_acc = accuracy_score(y_test, test_pred)
            
            log_message(
                f"Modelo treinado - Acurácia: Treino={train_acc:.2f}, Teste={test_acc:.2f}",
                'info'
            )
            
            # Marcar scaler como treinado
            self.is_scaler_fitted = True
            
            # Salvar modelo
            joblib.dump(self.model, config.ML_MODEL_PATH)
            joblib.dump(self.scaler, config.SCALER_PATH)
            
            return True
        except Exception as e:
            log_message(f"Falha no treinamento: {str(e)}", 'error')
            self.is_scaler_fitted = False
            return False
    
    def predict(self, features):
        if self.model is None or self.scaler is None:
            self.load_model()
        
        # Garantir que temos exatamente 5 features na ordem correta
        expected_features = ['rsi', 'macd', 'macd_signal', 'ema_cross', 'atr_strength']
        features = features[expected_features].values  # Adicione .values para converter para numpy array
        
        if not hasattr(self, 'is_scaler_fitted') or not self.is_scaler_fitted:
            # Fallback: treina com dados mínimos com 5 features
            dummy_data = np.zeros((1, 5))  # 1 amostra, 5 features
            self.scaler.fit(dummy_data)
            self.is_scaler_fitted = True
            log_message("Scaler não treinado - usando fallback com 5 features", 'warning')
        
        features_scaled = self.scaler.transform(features)
        return float(self.model.predict_proba(features_scaled)[0, 1])  # Converta para float explícito

def predict_with_model(df):
    trainer = MLTrainer()
    features = trainer.prepare_features(df)
    return trainer.predict(features)