import os
from datetime import time
from dotenv import load_dotenv

load_dotenv()

class config:
    # Credenciais da Binance
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    
    # Parâmetros de risco
    STOP_LOSS = 0.05  # 5%
    TAKE_PROFIT = 0.5  # 50%
    MIN_ORDER_VALUE = 15  # USD
    MAX_CAPITAL_USAGE = 0.9  # 90% da banca

    # Número máximo de trades que o bot pode abrir simultaneamente
    MAX_SIMULTANEOUS_TRADES = 3  # ou o valor desejado

    # Configurações de análise de mercado
    MIN_VOLUME_FILTER = 1000000  # 1 milhão USDT de volume mínimo
    MAX_WORKERS = 5  # Número de threads para processamento paralelo

    # Parâmetros de análise técnica
    EMA_SHORT_PERIOD = 8
    EMA_LONG_PERIOD = 21
    VOLUME_MA_PERIOD = 20
    VOLUME_SPIKE_THRESHOLD = 1.2
    RSI_BUY_MIN = 40
    RSI_BUY_MAX = 70
    ML_SCORE_THRESHOLD = 0.6
    SUPPORT_PERIOD = 20

    # Timeframes para Swing Trade
    PRIMARY_TIMEFRAME = '4h'
    CONFIRMATION_TIMEFRAME = '1d'

    # Limite mínimo para o score composto (0 a 100) para executar uma operação
    SCORE_THRESHOLD = 70

    # Horário de bloqueio (20h - 2h)
    BLOCK_START_TIME = time(20, 0)
    BLOCK_END_TIME = time(2, 0)

    # Valor padrão de trade em dólares
    TRADE_AMOUNT = 15  # ou qualquer valor que você queira utilizar

    # Percentual mínimo para iniciar o trailing stop
    TRAILING_START_PERCENT = 5  # Inicia o trailing stop após 5% de lucro
    TRAILING_PERCENT = 2  # Trailing stop a 2% do preço atual

    # Percentual mínimo para travar lucros
    LOCK_IN_PROFIT_PERCENT = 10  # Trava lucros após 10% de lucro
    LOCK_IN_STOP_PERCENT = 5  # Ajusta o stop loss para garantir 5% de lucro


    # Intervalo entre ciclos em segundos
    CYCLE_INTERVAL = 300  # exemplo: 300 segundos = 5 minutos
    
    # Configurações do modelo
    ML_MODEL_PATH = 'app/models/random_forest.pkl'
    SCALER_PATH = 'app/models/scaler.pkl'
    RETRAIN_INTERVAL = 24  # horas
    
    # Configurações do banco de dados
    DATABASE_PATH = 'data/trades.db'
    
    # Configurações da API OpenAI
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    GPT_MODEL = "gpt-4o"
    
    # Outras configurações
    DEBUG = False
    TESTNET = False  # Usar API de teste da Binance

    # Alternar entre modelo antigo e novo (com novas features)
    USE_EXTENDED_ML_FEATURES = True



config = config  # Instância única