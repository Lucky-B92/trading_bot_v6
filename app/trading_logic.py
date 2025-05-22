from binance.client import Client
from datetime import datetime, time
import pandas as pd
import numpy as np
import time as t
import threading
from .config import config
from .database import log_message, get_db_connection
from .ml_trainer import predict_with_model
from .pattern_detector import detect_patterns
from concurrent.futures import ThreadPoolExecutor

class TradingBot:
    def __init__(self):
        self.client = Client(config.BINANCE_API_KEY, config.BINANCE_SECRET_KEY)
        if config.TESTNET:
            self.client.API_URL = 'https://testnet.binance.vision/api'
        
        self.active = False
        self.last_cycle = None
        self.current_operations = 0
        self.last_log = ""

    def log(self, message, level='info', save_db=True):
        """Sistema de logging unificado"""
        timestamp = datetime.now().strftime('[%Y-%m-%d %H:%M:%S]')
        log_msg = f"{timestamp} {message}"
        print(log_msg)
        
        if save_db:
            log_message(message, level)
        
        self.last_log = log_msg
        return log_msg

    def get_market_data(self, symbol, interval='4h', limit=100):
        try:
            self.log(f"Obtendo dados de mercado para {symbol}...")
            
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])
            
            # Conversão de tipos
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Cálculo de indicadores
            df['rsi'] = self._calculate_rsi(df['close'])
            df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
            
            self.log(f"Dados de {symbol} processados. RSI: {df['rsi'].iloc[-1]:.2f}, MACD: {df['macd'].iloc[-1]:.4f}")
            return df
            
        except Exception as e:
            self.log(f"Erro ao obter dados para {symbol}: {str(e)}", 'error')
            return None
        
    def get_confirmation_data(self, symbol):
        """
        Obtém os dados de confirmação no timeframe diário.
        """
        try:
            daily_data = self.get_market_data(symbol, interval=config.CONFIRMATION_TIMEFRAME, limit=8)

            if daily_data is not None and not daily_data.empty:
                # Verificação proativa para valores NaN ou inconsistentes
                if daily_data['close'].isna().sum() > 0:
                    self.log(f"Dados incompletos ou inválidos para {symbol} no timeframe diário.", 'warning')
                    return {'ema8_daily': 'N/A', 'rsi_daily': 'N/A'}

                # Cálculo dos indicadores de confirmação
                ema8_daily = daily_data['close'].ewm(span=8, adjust=False).mean().iloc[-1]
                rsi_daily = self._calculate_rsi(daily_data['close'], period=10).iloc[-1]

                # Verificação proativa para NaN após os cálculos
                if pd.isna(ema8_daily) or pd.isna(rsi_daily):
                    self.log(f"Dados de confirmação contêm NaN para {symbol}.", 'warning')
                    return {'ema8_daily': 'N/A', 'rsi_daily': 'N/A'}

                return {'ema8_daily': ema8_daily, 'rsi_daily': rsi_daily}

            self.log(f"Dados diários não encontrados para {symbol}.", 'warning')
            return {'ema8_daily': 'N/A', 'rsi_daily': 'N/A'}

        except Exception as e:
            self.log(f"Erro ao obter dados de confirmação para {symbol}: {str(e)}", 'error')
            return {'ema8_daily': 'N/A', 'rsi_daily': 'N/A'}



    def _calculate_rsi(self, close_series, period=14):
        delta = close_series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_fibonacci_levels(self, swing_low, swing_high):
        """Calcula os níveis Fibonacci para um movimento dado."""
        levels = {
            '0.618': swing_low + (swing_high - swing_low) * 0.618,
            '1.618': swing_low + (swing_high - swing_low) * 1.618,
            '2.618': swing_low + (swing_high - swing_low) * 2.618,
            '3.618': swing_low + (swing_high - swing_low) * 3.618
        }
        return levels


    def _calculate_macd(self, close_series, fast=12, slow=26, signal=9):
        ema_fast = close_series.ewm(span=fast, adjust=False).mean()
        ema_slow = close_series.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        hist = macd - signal_line
        return macd, signal_line, hist

    def _calculate_atr(self, df, period=14):
        """Calcula o ATR (Average True Range) para o dataframe."""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr

    def get_resistance_levels(self, symbol, interval='1d', limit=100):
        """Identifica as resistências com base no timeframe mais longo."""
        df = self.get_market_data(symbol, interval=interval, limit=limit)

        if df.empty:
            return []

        highs = df['high']
        resistances = highs[highs > highs.shift(1)].nlargest(5).tolist()

        # Ordenar em ordem crescente (mais próxima primeiro)
        resistances.sort()

        self.log(f"Resistências encontradas para {symbol}: {resistances}")
        return resistances

    
    def analyze_market_regime(self, df):
        close_prices = df['close']
        sma_50 = close_prices.rolling(50).mean()
        sma_200 = close_prices.rolling(200).mean()
        
        # Adiciona análise de inclinação
        sma_50_slope = sma_50.diff(5).iloc[-1]
        sma_200_slope = sma_200.diff(5).iloc[-1]
        
        if sma_50.iloc[-1] > sma_200.iloc[-1] * 1.03 and sma_50_slope > 0:
            return 'bull'
        elif sma_50.iloc[-1] < sma_200.iloc[-1] * 0.97 and sma_50_slope < 0:
            return 'bear'
        else:
            return 'sideways'
    
    def get_top_assets(self, n=3):
        self.log("Selecionando os melhores ativos...")

        try:
            exchange_info = self.client.get_exchange_info()
            usdt_pairs = [
                s['symbol'] for s in exchange_info['symbols'] 
                if s['quoteAsset'] == 'USDT'
                and s['status'] == 'TRADING'
                and not any(x in s['baseAsset'] for x in ['USD', 'BUSD', 'TUSD', 'DAI'])
            ]

            # Remover tokens wrapped indesejados
            excluded_wrapped = ['WBTCUSDT', 'WBETHUSDT']
            usdt_pairs = [s for s in usdt_pairs if s not in excluded_wrapped]

            try:
                tickers = self.client.get_ticker()
                volume_map = {t['symbol']: float(t['quoteVolume']) for t in tickers}
                usdt_pairs = [s for s in usdt_pairs if volume_map.get(s, 0) > config.MIN_VOLUME_FILTER]
                self.log(f"Pares com volume suficiente: {len(usdt_pairs)}")
            except Exception as e:
                self.log(f"Erro no filtro de volume: {str(e)}", 'warning')

            def analyze_symbol(symbol):
                try:
                    df = self.get_market_data(symbol)
                    if df is None or df.empty:
                        self.log(f"Dados não encontrados para {symbol}. Ignorando ativo.", 'warning')
                        return None

                    price = df['close'].iloc[-1]
                    regime = self.analyze_market_regime(df)
                    patterns = detect_patterns(df)
                    for k, v in patterns.items():
                        df[k] = v  # Adiciona os padrões como colunas no df para o modelo ML

                    ml_score = predict_with_model(df)

                    # Obter dados de confirmação diários
                    confirmation_data = self.get_confirmation_data(symbol)

                    # Estimar risco/retorno com base em SL 2% e TP 5%
                    atr = self._calculate_atr(df).iloc[-1] if not df.empty else 0
                    stop_loss = price * 0.02  # 2% de SL estimado
                    take_profit = price * 0.05  # 5% de TP estimado

                    potential_loss = stop_loss
                    potential_profit = take_profit

                    risk_reward_ratio = potential_profit / potential_loss if potential_loss else 1


                    score = self.calculate_score(
                        rsi=df['rsi'].iloc[-1],
                        macd=df['macd'].iloc[-1],
                        regime=regime,
                        patterns=patterns,
                        ml_score=ml_score,
                        risk_reward_ratio=risk_reward_ratio,
                        ema8_daily=confirmation_data['ema8_daily'],
                        rsi_daily=confirmation_data['rsi_daily']
                    )

                    return {
                        'symbol': symbol,
                        'price': price,
                        'score': score['total'],  # Garante que só o valor numérico vá para frente
                        'score_details': score['details'],  # Se quiser salvar para debug ou log
                        'regime': regime,
                        'rsi': df['rsi'].iloc[-1],
                        'macd': df['macd'].iloc[-1],
                        'patterns': patterns,
                        'ml_score': ml_score,
                        'ema8_daily': confirmation_data['ema8_daily'],
                        'rsi_daily': confirmation_data['rsi_daily']
                    }

                except Exception as e:
                    self.log(f"Erro ao analisar {symbol}: {str(e)}", 'error')
                    return None

            assets_scores = []
            try:
                with ThreadPoolExecutor(max_workers=config.MAX_WORKERS) as executor:
                    results = list(executor.map(analyze_symbol, usdt_pairs))
                    assets_scores = [r for r in results if r is not None]
            except Exception as e:
                self.log(f"Erro no paralelismo: {str(e)} - usando serial", 'error')
                assets_scores = [analyze_symbol(symbol) for symbol in usdt_pairs if analyze_symbol(symbol) is not None]

            top_assets = sorted(assets_scores, key=lambda x: x['score']['total'] if isinstance(x['score'], dict) else x['score'], reverse=True)[:n]
            self.log(f"Top {n} ativos: {[a['symbol'] for a in top_assets]}")
            return top_assets

        except Exception as e:
            self.log(f"Erro crítico em get_top_assets: {str(e)}", 'error')
            return []

            
    def calculate_score(self, rsi, macd, regime, patterns, ml_score, risk_reward_ratio=0, ema8_daily=None, rsi_daily=None):
        # Garantir que todos os valores sejam escalares
        rsi = float(rsi)
        macd = float(macd)
        ml_score = float(ml_score)
        ema8_daily = float(ema8_daily) if ema8_daily is not None else 0.0
        rsi_daily = float(rsi_daily) if rsi_daily is not None else 0.0

        # Normalização dos componentes
        rsi_norm = np.clip((rsi - 30) / (70 - 30), 0, 1)
        macd_norm = np.tanh(macd * 0.1)
        ml_score_norm = ml_score
        # Normalização da proporção risco/recompensa (1:3 ideal)
        risk_reward_norm = min(risk_reward_ratio / 3, 1)
        rsi_daily_norm = np.clip((rsi_daily - 30) / (70 - 30), 0, 1)
        ema8_daily_norm = 1 if ema8_daily > rsi_daily else 0

        # Peso do regime
        regime_score = {
            'bull': 1.0,
            'sideways': 0.5, 
            'bear': 0.0
        }.get(regime, 0.5)
        
        # Padrões de price action
        pattern_checks = {
            'ema_cross': patterns.get('ema_cross', 0),
            'macd_bullish': patterns.get('macd_bullish', 0),
            'rsi_ok': patterns.get('rsi_ok', 0),
            'above_support': patterns.get('above_support', 0),
            'volume_spike': patterns.get('volume_spike', 0),
            'bullish_engulfing': patterns.get('bullish_engulfing', 0),
            'double_bottom': patterns.get('double_bottom', 0),
            'morning_star': patterns.get('morning_star', 0)
        }

        
        # Padrões atendidos
        pattern_score = sum(pattern_checks.values()) / len(pattern_checks)
        
        # Pesos
        weights = {
            'rsi': 0.1,
            'macd': 0.1,
            'regime': 0.15,
            'patterns': 0.25,
            'ml': 0.15,
            'risk_reward': 0.15,
            'ema8_daily': 0.05,
            'rsi_daily': 0.05
        }
        
        # Cálculo final ponderado
        score_details = {
            'rsi': rsi_norm * weights['rsi'] * 100,
            'macd': (macd_norm * 0.5 + 0.5) * weights['macd'] * 100,
            'regime': regime_score * weights['regime'] * 100,
            'patterns': pattern_score * weights['patterns'] * 100,
            'ml': ml_score_norm * weights['ml'] * 100,
            'risk_reward': risk_reward_norm * weights['risk_reward'] * 100,
            'ema8_daily': ema8_daily_norm * weights['ema8_daily'] * 100,
            'rsi_daily': rsi_daily_norm * weights['rsi_daily'] * 100
        }

        total_score = sum(score_details.values())

        return {
            'total': total_score,
            'details': score_details
        }

        
    def execute_trade(self, symbol, side, amount, stop_loss=None, take_profit=None, close_trade=False):
        try:
            # Confirmar o ambiente (TESTNET ou REAL)
            self.log(f"Ambiente: {'TESTNET' if config.TESTNET else 'REAL'}")

            # Obter informações do símbolo
            symbol_info = self.client.get_symbol_info(symbol)
            filters = {f['filterType']: f for f in symbol_info['filters']}
            min_notional = float(filters.get('MIN_NOTIONAL', {}).get('minNotional', 0))
            step_size = float(filters.get('LOT_SIZE', {}).get('stepSize', 1))
            min_qty = float(filters.get('LOT_SIZE', {}).get('minQty', 0))
            max_qty = float(filters.get('LOT_SIZE', {}).get('maxQty', 100))

            # Obter preço atual
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            price = float(ticker['price'])

            # SE FECHAMENTO DE OPERAÇÃO
            if close_trade:
                self.log(f"[FECHAMENTO] Iniciando fechamento da operação para {symbol}", "info")

                try:
                    # Buscar os dados do trade no banco de dados
                    conn = get_db_connection()
                    trade_data = conn.execute('SELECT * FROM trades WHERE symbol = ? AND status = "OPEN"', (symbol,)).fetchone()
                    conn.close()

                    if not trade_data:
                        self.log(f"Nenhuma operação aberta encontrada para {symbol}", "warning")
                        return False

                    trade_data = dict(trade_data)

                    # Realizar a venda
                    order = self.client.order_market_sell(symbol=symbol, quantity=amount)
                    self.log(f"Ordem de fechamento executada: {symbol} - {amount} unidades @ {price:.4f}", "info")

                    # Atualizar o status da operação no banco de dados
                    conn = get_db_connection()
                    conn.execute('''
                        UPDATE trades 
                        SET status = "CLOSED", close_price = ?, close_time = ?, pnl = ? 
                        WHERE id = ?
                    ''', (price, datetime.now(), (price - trade_data['open_price']) * amount, trade_data['id']))
                    conn.commit()
                    conn.close()

                    self.log(f"Trade fechado e atualizado no banco de dados: {symbol} - {amount} unidades @ {price:.4f}")
                    return True

                except Exception as e:
                    self.log(f"Erro ao fechar operação para {symbol}: {str(e)}", "error")
                    return False

            # Obter ATR e Resistências
            df = self.get_market_data(symbol, interval='1d', limit=100)
            atr = self._calculate_atr(df).iloc[-1] if not df.empty else None
            resistances = self.get_resistance_levels(symbol, interval='1d', limit=100)

            # Cálculo de Fibonacci com base no Swing High/Low
            swing_high = df['high'].max()
            swing_low = df['low'].min()
            fib_levels = self.calculate_fibonacci_levels(swing_low, swing_high)

            self.log(f"Níveis Fibonacci para {symbol}: {fib_levels}")

            # Calcular o amount em unidades com base no valor em USDT
            amount_in_usdt = config.TRADE_AMOUNT
            amount = round(amount_in_usdt / price, 8)
            notional = amount * price

            # Obter dados de confirmação no timeframe diário
            confirmation_data = self.get_confirmation_data(symbol)

            # Calcular ATR para estimativa de duração
            df = self.get_market_data(symbol, interval='1d', limit=20)
            atr = self._calculate_atr(df).iloc[-1] if not df.empty else None

            # Estimar a duração com base no ATR
            expected_duration = "N/A"
            if atr:
                target_distance = abs(take_profit - price) if take_profit else 0
                stop_distance = abs(price - stop_loss) if stop_loss else 0

                # Estimativa em dias
                if target_distance > 0:
                    expected_duration = round(target_distance / atr, 1)
                elif stop_distance > 0:
                    expected_duration = round(stop_distance / atr, 1)

                self.log(f"Estimativa de Duração: {expected_duration} dias")

            # Ajustar a quantidade para respeitar o step size
            amount_adjusted = round(amount - (amount % step_size), 8)

            expected_duration = "N/A"

            if atr:
                # Usar a resistência mais próxima acima do preço atual como alvo potencial
                next_resistance = next((res for res in resistances if res > price), None)

                # Inicializar potential_profit e potential_loss
                potential_loss = (price - stop_loss) * amount_adjusted if stop_loss else "N/A"
                potential_profit = atr * amount_adjusted  # Fallback inicial

                # Verificar se há resistência próxima
                if next_resistance:
                    potential_profit = (next_resistance - price) * amount_adjusted

                # Ajustar com Fibonacci
                fib_target = max(fib_levels['1.618'], fib_levels['2.618'])
                if fib_target > price:
                    potential_profit = max(potential_profit, (fib_target - price) * amount_adjusted)

                # Cálculo corrigido da Proporção Risco/Recompensa
                risk_reward_ratio = 0
                if potential_loss != "N/A" and potential_loss != 0:
                    risk_reward_ratio = potential_profit / potential_loss

                # Proteção contra score injustamente penalizado
                if risk_reward_ratio == 0:
                    risk_reward_ratio = 1



                # Verificação da proporção 1:3
                if potential_loss != "N/A" and potential_profit < potential_loss * 3:
                    self.log(f"[ALERTA] Potencial de lucro ({potential_profit}) é inferior à proporção 1:3 em relação à perda ({potential_loss}). Trade não realizado.", "warning")
                    return False

                self.log(f"Potencial de Lucro: {potential_profit} USDT | Potencial de Perda: {potential_loss} USDT")


                # Estimativa de Duração usando ATR
                target_distance = abs(next_resistance - price) if next_resistance else atr
                expected_duration = round(target_distance / atr, 1) if atr else "N/A"

            self.log(f"Potencial de Lucro: {potential_profit} USDT | Potencial de Perda: {potential_loss} USDT | Duração Estimada: {expected_duration} dias")


            # Atualizar os dados de trade com as informações de confirmação
            # Atualizar os dados de trade com as informações de confirmação e potenciais
            trade_data = {
                'symbol': symbol,
                'open_time': datetime.now(),
                'open_price': price,
                'amount': amount_adjusted,
                'side': side,
                'status': 'OPEN',
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'ema8_daily': confirmation_data['ema8_daily'],
                'rsi_daily': confirmation_data['rsi_daily'],
                'potential_profit': potential_profit,
                'potential_loss': potential_loss,
                'expected_duration': expected_duration,
                'risk_reward_ratio': risk_reward_ratio,
            }


            # Log detalhado das restrições
            self.log(f"[DEBUG] {symbol} - Preço: {price:.4f} | Notional: {notional:.4f} | "
                    f"Min Notional: {min_notional} | Step Size: {step_size} | Min Qty: {min_qty} | Max Qty: {max_qty}")

            # Verificar notional mínimo
            if min_notional and notional < min_notional:
                self.log(f"Ordem rejeitada - Notional ({notional:.4f}) menor que o mínimo permitido ({min_notional})", 'warning')
                return False

            # Verificar quantidade mínima
            if amount_adjusted < min_qty:
                self.log(f"Ordem rejeitada - Quantidade ({amount_adjusted}) menor que o mínimo permitido ({min_qty})", 'warning')
                return False

            # Verificar saldo disponível
            balance_info = self.client.get_asset_balance(asset='USDT')
            available_balance = float(balance_info['free'])
            self.log(f"Saldo disponível: {available_balance} USDT")

            # Verificar se o notional ajustado cabe no saldo disponível
            if amount_adjusted * price > available_balance:
                self.log(f"Saldo insuficiente para {symbol}: Ordem de {amount_adjusted * price:.2f} USDT, "
                        f"Saldo disponível: {available_balance:.2f} USDT", 'warning')
                return False

            # Executar ordem real
            try:
                order = self.client.create_order(
                    symbol=symbol,
                    side=side.upper(),
                    type='MARKET',
                    quantity=amount_adjusted
                )
                self.log(f"ORDEM EXECUTADA: {side} {amount_adjusted} {symbol} @ {price:.4f}")

                # Registrar no banco de dados
                try:
                    conn = get_db_connection()
                    conn.execute(
                        '''INSERT INTO trades 
                        (symbol, side, open_price, amount, stop_loss, take_profit, open_time, status, ema8_daily, rsi_daily, potential_profit, potential_loss, expected_duration, risk_reward_ratio)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                        (
                            trade_data['symbol'],
                            trade_data['side'],
                            trade_data['open_price'],
                            trade_data['amount'],
                            trade_data['stop_loss'],
                            trade_data['take_profit'],
                            trade_data['open_time'],
                            trade_data['status'],
                            trade_data['ema8_daily'],
                            trade_data['rsi_daily'],
                            trade_data['potential_profit'],
                            trade_data['potential_loss'],
                            trade_data['expected_duration'],
                            trade_data['risk_reward_ratio']
                        )
                    )

                    conn.commit()
                    conn.close()

                    self.log(f"Trade registrado no banco de dados: {trade_data}")

                except Exception as e:
                    self.log(f"Erro ao registrar trade no banco de dados: {str(e)}", 'error')


            except Exception as e:
                self.log(f"ERRO ao executar {side} em {symbol}: {str(e)}", 'error')
                return False

        except Exception as e:
            self.log(f"ERRO GERAL no execute_trade() para {symbol}: {str(e)}", 'error')
            return False


    
    def monitor_open_trades(self):
        conn = get_db_connection()
        open_trades = conn.execute('SELECT * FROM trades WHERE status = "OPEN"').fetchall()
        conn.close()
        
        for trade in open_trades:
            trade = dict(trade)
            symbol = trade['symbol']
            side = trade['side']
            open_price = trade['open_price']
            amount = trade['amount']
            stop_loss = trade['stop_loss']
            take_profit = trade['take_profit']

            try:
                # Obter preço atual
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])

                # Cálculo do lucro/prejuízo
                pnl = (current_price - open_price) * amount if side == "BUY" else \
                    (open_price - current_price) * amount
                profit_pct = (current_price - open_price) / open_price * 100 if side == "BUY" else \
                            (open_price - current_price) / open_price * 100

                # Condição de Trailing Stop (Ajuste o stop_loss conforme o preço sobe)
                if profit_pct >= config.TRAILING_START_PERCENT:
                    new_stop_loss = current_price * (1 - config.TRAILING_PERCENT / 100)

                    # Proteção: Trailing Stop nunca pode ser menor que o preço de entrada
                    if new_stop_loss < open_price:
                        new_stop_loss = open_price

                    # Apenas ajusta o stop_loss para cima, nunca para baixo
                    if side == "BUY" and new_stop_loss > stop_loss:
                        self.log(f"[TRAILING STOP] Atualizando Stop Loss para {symbol}: {new_stop_loss:.2f}")
                        stop_loss = new_stop_loss

                    # Update no banco de dados
                    conn = get_db_connection()
                    conn.execute('''
                        UPDATE trades 
                        SET stop_loss = ?
                        WHERE symbol = ? AND status = "OPEN"
                    ''', (stop_loss, symbol))
                    conn.commit()
                    conn.close()

                # Condição para Lock-In Profits (travar lucros)
                if profit_pct >= config.LOCK_IN_PROFIT_PERCENT:
                    lock_in_stop = current_price * (1 - config.LOCK_IN_STOP_PERCENT / 100)

                    # Proteção: Lock-in Stop nunca pode ser menor que o preço de entrada
                    if lock_in_stop < open_price:
                        lock_in_stop = open_price

                    if lock_in_stop > stop_loss:
                        self.log(f"[LOCK-IN PROFITS] Travando lucros em {symbol}: {lock_in_stop:.2f}")
                        stop_loss = lock_in_stop

                    # Update no banco de dados
                    conn = get_db_connection()
                    conn.execute('''
                        UPDATE trades 
                        SET stop_loss = ?
                        WHERE symbol = ? AND status = "OPEN"
                    ''', (stop_loss, symbol))
                    conn.commit()
                    conn.close()

                # Verificar se o stop_loss foi atingido
                if side == "BUY" and current_price <= stop_loss:
                    self.log(f"STOP LOSS ATIVADO: {symbol} @ {current_price:.2f}")
                    self.execute_trade(symbol, "SELL", amount)

                elif side == "SELL" and current_price >= stop_loss:
                    self.log(f"STOP LOSS ATIVADO (SHORT): {symbol} @ {current_price:.2f}")
                    self.execute_trade(symbol, "BUY", amount)

            except Exception as e:
                self.log(f"ERRO ao monitorar trade {symbol}: {str(e)}", 'error')


    def update_equity(self):
        """Atualiza ou insere o equity do dia atual na tabela equity_history."""
        try:
            # Obter saldo disponível em USDT
            balance_info = self.client.get_asset_balance(asset='USDT')
            available_balance = float(balance_info['free'])

            # Obter operações abertas
            conn = get_db_connection()
            open_trades = conn.execute('SELECT * FROM trades WHERE status = "OPEN"').fetchall()
            conn.close()

            # Calcular o valor atualizado das operações abertas
            invested_amount = 0.0
            for trade in open_trades:
                symbol = trade['symbol']
                amount = float(trade['amount'])

                try:
                    # Obter o preço atual do ativo
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                except Exception:
                    current_price = float(trade['open_price'])  # Fallback

                # Valor da posição é quantidade * preço atual
                invested_amount += current_price * amount

            # Equity é a soma do saldo disponível + valor investido
            total_equity = available_balance + invested_amount

            # Data de hoje em formato YYYY-MM-DD
            today = datetime.now().strftime('%Y-%m-%d')

            # Verificar se já existe um registro para o dia atual
            conn = get_db_connection()
            existing_record = conn.execute(
                'SELECT id FROM equity_history WHERE DATE(date) = ?', (today,)
            ).fetchone()

            if existing_record:
                # Atualizar o registro existente
                conn.execute('''
                    UPDATE equity_history 
                    SET equity = ?, available_balance = ?, date = ? 
                    WHERE id = ?
                ''', (total_equity, available_balance, datetime.now(), existing_record['id']))
                self.log(f"Equity atualizado para hoje: {total_equity:.2f} USDT", "info")

            else:
                # Inserir um novo registro
                conn.execute('''
                    INSERT INTO equity_history (date, equity, available_balance) 
                    VALUES (?, ?, ?)
                ''', (datetime.now(), total_equity, available_balance))
                self.log(f"Novo registro de equity criado para hoje: {total_equity:.2f} USDT", "info")

            conn.commit()
            conn.close()

        except Exception as e:
            self.log(f"Erro ao atualizar equity: {str(e)}", "error")


    
    def run_cycle(self):
        while self.active:
            try:
                start_time = datetime.now()
                self.log(f"\n--- INÍCIO DO CICLO {start_time} ---")
                
                # 1. Selecionar melhores ativos
                top_assets = self.get_top_assets(n=config.MAX_SIMULTANEOUS_TRADES)
                
                # 2. Executar estratégia para cada ativo
                for asset in top_assets:
                    if self.current_operations >= config.MAX_SIMULTANEOUS_TRADES:
                        self.log("Limite máximo de operações atingido")
                        break
                    
                    patterns = asset['patterns']
                    
                    # Calcular o score composto
                    score_result = asset['score'] if isinstance(asset['score'], dict) else {'total': asset['score'], 'details': {}}
                    score = score_result['total']
                    score_threshold = config.SCORE_THRESHOLD
                    conditions_met = score >= score_threshold

                    self.log(f"[SCORE] {asset['symbol']} - Score: {score:.2f} | Threshold: {score_threshold} | Conditions Met: {conditions_met}")


                    
                    if conditions_met:
                        stop_loss = asset['price'] * (1 - config.STOP_LOSS)
                        take_profit = asset['price'] * (1 + config.TAKE_PROFIT)
                        
                        if self.execute_trade(
                            symbol=asset['symbol'],
                            side='BUY',
                            amount=config.TRADE_AMOUNT,
                            stop_loss=stop_loss,
                            take_profit=take_profit
                        ):
                            self.current_operations += 1
                
                # 3. Monitorar trades abertos
                self.monitor_open_trades()

                # 4. Log dos Top 3 ativos ao final do ciclo
                if top_assets:
                    self.log(f"Top {len(top_assets)} ativos: {[a['symbol'] for a in top_assets]}")
                    
                    for asset in top_assets:
                        patterns = asset['patterns']
                        ml_score = asset['ml_score']
                        ml_score_check = ml_score > 0.6

                        # Usar o score já calculado (com RR)
                        score_result = asset['score'] if isinstance(asset['score'], dict) else {'total': asset['score'], 'details': {}}
                        score = score_result['total']
                        details = score_result.get('details', {})

                        score_threshold = config.SCORE_THRESHOLD
                        conditions_met = score >= score_threshold

                        log_message = (
                            f"[SCORE] {asset['symbol']} - Score: {score:.2f} | Threshold: {score_threshold} | Conditions Met: {conditions_met} | "
                            f"RSI: {details.get('rsi', 0):.1f} | MACD: {details.get('macd', 0):.1f} | Regime: {details.get('regime', 0):.1f} | "
                            f"Patterns: {details.get('patterns', 0):.1f} | ML: {details.get('ml', 0):.1f} | RR: {details.get('risk_reward', 0):.1f} | "
                            f"EMA8 Daily: {details.get('ema8_daily', 0):.1f} | RSI Daily: {details.get('rsi_daily', 0):.1f}"
                        )

                        self.log(log_message)

                
                # Atualizar equity no banco de dados
                self.update_equity()

                # 5. Finalizar ciclo
                self.last_cycle = datetime.now()  # Atualiza o último ciclo ANTES do cálculo de duração
                cycle_duration = (datetime.now() - start_time).total_seconds()
                self.log(f"--- FIM DO CICLO (duração: {cycle_duration:.2f}s) ---\n")

                t.sleep(config.CYCLE_INTERVAL)

                
            except Exception as e:
                self.log(f"ERRO NO CICLO: {str(e)}", 'error')
                t.sleep(60)
    
    def start(self):
        self.active = True
        self.thread = threading.Thread(target=self.run_cycle)
        self.thread.daemon = True
        self.thread.start()
        self.log("Bot iniciado com sucesso", 'info')
    
    def stop(self):
        self.active = False
        self.log("Bot parado", 'info')