from flask import render_template, jsonify, request
from datetime import datetime, time
from . import app  # Importa o app Flask
from .config import config
from .database import get_db_connection, update_setting
from .trading_logic import TradingBot

# Inicialização do Bot (se necessário)
bot = TradingBot()
bot_thread = None

# Variável global para status do bot
bot_status = {
    'active': False,
    'last_cycle': None,
    'current_operations': 0,
    'blocked_time': False
}

def register_routes(flask_app):
    """Registra todas as rotas no aplicativo Flask"""

    # --- Rotas Principais ---
    @flask_app.route('/')
    def dashboard():
        return render_template('dashboard.html')
    
    @flask_app.route('/settings')
    def settings():
        """Renderiza a página de configurações."""
        return render_template('settings.html')


    # --- Rotas de Status ---
    @flask_app.route('/get_status')
    def get_status():

        # Contar operações ativas no banco de dados
        conn = get_db_connection()
        active_trades = conn.execute('SELECT COUNT(*) FROM trades WHERE status = "OPEN"').fetchone()[0]
        conn.close()
        bot_status['current_operations'] = active_trades

        # Indicar blocked time
        now = datetime.now().time()
        blocked = config.BLOCK_START_TIME <= now <= config.BLOCK_END_TIME
        bot_status['blocked_time'] = blocked

        # Novo campo last_cycle, sem impactar o campo 'status'
        last_cycle = bot.last_cycle.strftime('%Y-%m-%d %H:%M:%S') if bot.last_cycle else 'N/A'

        return jsonify({
            'status': bot_status,  # Mantém a estrutura intacta
            'blocked': blocked,
            'last_cycle': last_cycle,  # Novo campo
            'settings': {
                'block_start': config.BLOCK_START_TIME.strftime('%H:%M'),
                'block_end': config.BLOCK_END_TIME.strftime('%H:%M')
            }
        })


    @flask_app.route('/toggle_bot', methods=['POST'])
    def toggle_bot():
        data = request.get_json()
        bot_status['active'] = data.get('active', False)
        
        if bot_status['active']:
            bot.start()
        else:
            bot.stop()
        
        return jsonify({'success': True})

    # --- Rotas de Trading ---
    @flask_app.route('/manual_sell', methods=['POST'])
    def manual_sell():
        data = request.get_json()
        trade_id = data.get('trade_id')
        reason = data.get('reason', 'Manual sell')
        
        try:
            success = bot.manual_sell(trade_id, reason)
            return jsonify({'success': success})
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400

    # --- Rotas de Dados ---
    @flask_app.route('/get_trades')
    def get_trades():
        status = request.args.get('status')
        conn = get_db_connection()
        
        # Verificar operações abertas
        if status == "open":
            trades = conn.execute('SELECT * FROM trades WHERE status = "OPEN"').fetchall()
        elif status == "closed":
            trades = conn.execute('SELECT * FROM trades WHERE status = "CLOSED"').fetchall()
        else:
            trades = conn.execute('SELECT * FROM trades').fetchall()
        
        conn.close()

        trade_data = []
        for trade in trades:
            trade_dict = dict(trade)

            try:
                # Obter preço atual
                ticker = bot.client.get_symbol_ticker(symbol=trade_dict['symbol'])
                current_price = float(ticker['price'])
                open_price = trade_dict['open_price']
                amount = trade_dict['amount']

                # Cálculo do lucro/prejuízo
                pnl = (current_price - open_price) * amount if trade_dict['side'] == "BUY" else \
                    (open_price - current_price) * amount

                profit_pct = ((current_price - open_price) / open_price) * 100 if trade_dict['side'] == "BUY" else \
                            ((open_price - current_price) / open_price) * 100

                # Atualizar dicionário
                trade_dict['current_price'] = round(current_price, 4)
                trade_dict['pnl'] = round(pnl, 2)
                trade_dict['profit_pct'] = round(profit_pct, 2)

            except Exception as e:
                trade_dict['current_price'] = 'N/A'
                trade_dict['pnl'] = 'N/A'
                trade_dict['profit_pct'] = 'N/A'

            # Incluir os novos campos
            trade_dict['potential_profit'] = trade_dict.get('potential_profit', 'N/A')
            trade_dict['potential_loss'] = trade_dict.get('potential_loss', 'N/A')
            trade_dict['expected_duration'] = trade_dict.get('expected_duration', 'N/A')
            trade_dict['stop_loss'] = trade_dict.get('stop_loss', 'N/A')

            trade_data.append(trade_dict)

        return jsonify(trade_data)



    @flask_app.route('/close_trade', methods=['POST'])
    def close_trade():
        data = request.get_json()
        trade_id = data.get('id')

        try:
            conn = get_db_connection()
            trade = conn.execute('SELECT * FROM trades WHERE id = ? AND status = "OPEN"', (trade_id,)).fetchone()

            if trade:
                trade = dict(trade)
                symbol = trade['symbol']
                amount = trade['amount']

                # Realizar a venda
                bot.execute_trade(symbol=symbol, side='SELL', amount=amount, close_trade=True)


                # Atualizar o status da operação
                conn.execute('UPDATE trades SET status = "CLOSED" WHERE id = ?', (trade_id,))
                conn.commit()

                return jsonify({'success': True})

            return jsonify({'success': False, 'error': 'Trade não encontrado ou já fechado'})

        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500



    @flask_app.route('/get_equity_history')
    def get_equity_history():
        conn = get_db_connection()
        history = conn.execute('''
            SELECT date, equity, available_balance 
            FROM equity_history 
            ORDER BY date
        ''').fetchall()
        conn.close()
        return jsonify([dict(record) for record in history])

    # --- Rotas de Configuração ---
    @flask_app.route('/get_settings')
    def get_settings():
        return jsonify({
            'risk': {
                'stop_loss': config.STOP_LOSS * 100,
                'take_profit': config.TAKE_PROFIT * 100,
                'max_capital': config.MAX_CAPITAL_USAGE * 100,
                'enable_short': config.ENABLE_SHORT
            },
            'schedule': {
                'block_start': config.BLOCK_START_TIME.strftime('%H:%M'),
                'block_end': config.BLOCK_END_TIME.strftime('%H:%M')
            },
            'model': {
                'ema_short': config.EMA_SHORT_PERIOD,
                'rsi': config.RSI_PERIOD,
                'macd_fast': config.MACD_FAST,
                'macd_slow': config.MACD_SLOW,
                'macd_signal': config.MACD_SIGNAL
            },
            'timeframes': {
                'primary': config.PRIMARY_TIMEFRAME,
                'confirmation': config.CONFIRMATION_TIMEFRAME
            },
            'gpt': {
                'model': config.GPT_MODEL,
                'temperature': config.GPT_TEMPERATURE
            }
        })


    @flask_app.route('/update_settings', methods=['POST'])
    def update_settings():
        data = request.get_json()
        setting_type = data.get('type')
        new_settings = data.get('settings', {})
        
        try:
            
            if setting_type == 'risk':
                for key in ['stop_loss', 'take_profit', 'max_capital']:
                    if key in new_settings:
                        update_setting(key.upper(), str(new_settings[key] / 100))
                if 'enable_short' in new_settings:
                    update_setting('ENABLE_SHORT', str(new_settings['enable_short']).lower())
            
            elif setting_type == 'schedule':
                for key in ['block_start', 'block_end']:
                    if key in new_settings:
                        update_setting(f"BLOCK_{key.upper()}", new_settings[key])
            
            elif setting_type == 'model':
                for key in ['ema_short_period', 'rsi_period', 'macd_fast', 'macd_slow', 'macd_signal']:
                    if key in new_settings:
                        update_setting(key.upper(), str(new_settings[key]))

            elif setting_type == 'timeframes':
                for key in ['primary', 'confirmation']:
                    if key in new_settings:
                        update_setting(f"{key.upper()}_TIMEFRAME", new_settings[key])

            
            config.reload_from_db()
            return jsonify({'success': True})
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        
    
    @flask_app.route('/update_strategy', methods=['POST'])
    def update_strategy():
        data = request.get_json()
        strategy = data.get('strategy', 'swing_trade')
        
        try:
            update_setting('STRATEGY', strategy)
            return jsonify({'success': True, 'strategy': strategy})
        
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 400


    # --- Rotas de Análise ---
    @app.route('/get_insights')
    def get_insights():
        try:
            conn = get_db_connection()
            insights = conn.execute('SELECT * FROM insights ORDER BY timestamp DESC LIMIT 10').fetchall()
            conn.close()
            return jsonify([dict(i) for i in insights])
        except Exception as e:
            app.logger.error(f"Erro em /get_insights: {str(e)}")
            return jsonify({"error": "Internal server error"}), 500
        
    # --- Rotas de Logs ---
    @flask_app.route('/get_system_logs')
    def get_system_logs():
        try:
            conn = get_db_connection()
            logs = conn.execute('''
                SELECT timestamp, level AS type, message 
                FROM logs 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''').fetchall()
            conn.close()
            return jsonify([dict(log) for log in logs])
        except Exception as e:
            print(f"Erro ao buscar logs: {str(e)}")
            return jsonify({"error": "Falha ao carregar logs"}), 500
        
        
        # Inicializa a thread do bot
    global bot_thread
    if bot_thread is None:
        import threading
        bot_thread = threading.Thread(target=bot.run_cycle, daemon=True)
        bot_thread.start()