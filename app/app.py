from .trading_logic import TradingBot

# Inicialização do Bot
bot = TradingBot()

def init_bot():
    """Inicializa a thread do bot"""
    import threading
    bot_thread = threading.Thread(target=bot.run_cycle, daemon=True)
    bot_thread.start()
    return bot_thread