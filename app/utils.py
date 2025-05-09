from datetime import datetime
import pytz

def is_blocked_time():
    """Verifica se está no horário bloqueado (20h-2h)"""
    now = datetime.now(pytz.utc).time()
    return Config.BLOCK_START_TIME <= now or now <= Config.BLOCK_END_TIME

def calculate_pnl(entry_price, exit_price, amount, side):
    """Calcula PnL de uma operação"""
    if side == 'buy':
        return (exit_price - entry_price) * amount
    else:
        return (entry_price - exit_price) * amount