import sqlite3
import shutil
from datetime import datetime
import os
from .config import config
import time
from contextlib import contextmanager

# Diretório de backups
BACKUP_DIR = 'backups'
os.makedirs(BACKUP_DIR, exist_ok=True)

def get_db_connection():
    """Retorna uma nova conexão com o banco de dados"""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_connection():
    """Gerenciador de contexto seguro para conexões com o banco"""
    conn = None
    try:
        conn = get_db_connection()
        conn.execute("PRAGMA foreign_keys = ON")
        yield conn
    except sqlite3.Error as e:
        print(f"Erro na conexão com o banco: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def check_db_tables():
    """Verifica se todas as tabelas necessárias existem"""
    required_tables = {'trades', 'logs', 'equity_history', 'settings'}
    max_attempts = 3
    attempt = 0
    
    while attempt < max_attempts:
        try:
            with db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                existing_tables = {row['name'] for row in cursor.fetchall()}
                missing = required_tables - existing_tables
                
                if not missing:
                    return True
                
                print(f"Tabelas faltando: {missing}")
                attempt += 1
                time.sleep(0.5)
                
        except sqlite3.Error as e:
            print(f"Erro na verificação (tentativa {attempt+1}/{max_attempts}): {str(e)}")
            attempt += 1
            time.sleep(1)
    
    return False

def init_db():
    """Inicializa o banco de dados e cria tabelas se não existirem"""
    try:
        os.makedirs(os.path.dirname(config.DATABASE_PATH), exist_ok=True)

        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA synchronous=NORMAL")
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                name TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                open_time DATETIME NOT NULL,
                close_time DATETIME,
                open_price REAL NOT NULL,
                close_price REAL,
                amount REAL NOT NULL,
                side TEXT NOT NULL,
                pnl REAL,
                status TEXT NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit REAL NOT NULL,
                ml_score REAL,
                close_reason TEXT,
                strategy TEXT,
                ema8_daily REAL,
                rsi_daily REAL
            )''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                message TEXT NOT NULL,
                level TEXT NOT NULL,
                details TEXT
            )''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS equity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATETIME NOT NULL,
                equity REAL NOT NULL,
                available_balance REAL NOT NULL
            )''')
            
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                suggestion TEXT NOT NULL,
                source TEXT NOT NULL,
                implemented BOOLEAN DEFAULT 0
            )''')

            cursor.execute('''
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            ''')

            
            default_settings = {
                'STOP_LOSS': '0.05',
                'TAKE_PROFIT': '0.5',
                'MAX_CAPITAL_USAGE': '0.5',
                'ENABLE_SHORT': 'false',
                'BLOCK_START_TIME': '20:00',
                'BLOCK_END_TIME': '02:00',
                'RSI_WEIGHT': '50',
                'MACD_WEIGHT': '50',
                'GPT_MODEL': 'gpt-4',
                'GPT_TEMPERATURE': '0.7'
            }
            
            for name, value in default_settings.items():
                cursor.execute('''
                INSERT OR IGNORE INTO settings (name, value)
                VALUES (?, ?)
                ''', (name, value))
            
            conn.commit()
        
        if not check_db_tables():
            raise RuntimeError("Falha na criação das tabelas")
            
        print("✅ Banco de dados inicializado com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro durante init_db: {str(e)}")
        raise

def backup_database():
    """Cria um backup seguro do banco de dados"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"backup_{timestamp}.db")
    
    try:
        with db_connection() as src:
            with sqlite3.connect(backup_path) as dst:
                src.backup(dst)
        print(f"✅ Backup criado: {backup_path}")
        return backup_path
    except Exception as e:
        print(f"❌ Falha no backup: {str(e)}")
        return None

def rotate_backups(max_backups=5):
    """Mantém apenas os últimos N backups"""
    try:
        backups = sorted([
            f for f in os.listdir(BACKUP_DIR) 
            if f.startswith('backup_') and f.endswith('.db')
        ])
        
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            os.remove(os.path.join(BACKUP_DIR, oldest))
    except Exception as e:
        print(f"❌ Falha na rotação de backups: {str(e)}")

def get_setting(name, default=None):
    """Obtém um valor de configuração"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT value FROM settings WHERE name = ?', (name,))
            result = cursor.fetchone()
            return result['value'] if result else default
    except sqlite3.Error as e:
        print(f"Erro ao obter setting {name}: {str(e)}")
        return default

def get_settings_dict():
    """Retorna todas as configurações como dicionário"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT name, value FROM settings')
            return {row['name']: row['value'] for row in cursor.fetchall()}
    except sqlite3.Error as e:
        print(f"Erro ao obter configurações: {str(e)}")
        return {}

def update_setting(name, value):
    """Atualiza uma configuração"""
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT OR REPLACE INTO settings (name, value)
            VALUES (?, ?)
            ''', (name, str(value)))
            conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Erro ao atualizar setting {name}: {str(e)}")
        return False

def log_message(message, level='INFO', details=None):
    """Registra uma mensagem de log com nível padronizado"""
    level = level.upper()
    valid_levels = ['INFO', 'WARNING', 'ERROR', 'DEBUG']
    if level not in valid_levels:
        level = 'INFO'
    try:
        with db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO logs (timestamp, message, level, details)
            VALUES (?, ?, ?, ?)
            ''', (datetime.now(), message, level, details))
            conn.commit()
        return True
    except sqlite3.Error as e:
        print(f"Erro ao registrar log: {str(e)}")
        return False

def insert_trade(trade_data):
    """Insere um novo trade"""
    try:
        with db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()
            cursor.execute('''
            INSERT INTO trades (
                symbol, open_time, open_price, amount, side, 
                status, stop_loss, take_profit, strategy, ema8_daily, rsi_daily
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_data['symbol'],
                trade_data['open_time'],
                trade_data['open_price'],
                trade_data['amount'],
                trade_data['side'],
                'open',
                trade_data['stop_loss'],
                trade_data['take_profit'],
                trade_data.get('strategy', 'swing_trade'),
                trade_data.get('ema8_daily', None),
                trade_data.get('rsi_daily', None)
            ))
            trade_id = cursor.lastrowid
            conn.commit()
            return trade_id
    except sqlite3.Error as e:
        conn.rollback()
        log_message(f"Erro ao inserir trade: {str(e)}", level='error')
        raise

def update_trade(trade_id, updates):
    """Atualiza um trade existente"""
    try:
        with db_connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            cursor = conn.cursor()
            
            set_clause = ', '.join(f"{k} = ?" for k in updates.keys())
            values = list(updates.values())
            values.append(trade_id)
            
            cursor.execute(f'''
            UPDATE trades SET {set_clause} WHERE id = ?
            ''', values)
            
            conn.commit()
            return True

    except sqlite3.Error as e:
        conn.rollback()
        log_message(f"Erro ao atualizar trade {trade_id}: {str(e)}", level='error')
        return False


if __name__ == '__main__':
    if init_db():
        print("Configurações iniciais:", get_settings_dict())