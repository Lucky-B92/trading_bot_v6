import pandas as pd
import json
from datetime import datetime
import sqlite3
from .config import config  # Importa a instância configurada
import openai

class PerformanceAnalyzer:
    def __init__(self):
        self.conn = sqlite3.connect(config.DATABASE_PATH)
    
    def get_trades_data(self, days_back=30):
        """Obtém trades dos últimos N dias com análise técnica"""
        query = f'''
        SELECT 
            t.symbol, 
            t.open_time, 
            t.close_time, 
            t.open_price, 
            t.close_price,
            t.amount, 
            t.side, 
            t.pnl,
            t.status, 
            t.ml_score,
            t.close_reason,
            t.strategy,
            h.rsi,
            h.macd,
            h.ema_20
        FROM trades t
        JOIN (
            SELECT symbol, MAX(open_time) as last_time 
            FROM trades 
            GROUP BY symbol
        ) last ON t.symbol = last.symbol AND t.open_time = last.last_time
        JOIN historical_analysis h ON t.symbol = h.symbol AND t.open_time = h.date
        WHERE t.open_time >= datetime('now', '-{days_back} days')
        '''
        return pd.read_sql(query, self.conn)
    
    def generate_performance_report(self):
        """Gera relatório detalhado de performance"""
        df = self.get_trades_data()
        
        if df.empty:
            return {"error": "No trades data available"}
        
        closed_trades = df[df['status'] == 'closed']
        
        # Cálculo de métricas avançadas
        report = {
            'summary': {
                'total_trades': len(df),
                'closed_trades': len(closed_trades),
                'win_rate': len(closed_trades[closed_trades['pnl'] > 0]) / len(closed_trades) if not closed_trades.empty else 0,
                'avg_pnl': closed_trades['pnl'].mean() if not closed_trades.empty else 0,
                'sharpe_ratio': self._calculate_sharpe(closed_trades['pnl']) if not closed_trades.empty else 0,
            },
            'by_symbol': df.groupby('symbol').apply(self._symbol_stats).to_dict(),
            'time_analysis': self._time_analysis(df)
        }
        
        return report
    
    def generate_gpt_insights(self):
        """Gera insights estratégicos usando OpenAI GPT"""
        report = self.generate_performance_report()
        trades_sample = self.get_trades_data().head(3).to_dict('records')
        
        prompt = {
            "role": "system",
            "content": f'''
            Analise este relatório de trading e forneça recomendações específicas:
            
            ## Dados de Performance:
            {json.dumps(report['summary'], indent=2)}
            
            ## Trades Recentes (Amostra):
            {json.dumps(trades_sample, indent=2)}
            
            ## Instruções:
            1. Identifique padrões de sucesso/falha
            2. Sugira ajustes nos parâmetros (SL/TP)
            3. Recomende melhorias no modelo de ML
            4. Formate a resposta em markdown
            '''
        }
        
        try:
            response = openai.ChatCompletion.create(
                api_key=config.OPENAI_API_KEY,
                model=config.GPT_MODEL,
                messages=[prompt],
                temperature=config.GPT_TEMPERATURE,
                max_tokens=config.GPT_MAX_TOKENS
            )
            return response.choices[0].message.content
        except Exception as e:
            error_msg = f"Erro na API OpenAI: {str(e)}"
            self._log_error(error_msg)
            return error_msg
    
    # -- Métodos auxiliares -- #
    def _calculate_sharpe(self, returns):
        """Calcula o Índice Sharpe"""
        if len(returns) < 2:
            return 0
        return (returns.mean() / returns.std()) * (252**0.5)  # Annualizado
    
    def _symbol_stats(self, df):
        """Estatísticas por ativo"""
        closed = df[df['status'] == 'closed']
        return {
            'win_rate': len(closed[closed['pnl'] > 0]) / len(closed) if len(closed) > 0 else 0,
            'avg_pnl': closed['pnl'].mean() if len(closed) > 0 else 0,
            'rsi_correlation': closed['pnl'].corr(closed['rsi']) if len(closed) > 1 else 0
        }
    
    def _time_analysis(self, df):
        """Análise por horário do dia"""
        df['hour'] = pd.to_datetime(df['open_time']).dt.hour
        return df.groupby('hour')['pnl'].mean().to_dict()
    
    def _log_error(self, message):
        """Registra erros no banco de dados"""
        self.conn.execute('''
        INSERT INTO logs (timestamp, message, level)
        VALUES (?, ?, ?)
        ''', (datetime.now(), message, 'error'))
        self.conn.commit()
    
    def __del__(self):
        self.conn.close()