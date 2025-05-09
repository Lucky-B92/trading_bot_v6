#!/usr/bin/env python
from app import app, init_db
from database import backup_database

if __name__ == '__main__':
    # 1. Inicializa o banco de dados
    init_db()  
    
    # 2. Cria um backup inicial (opcional)
    backup_database()  
    
    # 3. Inicia o servidor Flask com auto-reload
    app.run(
        host='0.0.0.0', 
        port=5000, 
        debug=True,  # Ativa modo debug
        use_reloader=True  # Recarrega automaticamente
    )