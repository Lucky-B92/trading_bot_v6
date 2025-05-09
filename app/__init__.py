from flask import Flask
from .config import config
from .database import init_db, check_db_tables

# Cria a instância do Flask aqui (evita circular import)
app = Flask(__name__)
app.config.from_object(config)


def initialize_app():
    """Função de inicialização verificada"""
    print("🔍 Verificando banco de dados...")
    
    if not check_db_tables():
        print("🔄 Criando estrutura do banco de dados...")
        init_db()
        
        if not check_db_tables():
            raise RuntimeError("❌ Falha crítica: Não foi possível criar tabelas")
    
    print("✅ Banco de dados verificado com sucesso!")

# Importa as rotas DEPOIS de criar o app
from .routes import register_routes
register_routes(app)