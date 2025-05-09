import sys
from pathlib import Path

# 1. Configura caminhos absolutos
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))  # Adiciona a raiz do projeto ao PATH

# 2. Agora importa a aplicação
from app.app import app  # Supondo que sua app Flask está em app/app.py
from app.database import init_db

if __name__ == "__main__":
    init_db()
    app.run(host='0.0.0.0', port=8000)