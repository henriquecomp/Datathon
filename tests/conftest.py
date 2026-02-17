import sys
import os

# Obtém o caminho absoluto para a pasta 'src'
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))

# Adiciona ao sys.path se ainda não estiver lá
if src_path not in sys.path:
    sys.path.insert(0, src_path)