import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.dkb.client_sqlite import DKBClient

dkb = DKBClient("dkb.sqlite")
print("Architectures:\n", dkb.query_architectures())
dkb.close()
