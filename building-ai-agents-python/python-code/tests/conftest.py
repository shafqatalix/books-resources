import os
import sys
from pathlib import Path

# Make `import lib` work when pytest runs from anywhere, and give lib.config a model name.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("MODEL", "test:none")
