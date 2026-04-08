import sys
import os

# Add parent directory to path to find the actual server.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app
