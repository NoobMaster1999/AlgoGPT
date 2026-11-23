"""
Convenience entry point to run the live data and Kronos forecast update pipeline.
Usage:
    python run_update.py
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from algogpt.update_live_data import main

if __name__ == "__main__":
    main()
