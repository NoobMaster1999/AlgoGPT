"""
AlgoGPT - Quick Start Script
=============================
Run this to test the installation and run a quick demo.
"""
import sys
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required packages are installed."""
    required = [
        "torch",
        "pandas",
        "numpy",
        "yfinance",
        "huggingface_hub",
        "einops",
        "tqdm",
    ]

    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)

    if missing:
        print(f"Missing packages: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    return True


def run_quick_test():
    """Run a quick test with Yahoo Finance data."""
    print("=" * 60)
    print("AlgoGPT + Kronos Quick Test")
    print("=" * 60)

    # This will use Yahoo Finance to fetch NIFTY data
    # and run a quick forecast with the pretrained Kronos model
    from algogpt.main import run_trading_pipeline

    try:
        results = run_trading_pipeline(
            symbol="NIFTY",
            start="2023-01-01",
            end="2024-06-01",
            interval="1d",  # Daily data for quick test
            data_source="yahoo",
            model_id="NeoQuasar/Kronos-small",
            tokenizer_id="NeoQuasar/Kronos-Tokenizer-base",
            pred_len=5,
            sample_count=1,
            lookback=100,
            plot=False,
            verbose=True,
        )
        print("\n✓ Quick test completed successfully!")
        return True
    except Exception as e:
        print(f"\n✗ Quick test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)

    print("Dependencies OK")
    print()

    success = run_quick_test()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()