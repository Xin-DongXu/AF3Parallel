#!/usr/bin/env python3
"""CLI: af3parallel stats (also runnable as a script)."""
from af3parallel.result_stats import main

if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    raise SystemExit(main())
