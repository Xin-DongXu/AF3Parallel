"""Console-script entry points with correct initialization per command."""

from __future__ import annotations

import signal


def run_parallel() -> int:
    from af3parallel.parallel import main, signal_handler

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
    return 0


def run_gpu_memory_profiler() -> int:
    from af3parallel.gpu_memory_profiler import main

    main()
    return 0


def run_gpu_memory_timeseries_profiler() -> int:
    from af3parallel.gpu_memory_timeseries_profiler import main

    main()
    return 0


def run_gpu_time_estimate() -> int:
    from af3parallel.gpu_time_estimate import main

    result = main()
    return int(result) if result is not None else 0


def run_cpu_time_estimate() -> int:
    from af3parallel.cpu_time_estimate import main

    result = main()
    return int(result) if result is not None else 0


def run_json_integrator() -> int:
    from af3parallel.json_integrator import main

    result = main()
    return int(result) if result is not None else 0


def run_gpu_monitor() -> int:
    from af3parallel.gpu_monitor import main

    main()
    return 0
