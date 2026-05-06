#!/usr/bin/env python3
"""AlphaFold 3 Multi-GPU Parallel Executor (AF3Parallel).

A profile-driven scheduling framework for high-throughput AlphaFold 3
inference on multi-GPU clusters. The framework comprises four cooperating
components:

  1. Stepwise memory-runtime profile
     TokenMemoryProfileLoader exposes the discrete VRAM-vs-token
     staircase of AF3 as a binary-searchable table. Built-in profiles for
     A800 80 GB and RTX 4090 24 GB are included; users can supply their
     own TSV for any other GPU. Sparse external profiles are filled by
     linear interpolation between adjacent steps.

  2. Multi-anchor temporal wave scheduler
     DualDimensionTaskOptimizer packs tasks into per-GPU batches under
     a joint VRAM and wall-clock budget. On top of the regular Next-Fit
     Decreasing packer, the optimizer can build TemporalWaveBatch
     objects that schedule successive waves of lightweight tasks inside
     the unused VRAM "shadow" of long-running anchor tasks, raising GPU
     utilization toward saturation.

  3. Multi-GPU LPT distribution
     distribute_tasks_by_tokens spreads the global task list across
     GPUs in token-balanced fashion using a min-heap-backed Longest
     Processing Time (LPT) algorithm with O(N log G) complexity.

  4. Resilient AF3 driver
     run_alphafold3_task and run_gpu_worker launch AF3 inside a
     singularity container, monitor per-task peak VRAM via nvidia-smi,
     stream every per-task result to a TSV log immediately on completion,
     retry transient failures individually, and restore staged JSON files
     to the input directory on SIGINT/SIGTERM.

VRAM-overflow tasks (whose predicted footprint exceeds the effective
VRAM) are isolated into single-task batches and routed through AF3's
CPU memory offloading mode, or skipped outright when
--skip-vram-overflow is set.
"""

import os
import sys
import argparse
import time
import subprocess
import threading
import json
import csv
import re
import signal
import shutil
import multiprocessing
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
import itertools
import bisect
import heapq

_NVIDIA_SMI_SEMAPHORE = threading.Semaphore(4)

_TASK_CONCURRENCY_SEMAPHORE: Optional[threading.Semaphore] = None
_TASK_CONCURRENCY_CAP: Optional[int] = None  # for logging/reporting

# ---------------------------------------------
#  Built-in stepwise memory-runtime profiles
#
#  Each entry maps a [min_token, max_token) interval to the peak VRAM
#  (MB) and average end-to-end runtime (s) measured on that GPU. Looked
#  up in O(log S) via bisect over the boundary list. Values for A800 are
#  auto-generated from AF3_A800_80G_All_Len_stat.txt; RTX 4090 values
#  come from the corresponding RTX 4090 24 GB profiling run.
# ---------------------------------------------

MEMORY_PROFILE_STEPS = [
    {"min_token": 0, "max_token": 257, "memory_mb": 2775, "runtime_avg": 80.81},
    {"min_token": 257, "max_token": 513, "memory_mb": 3931, "runtime_avg": 131.93},
    {"min_token": 513, "max_token": 769, "memory_mb": 5467, "runtime_avg": 208.15},
    {"min_token": 769, "max_token": 1025, "memory_mb": 7263, "runtime_avg": 307.22},
    {"min_token": 1025, "max_token": 1281, "memory_mb": 9309, "runtime_avg": 443.78},
    {"min_token": 1281, "max_token": 1537, "memory_mb": 11485, "runtime_avg": 612.15},
    {"min_token": 1537, "max_token": 2053, "memory_mb": 17247, "runtime_avg": 1069.91},
    {"min_token": 2053, "max_token": 2579, "memory_mb": 24929, "runtime_avg": 1727.23},
    {"min_token": 2579, "max_token": 3157, "memory_mb": 32227, "runtime_avg": 2607.97},
    {"min_token": 3157, "max_token": 3599, "memory_mb": 46309, "runtime_avg": 3859.43},
    {"min_token": 3599, "max_token": 4198, "memory_mb": 55659, "runtime_avg": 5390.29},
    {"min_token": 4198, "max_token": 4731, "memory_mb": 69357, "runtime_avg": 7200.00},
    {"min_token": 4731, "max_token": 5123, "memory_mb": 69229, "runtime_avg": 7200.00},
    {"min_token": 5123, "max_token": 5425, "memory_mb": 75375, "runtime_avg": 7200.00},
    {"min_token": 5425, "max_token": None, "memory_mb": 81037, "runtime_avg": 7200.00},
]

MEMORY_PROFILE_STEPS_RTX4090 = [
    {"min_token": 0,    "max_token": 257,  "memory_mb": 2741,  "runtime_avg": 72.85},
    {"min_token": 257,  "max_token": 513,  "memory_mb": 3899,  "runtime_avg": 128.12},
    {"min_token": 513,  "max_token": 769,  "memory_mb": 5433,  "runtime_avg": 221.50},
    {"min_token": 769,  "max_token": 1025, "memory_mb": 7227,  "runtime_avg": 339.80},
    {"min_token": 1025, "max_token": 1281, "memory_mb": 9149,  "runtime_avg": 473.40},
    {"min_token": 1281, "max_token": 1537, "memory_mb": 11453, "runtime_avg": 655.20},
    {"min_token": 1537, "max_token": 2053, "memory_mb": 17213, "runtime_avg": 1098.70},
    {"min_token": 2053, "max_token": 2579, "memory_mb": 22975, "runtime_avg": 1698.50},
    {"min_token": 2579, "max_token": 3157, "memory_mb": 24209, "runtime_avg": 3300.00},
    {"min_token": 3157, "max_token": None, "memory_mb": 24209, "runtime_avg": 7200.00},
]

GPU_PRESETS: Dict[str, Dict] = {
    "a800-80g":  {"vram_mb": 80 * 1024,  "profile": "a800",    "label": "NVIDIA A800 80 GB"},
    "a100-80g":  {"vram_mb": 80 * 1024,  "profile": "a800",    "label": "NVIDIA A100 80 GB"},
    "a100-40g":  {"vram_mb": 40 * 1024,  "profile": "a800",    "label": "NVIDIA A100 40 GB"},
    "h100-80g":  {"vram_mb": 80 * 1024,  "profile": "a800",    "label": "NVIDIA H100 80 GB"},
    "h100-94g":  {"vram_mb": 94 * 1024,  "profile": "a800",    "label": "NVIDIA H100 NVL 94 GB"},
    "a6000-48g": {"vram_mb": 48 * 1024,  "profile": "a800",    "label": "NVIDIA RTX A6000 48 GB"},
    "v100-32g":  {"vram_mb": 32 * 1024,  "profile": "a800",    "label": "NVIDIA V100 32 GB"},
    "rtx4090":   {"vram_mb": 24 * 1024,  "profile": "rtx4090", "label": "NVIDIA RTX 4090 24 GB"},
    "rtx3090":   {"vram_mb": 24 * 1024,  "profile": "rtx4090", "label": "NVIDIA RTX 3090 24 GB"},
    "rtx3090ti": {"vram_mb": 24 * 1024,  "profile": "rtx4090", "label": "NVIDIA RTX 3090 Ti 24 GB"},
    "rtx4080":   {"vram_mb": 16 * 1024,  "profile": "rtx4090", "label": "NVIDIA RTX 4080 16 GB"},
}

DEFAULT_GPU_VRAM_MB = 80 * 1024  # 81920 MB

COMMON_CCD_HEAVY_ATOMS: Dict[str, int] = {
    "ATP": 31, "ADP": 27, "AMP": 23, "GTP": 32, "GDP": 28, "GMP": 24,
    "CTP": 30, "CDP": 26, "CMP": 22, "UTP": 29, "UDP": 25, "UMP": 21,
    "NAD": 44, "NADH": 44, "NADP": 47, "FAD": 53, "FMN": 31, "SAM": 27,
    "SAH": 26, "COA": 51, "HEM": 43, "CLA": 55,
    "GLC": 12, "GAL": 12, "FUC": 12, "MAN": 12, "FRU": 12, "SUC": 23,
    "BGC": 12, "SIA": 14, "NDG": 15, "NAG": 14,
    "MG": 1, "ZN": 1, "CA": 1, "MN": 1, "FE": 1, "CU": 1, "NI": 1,
    "CO": 1, "CD": 1, "HG": 1, "PT": 1, "NA": 1, "K": 1, "CL": 1,
    "BR": 1, "IOD": 1, "SO4": 5, "PO4": 5, "NO3": 4, "ACT": 4,
    "HEC": 43, "GOL": 6, "EDO": 6, "PEG": 14, "EPE": 22, "MES": 18,
    "TRS": 11, "BTB": 22, "IMD": 5, "DMS": 4, "ACE": 4, "ACY": 4,
    "FMT": 3, "PCA": 9, "CSO": 8, "CME": 12, "MSE": 9, "TPO": 12,
    "SEP": 11, "PTR": 18, "M3L": 15, "KCX": 12,
    "STI": 35, "GNF": 31, "LAP": 28, "ERL": 27, "GEF": 29,
}

DEFAULT_CCD_HEAVY_ATOMS = 20
DEFAULT_SMILES_HEAVY_ATOMS = 20


class Colors:
    """ANSI escape codes used by the colored logging helpers."""

    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    MAGENTA = '\033[0;35m'
    CYAN = '\033[0;36m'
    NC = '\033[0m'


class StreamingResultWriter:
    """Thread-safe writer for streaming TSV results in real-time."""

    def __init__(self, output_file: Path):
        self.output_file = output_file
        self._write_lock = threading.Lock()
        self.header_written = False
        self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        with self._write_lock:
            if self._fh and not self._fh.closed:
                try:
                    self._fh.flush()
                    self._fh.close()
                except OSError:
                    pass
            self._fh = None

    def _ensure_handle(self):
        if self._fh is None or self._fh.closed:
            self._fh = open(self.output_file, 'a', newline='', encoding='utf-8')

    def write_header(self):
        with self._write_lock:
            if self._fh and not self._fh.closed:
                self._fh.close()
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f, delimiter='\t')
                writer.writerow([
                    'gpu_id', 'batch_id', 'is_retry',
                    'batch_type', 'wave_id',
                    'batch_peak_memory_mb', 'batch_runtime_seconds',
                    'task_id', 'json_file', 'protein_name',
                    'token_count', 'protein_length', 'rna_length', 'dna_length', 'ligand_count',
                    'total_sequences',
                    'estimated_memory_mb', 'estimated_runtime_s',
                    'task_peak_memory_mb', 'runtime_seconds',
                    'timeout_risk', 'success',
                    'timestamp'
                ])
            self._fh = open(self.output_file, 'a', newline='', encoding='utf-8')
            self.header_written = True

    def write_task_result(self, task: 'PredictionTask', ok: bool, runtime: float,
                         peak_mem: int, gpu_id: int, batch_id: str,
                         batch_peak_memory: int, batch_runtime: float,
                         is_retry: bool = False,
                         batch_type: str = 'normal',
                         wave_id: str = ''):
        with self._write_lock:
            self._ensure_handle()
            writer = csv.writer(self._fh, delimiter='\t')
            ji = task.json_info
            writer.writerow([
                gpu_id, batch_id, is_retry,
                batch_type, wave_id,
                batch_peak_memory, f"{batch_runtime:.2f}",
                task.task_id, task.json_file.name,
                ji['name'], ji['token_count'], ji['protein_length'],
                ji.get('rna_length', 0), ji.get('dna_length', 0), ji['ligand_count'],
                ji['total_sequences'],
                task.estimated_memory, f"{task.estimated_runtime:.1f}",
                peak_mem, f"{runtime:.2f}",
                task.timeout_risk, ok,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
            self._fh.flush()

    def write_skipped_task(self, json_file: Path, ji: Dict,
                           reason: str = 'skipped_existing_output'):
        with self._write_lock:
            self._ensure_handle()
            writer = csv.writer(self._fh, delimiter='\t')
            writer.writerow([
                'N/A', 'skipped', False,
                'skipped', '',
                'N/A', 'N/A',
                'N/A', json_file.name, ji['name'],
                ji['token_count'], ji['protein_length'],
                ji.get('rna_length', 0), ji.get('dna_length', 0), ji['ligand_count'],
                ji['total_sequences'],
                'N/A', 'N/A',
                'N/A', 'N/A',
                'N/A', reason,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ])
            self._fh.flush()


@dataclass
class TokenMemoryProfile:
    """One profiling data point: token count -> measured memory and runtime."""

    token_count: int
    memory_usage_mb: int
    runtime_seconds: float
    success: bool = True


@dataclass
class PredictionTask:
    """A single AF3 inference task plus its estimated VRAM and runtime,
    derived from the loaded profile. timeout_risk and vram_overflow
    flag tasks that need solo-batch scheduling."""

    json_file: Path
    json_info: Dict
    estimated_memory: int
    estimated_runtime: float
    task_id: str
    timeout_risk: bool = False
    vram_overflow: bool = False


@dataclass
class TaskBatch:
    """A regular batch: tasks that fit jointly within the per-GPU VRAM
    and wall-clock budgets and run concurrently on the same GPU."""

    tasks: List[PredictionTask]
    total_memory: int
    estimated_max_runtime: float
    batch_id: str


@dataclass
class TaskWave:
    """A single wave of lightweight tasks scheduled inside an anchor task's
    VRAM shadow. Multiple waves share the same anchor."""

    tasks: List[PredictionTask]
    total_memory: int
    estimated_max_runtime: float
    wave_id: str


@dataclass
class TemporalWaveBatch:
    """A multi-anchor temporal-wave batch: one or more long-running anchor
    tasks plus successive waves of lightweight tasks dispatched within the
    anchor runtime window. total_memory accounts for the worst-case
    co-residency of all anchors and the largest single wave."""

    anchor_tasks: List[PredictionTask]
    waves: List[TaskWave]
    anchor_group_memory: int
    wave_memory_budget: int
    estimated_anchor_runtime: float
    batch_id: str

    @property
    def anchor(self) -> Optional[PredictionTask]:
        return self.anchor_tasks[0] if self.anchor_tasks else None

    @property
    def anchor_memory(self) -> int:
        return self.anchor_group_memory

    @property
    def tasks(self) -> List[PredictionTask]:
        all_tasks = list(self.anchor_tasks)
        for wave in self.waves:
            all_tasks.extend(wave.tasks)
        return all_tasks

    @property
    def total_memory(self) -> int:
        wave_max = max((w.total_memory for w in self.waves), default=0)
        return self.anchor_group_memory + wave_max

    @property
    def estimated_max_runtime(self) -> float:
        return self.estimated_anchor_runtime

    @property
    def wave_task_count(self) -> int:
        return sum(len(w.tasks) for w in self.waves)


@dataclass
class GPUWorker:
    """Per-GPU execution unit: the GPU index, its assigned task list, the
    pre-computed batch plan (regular + temporal-wave), and the per-GPU
    working directory used to stage JSON inputs."""

    gpu_id: int
    tasks: List[PredictionTask]
    total_tokens: int
    batches: List
    working_dir: Path


# ---------------------------------------------
#  Logging helpers
# ---------------------------------------------

def print_colored(message: str, color: str = Colors.NC):
    print(f"{color}{message}{Colors.NC}")

def info(message: str):
    print_colored(f"[INFO] {message}", Colors.BLUE)

def success(message: str):
    print_colored(f"[SUCCESS] {message}", Colors.GREEN)

def warning(message: str):
    print_colored(f"[WARNING] {message}", Colors.YELLOW)

def error(message: str):
    print_colored(f"[ERROR] {message}", Colors.RED)


_VERBOSE = False

def set_verbose(flag: bool):
    global _VERBOSE
    _VERBOSE = bool(flag)

def debug(message: str):
    if _VERBOSE:
        print_colored(f"[DEBUG] {message}", Colors.CYAN)


def detect_gpu_vram_mb(gpu_id: int) -> Optional[int]:
    """Query actual VRAM of a specific GPU via nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', f'--id={gpu_id}',
             '--query-gpu=memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            raw = result.stdout.strip()
            if raw.isdigit():
                return int(raw)
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError, OSError):
        pass
    return None


def _guess_gpu_preset_from_vram(vram_mb: int) -> Optional[str]:
    """Match a measured VRAM size (MB) to a GPU_PRESETS key within +/-512 MB."""
    for name, preset in GPU_PRESETS.items():
        if abs(preset['vram_mb'] - vram_mb) <= 512:
            return name
    return None


class GPUMonitor:
    """Background polling thread that records per-GPU VRAM, utilization
    and temperature via nvidia-smi at a fixed interval, tracking peak
    memory across the run. Polling errors trigger exponential backoff and
    the loop exits permanently after MAX_CONSECUTIVE_ERRORS failures."""

    MAX_GPU_DATA_ENTRIES = 86_400
    
    MAX_CONSECUTIVE_ERRORS = 30

    def __init__(self, interval: int = 1, gpu_id: int = 0):
        self.interval = interval
        self.gpu_id = gpu_id
        self.monitoring = False
        self.monitor_thread = None
        self.current_memory = 0
        self.peak_memory = 0
        self.gpu_data = []
        self._lock = threading.Lock()
        self._consecutive_errors = 0

    def check_nvidia_smi(self) -> bool:
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def get_gpu_info(self, specific_gpu: int = None) -> Dict:
        
        _DEFAULT = {'memory_used': 0, 'memory_total': 0, 'memory_percent': 0,
                     'gpu_util': 'N/A', 'temperature': 'N/A'}
        gpu_idx = specific_gpu if specific_gpu is not None else self.gpu_id
        
        acquired = _NVIDIA_SMI_SEMAPHORE.acquire(timeout=self.interval * 2)
        if not acquired:
            return _DEFAULT
        try:
            result = subprocess.run(
                ['nvidia-smi', f'--id={gpu_idx}',
                 '--query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(',')]
                if len(parts) >= 4:
                    memory_used  = int(parts[0]) if parts[0].isdigit() else 0
                    memory_total = int(parts[1]) if parts[1].isdigit() else 0
                    gpu_util     = parts[2] if parts[2] else 'N/A'
                    temperature  = parts[3] if parts[3] else 'N/A'
                    memory_percent = (memory_used / memory_total * 100) if memory_total > 0 else 0
                    return {
                        'memory_used': memory_used, 'memory_total': memory_total,
                        'memory_percent': memory_percent, 'gpu_util': gpu_util,
                        'temperature': temperature
                    }
        except (subprocess.TimeoutExpired, ValueError, IndexError, OSError):
            pass
        finally:
            _NVIDIA_SMI_SEMAPHORE.release()
        return _DEFAULT

    def get_current_memory_usage(self) -> int:
        return self.get_gpu_info()['memory_used']

    def _monitor_loop(self):
        
        backoff_interval = self.interval
        while self.monitoring:
            try:
                gpu_info = self.get_gpu_info()
                current_memory = gpu_info['memory_used']
                with self._lock:
                    if current_memory > self.peak_memory:
                        self.peak_memory = current_memory
                    self.current_memory = current_memory
                    if len(self.gpu_data) < self.MAX_GPU_DATA_ENTRIES:
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        self.gpu_data.append({
                            'timestamp': timestamp,
                            'memory_used': current_memory,
                            'memory_total': gpu_info['memory_total'],
                            'memory_percent': gpu_info['memory_percent'],
                            'gpu_util': gpu_info['gpu_util'],
                            'temperature': gpu_info['temperature']
                        })
                # Successful - reset backoff
                self._consecutive_errors = 0
                backoff_interval = self.interval
            except Exception:
                self._consecutive_errors += 1
                if self._consecutive_errors >= self.MAX_CONSECUTIVE_ERRORS:
                    break  # stop monitoring; peak_memory retains last good value
                backoff_interval = min(backoff_interval * 2, 60)
            time.sleep(backoff_interval)

    def start_monitoring(self):
        initial_memory = self.get_current_memory_usage()
        with self._lock:
            self.monitoring = True
            self.peak_memory = initial_memory
            self.current_memory = initial_memory
            self.gpu_data = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=3)


# ---------------------------------------------
#  GPU Detection
# ---------------------------------------------

def detect_available_gpus() -> List[int]:
    """Return the list of visible GPU indices reported by nvidia-smi, or [] on failure."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--list-gpus'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            return list(range(len(lines)))
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return []


def parse_gpu_list(gpu_str: str) -> List[int]:
    """Parse a CLI GPU spec like '0,1,3' or '0-3,5' into a sorted list of GPU IDs;
    if empty, fall back to all GPUs detected by nvidia-smi."""
    if not gpu_str:
        return detect_available_gpus()
    gpus = set()
    parts = gpu_str.split(',')
    for part in parts:
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            gpus.update(range(int(start), int(end) + 1))
        elif part.isdigit():
            gpus.add(int(part))
    return sorted(list(gpus))


# ---------------------------------------------
#  Token-level Memory Profile Loader
# ---------------------------------------------

class TokenMemoryProfileLoader:
    """Loads and queries the stepwise AF3 memory-runtime profile.

    A profile is a list of (min_token, max_token, memory_mb, runtime_avg)
    tuples, sorted by min_token. Lookups use bisect_right for
    O(log S) memory and runtime estimates. Sources, in priority order:

      - an external TSV passed via profile_file (columns:
        token_count, peak_memory_mb, runtime_seconds;
        anomalous >30% memory drops are filtered, gaps optionally
        interpolated by _fill_profile_gaps);
      - the built-in A800 80 GB or RTX 4090 24 GB tables, selected by
        builtin_profile.

    set_gpu_vram records the per-GPU VRAM cap and identifies the
    smallest token count whose predicted memory exceeds the effective
    VRAM (after applying vram_margin); tasks above that threshold are
    flagged for CPU memory offloading or skipping.
    """

    MEMORY_FLOOR_MB = 2741

    def __init__(self, profile_file: Optional[Path] = None, vram_margin: float = 0.95,
                 builtin_profile: str = 'a800', profile_gap_fill: bool = True):
        self.vram_margin = vram_margin
        self.profiles: Dict[int, TokenMemoryProfile] = {}
        self.profile_file = profile_file
        self._builtin_profile = builtin_profile
        self._do_gap_fill: bool = profile_gap_fill
        self._memory_steps: List[Tuple[int, Optional[int], int, float]] = []
        self.gpu_vram_mb: Optional[int] = None
        self.effective_vram_mb: Optional[int] = None
        self._vram_overflow_token: Optional[int] = None
        self.profile_source: str = 'builtin'
        self.profile_source_label: str = 'Built-in A800 80GB profile (15 steps)'
        self._step_min_tokens: List[int] = []

        if profile_file and profile_file.exists():
            self._load_external_profiles()
        else:
            self._load_builtin_profiles()

    def _rebuild_step_index(self):
        self._step_min_tokens = [s[0] for s in self._memory_steps]

    def _load_builtin_profiles(self):
        if self._builtin_profile == 'rtx4090':
            steps_src = MEMORY_PROFILE_STEPS_RTX4090
            label_base = 'Built-in RTX 4090 24GB profile'
        else:
            steps_src = MEMORY_PROFILE_STEPS
            label_base = 'Built-in A800 80GB profile'

        for step in steps_src:
            self._memory_steps.append((
                step["min_token"], step["max_token"],
                step["memory_mb"], step["runtime_avg"]
            ))
        self.profile_source = 'builtin'
        self.profile_source_label = f'{label_base} ({len(self._memory_steps)} steps)'
        self._rebuild_step_index()
        info(f"Loaded {self.profile_source_label}")

    def _load_external_profiles(self):
        REQUIRED_COLUMNS = {'token_count', 'peak_memory_mb', 'runtime_seconds'}
        try:
            with open(self.profile_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                if reader.fieldnames is None:
                    raise RuntimeError("Profile file appears to be empty (no header row)")
                missing = REQUIRED_COLUMNS - set(reader.fieldnames)
                if missing:
                    raise RuntimeError(
                        f"Profile file is missing required column(s): {sorted(missing)}. "
                        f"Found columns: {list(reader.fieldnames)}"
                    )
                for row in reader:
                    try:
                        token_count = int(row['token_count'])
                        memory_mb = int(row['peak_memory_mb'])
                        runtime = float(row['runtime_seconds'])
                        success_flag = row.get('success', 'True').strip().lower() == 'true'
                        existing = self.profiles.get(token_count)
                        if existing is None:
                            self.profiles[token_count] = TokenMemoryProfile(
                                token_count=token_count, memory_usage_mb=memory_mb,
                                runtime_seconds=runtime, success=success_flag
                            )
                        else:
                            if memory_mb > existing.memory_usage_mb:
                                existing.memory_usage_mb = memory_mb
                            if runtime > existing.runtime_seconds:
                                existing.runtime_seconds = runtime
                            if not success_flag:
                                existing.success = False
                    except (ValueError, KeyError) as e:
                        warning(f"Skipping invalid profile row: {type(e).__name__}: {e}")

            self._build_steps_from_profiles()
            n_steps = len(self._memory_steps)
            self.profile_source = 'external'
            self.profile_source_label = (
                f'External profile: {self.profile_file.name} '
                f'({len(self.profiles)} data points -> {n_steps} steps)'
            )
            info(f"Loaded {len(self.profiles)} token-level memory profiles from {self.profile_file}")
            info(f"Built {n_steps} discrete memory steps from external profile")
        except Exception as e:
            warning(f"Failed to load external profiles: {type(e).__name__}: {e}, using built-in")
            self._load_builtin_profiles()

    def _build_steps_from_profiles(self):
        if not self.profiles:
            return
        sorted_tokens = sorted(self.profiles.keys())
        step_start_token: int = sorted_tokens[0]
        step_tokens_list: List[int] = []
        prev_mem: Optional[int] = None

        for token in sorted_tokens:
            mem = self.profiles[token].memory_usage_mb
            if prev_mem is not None and mem != prev_mem:
                avg_runtime = (
                    sum(self.profiles[t].runtime_seconds for t in step_tokens_list)
                    / len(step_tokens_list)
                )
                self._memory_steps.append((step_start_token, token, prev_mem, avg_runtime))
                step_start_token = token
                step_tokens_list = []
            step_tokens_list.append(token)
            prev_mem = mem

        if step_tokens_list and prev_mem is not None:
            avg_runtime = (
                sum(self.profiles[t].runtime_seconds for t in step_tokens_list)
                / len(step_tokens_list)
            )
            self._memory_steps.append((step_start_token, None, prev_mem, avg_runtime))

        if len(self._memory_steps) > 1:
            cleaned = [self._memory_steps[0]]
            for step in self._memory_steps[1:]:
                prev_step = cleaned[-1]
                prev_step_mem = prev_step[2]
                cur_mem = step[2]
                drop_ratio = (prev_step_mem - cur_mem) / prev_step_mem if prev_step_mem else 0
                if drop_ratio > 0.30:
                    warning(
                        f"Skipping anomalous memory step at token >= {step[0]}: "
                        f"{cur_mem:,} MB drops >30% from previous step {prev_step_mem:,} MB."
                    )
                else:
                    cleaned.append(step)
            self._memory_steps = cleaned

        self._rebuild_step_index()
        self._fill_profile_gaps()

    def _fill_profile_gaps(self):
        if not getattr(self, '_do_gap_fill', True):
            return
        if len(self._memory_steps) < 2:
            return
        filled: List[Tuple[int, Optional[int], int, float]] = []
        gaps_found = 0
        for i, step in enumerate(self._memory_steps):
            filled.append(step)
            if i >= len(self._memory_steps) - 1:
                continue
            curr_min, curr_max, curr_mem, curr_rt = step
            next_min, next_max, next_mem, next_rt = self._memory_steps[i + 1]
            if curr_max is None:
                continue
            gap = next_min - curr_max
            if gap <= 0:
                continue
            gap_mem = int(curr_mem + (next_mem - curr_mem) * 0.5)
            gap_rt  = curr_rt + (next_rt - curr_rt) * 0.5
            filled.append((curr_max, next_min, gap_mem, gap_rt))
            gaps_found += 1
            warning(
                f"Profile gap: tokens {curr_max}-{next_min} ({gap} tokens wide, "
                f"no measurements) -> interpolated step: {gap_mem} MB, {gap_rt:.0f}s."
            )
        if gaps_found:
            self._memory_steps = filled
            self._rebuild_step_index()
            info(f"Profile gap-fill: {gaps_found} gap(s) interpolated "
                 f"({len(self._memory_steps)} total steps after fill)")

    def set_gpu_vram(self, vram_mb: int):
        self.gpu_vram_mb = vram_mb
        self.effective_vram_mb = int(vram_mb * self.vram_margin)
        if self._memory_steps:
            for min_token, max_token, mem, _ in self._memory_steps:
                if mem > self.effective_vram_mb:
                    self._vram_overflow_token = min_token
                    break
            if self._vram_overflow_token:
                warning(f"GPU VRAM limit: {vram_mb}MB ({vram_mb/1024:.1f}GB)")
                warning(f"Effective limit (with {self.vram_margin*100:.0f}% margin): {self.effective_vram_mb}MB")
                warning(f"VRAM overflow at token >= {self._vram_overflow_token} (uses CPU buffer, drastically slower)")
            else:
                info(f"GPU VRAM limit: {vram_mb}MB ({vram_mb/1024:.1f}GB) - all profiled tasks fit within VRAM")

    def _lookup_step(self, token_count: int) -> int:
        if not self._step_min_tokens:
            return -1
        idx = bisect.bisect_right(self._step_min_tokens, token_count) - 1
        return max(0, min(idx, len(self._memory_steps) - 1))

    def estimate_memory_mb(self, token_count: int) -> int:
        if not self._memory_steps:
            return self.MEMORY_FLOOR_MB
        return self._memory_steps[self._lookup_step(token_count)][2]

    def estimate_runtime_seconds(self, token_count: int) -> float:
        if not self._memory_steps:
            return 7200.0
        return self._memory_steps[self._lookup_step(token_count)][3]

    def is_timeout_risk(self, token_count: int, timeout_seconds: float = 7200.0) -> bool:
        est_runtime = self.estimate_runtime_seconds(token_count)
        return est_runtime >= timeout_seconds * 0.85

    def is_over_gpu_vram(self, token_count: int) -> bool:
        if self._vram_overflow_token is None:
            return False
        return token_count >= self._vram_overflow_token

    def get_memory_step_summary(self) -> List[Tuple[int, Optional[int], int, bool]]:
        if not self._memory_steps:
            return []
        result = []
        for min_token, max_token, mem, _ in self._memory_steps:
            exceeds = self.effective_vram_mb is not None and mem > self.effective_vram_mb
            result.append((min_token, max_token, mem, exceeds))
        return result

    def get_vram_overflow_threshold(self) -> Optional[int]:
        return self._vram_overflow_token


# ---------------------------------------------
#  AF3 JSON Token Counter
# ---------------------------------------------

def _count_smiles_heavy_atoms(smiles: str) -> int:
    """Count heavy (non-hydrogen) atoms in a SMILES string.

    Uses RDKit when available; otherwise falls back to a regex parser
    that recognises bracket atoms ([Fe2+], [NH4+], ...) and the SMILES
    organic subset (C, N, O, P, S, F, Cl, Br, I and aromatic forms).
    Returns at least 1 so that empty/parsing-failed ligands still count.
    """
    try:
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            return mol.GetNumHeavyAtoms()
    except Exception:
        pass

    # Regex fallback - counts bracket atoms ([Fe2+], [NH4+], ...) and
    # organic-subset atoms (C, N, O, P, S, F, Cl, Br, I and aromatic
    # b/c/n/o/p/s).  Two-letter elements (Cl, Br) are matched first via
    # alternation so they aren't split.  H is never in the regex set, so
    # we never need a separate H filter.
    count = 0
    bracket_re = re.compile(r'\[([^]]+)\]')
    for match in bracket_re.finditer(smiles):
        inner = match.group(1)
        elem = re.match(r'\d*([A-Z][a-z]?)', inner)
        if elem and elem.group(1) != 'H':
            count += 1
    bracket_stripped = bracket_re.sub('', smiles)

    organic_atoms = re.findall(r'Cl|Br|[BCNOPSFInbcnops]', bracket_stripped)
    count += len(organic_atoms)
    return max(1, count)


def _count_ccd_heavy_atoms(ccd_id: str) -> int:
    """Look up the heavy-atom count for a CCD residue code, falling back to
    DEFAULT_CCD_HEAVY_ATOMS for codes not in the built-in table."""
    cid = ccd_id.strip().upper()
    return COMMON_CCD_HEAVY_ATOMS.get(cid, DEFAULT_CCD_HEAVY_ATOMS)


def count_tokens_from_af3_json(data: Dict) -> int:
    """Compute the AF3 token count for one input JSON.

    Follows the AF3 specification exactly: 1 token per residue for
    proteins, 1 token per nucleotide for RNA/DNA, and 1 token per heavy
    atom for ligands (CCD lookup or SMILES parse). Both the
    "type": "..." and the legacy {"protein": {...}} JSON shapes
    are accepted, and count multipliers are honoured.
    """
    total_tokens = 0
    sequences = data.get('sequences', [])

    for seq_entry in sequences:
        seq_type = seq_entry.get('type', '').lower()
        count = max(1, int(seq_entry.get('count', 1)))

        if seq_type == 'protein':
            seq = seq_entry.get('sequence', '')
            total_tokens += len(seq) * count
        elif seq_type in ('rna', 'dna'):
            seq = seq_entry.get('sequence', '')
            total_tokens += len(seq) * count
        elif seq_type == 'ligand':
            ccd_ids = seq_entry.get('ccd_ids', [])
            smiles = seq_entry.get('smiles', '')
            if ccd_ids:
                for cid in ccd_ids:
                    total_tokens += _count_ccd_heavy_atoms(cid) * count
            elif smiles:
                total_tokens += _count_smiles_heavy_atoms(smiles) * count
            else:
                total_tokens += DEFAULT_CCD_HEAVY_ATOMS * count
        elif 'protein' in seq_entry:
            p = seq_entry['protein']
            seq = p.get('sequence', '')
            c = max(1, int(seq_entry.get('count', 1)))
            total_tokens += len(seq) * c
        elif 'rna' in seq_entry:
            r = seq_entry['rna']
            seq = r.get('sequence', '')
            c = max(1, int(seq_entry.get('count', 1)))
            total_tokens += len(seq) * c
        elif 'dna' in seq_entry:
            d = seq_entry['dna']
            seq = d.get('sequence', '')
            c = max(1, int(seq_entry.get('count', 1)))
            total_tokens += len(seq) * c
        elif 'ligand' in seq_entry:
            lig = seq_entry['ligand']
            c = max(1, int(seq_entry.get('count', 1)))
            ccd_ids = lig.get('ccdCodes', lig.get('ccd_ids', []))
            smiles = lig.get('smilesString', lig.get('smiles', ''))
            if isinstance(ccd_ids, str):
                ccd_ids = [ccd_ids]
            if ccd_ids:
                for cid in ccd_ids:
                    total_tokens += _count_ccd_heavy_atoms(cid) * c
            elif smiles:
                total_tokens += _count_smiles_heavy_atoms(smiles) * c
            else:
                total_tokens += DEFAULT_CCD_HEAVY_ATOMS * c

    return max(1, total_tokens)


# ---------------------------------------------
#  Dual-dimension Task Optimizer + Temporal Wave Scheduler
# ---------------------------------------------

class DualDimensionTaskOptimizer:
    """Per-GPU batch planner enforcing a joint VRAM and wall-clock budget.

    available_memory_mb = max_memory_mb x (1 - safety_margin)
    reserves headroom for runtime fluctuations and framework overhead
    (default 10%). The planner first peels off timeout-risk and
    VRAM-overflow tasks into single-task batches, then optionally builds
    TemporalWaveBatch objects via _build_temporal_wave_batches,
    and finally packs the remainder into regular TaskBatch objects
    using a Next-Fit Decreasing heuristic.
    """

    def __init__(self, max_memory_mb: int,
                 safety_margin: float = 0.1,
                 max_batch_runtime_seconds: float = 7200.0):
        self.max_memory_mb = max_memory_mb
        self.available_memory_mb = int(max_memory_mb * (1 - safety_margin))
        self.safety_margin = safety_margin
        self.max_batch_runtime = max_batch_runtime_seconds

    def _greedy_skeleton(self, sorted_tasks, batch_id_start):
        """Pack tasks into regular batches using a Next-Fit Decreasing heuristic.

        Tasks must be pre-sorted by descending memory demand. A task is
        appended to the current batch as long as both the cumulative
        memory and the running max runtime remain within the configured
        budgets; otherwise the batch is sealed and the task seeds a fresh
        one. Tasks larger than available_memory_mb are emitted as solo
        ..._oversized batches with a warning.
        """
        batches = []
        batch_id = batch_id_start
        current_tasks = []
        current_memory = 0
        current_max_runtime = 0.0

        for task in sorted_tasks:
            mem_ok = (current_memory + task.estimated_memory) <= self.available_memory_mb
            new_max_runtime = max(current_max_runtime, task.estimated_runtime)
            runtime_ok = new_max_runtime <= self.max_batch_runtime

            if mem_ok and runtime_ok and current_tasks:
                current_tasks.append(task)
                current_memory += task.estimated_memory
                current_max_runtime = new_max_runtime
            else:
                if current_tasks:
                    batches.append(TaskBatch(
                        tasks=current_tasks.copy(), total_memory=current_memory,
                        estimated_max_runtime=current_max_runtime,
                        batch_id=f"batch_{batch_id:03d}"
                    ))
                    batch_id += 1
                current_tasks = [task]
                current_memory = task.estimated_memory
                current_max_runtime = task.estimated_runtime
                if task.estimated_memory > self.available_memory_mb:
                    warning(f"Task {task.task_id} ({task.estimated_memory}MB) exceeds available "
                            f"memory - scheduling solo (oversized)")
                    batches.append(TaskBatch(
                        tasks=[task], total_memory=task.estimated_memory,
                        estimated_max_runtime=task.estimated_runtime,
                        batch_id=f"batch_{batch_id:03d}_oversized"
                    ))
                    batch_id += 1
                    current_tasks = []
                    current_memory = 0
                    current_max_runtime = 0.0

        if current_tasks:
            batches.append(TaskBatch(
                tasks=current_tasks, total_memory=current_memory,
                estimated_max_runtime=current_max_runtime,
                batch_id=f"batch_{batch_id:03d}"
            ))
            batch_id += 1
        return batches, batch_id

    @staticmethod
    def _fill_gaps(batches, filler_pool, available_memory_mb):
        """Backfill spare VRAM headroom in already-built batches with leftover
        small tasks whose runtime fits the batch wall-clock budget."""
        remaining = sorted(filler_pool, key=lambda t: t.estimated_memory, reverse=True)
        for batch in batches:
            if not remaining:
                break
            free_mem = available_memory_mb - batch.total_memory
            still_remaining = []
            for filler in remaining:
                if filler.estimated_memory <= free_mem and filler.estimated_runtime <= batch.estimated_max_runtime:
                    batch.tasks.append(filler)
                    batch.total_memory += filler.estimated_memory
                    free_mem -= filler.estimated_memory
                else:
                    still_remaining.append(filler)
            remaining = still_remaining
        return remaining

    def _build_temporal_wave_batches(self, all_tasks, batch_id_start,
                                     min_anchor_ratio=2.0, max_anchor_group_ratio=1.5):
        """Build Multi-Anchor Temporal Wave batches.

        Algorithm:
          1. Sort tasks by descending estimated runtime; pick the longest
             unscheduled task as the seed anchor.
          2. Greedily extend the anchor group with tasks whose runtime is
             within a max_anchor_group_ratio factor of the seed and
             whose joint memory leaves at least MIN_WAVE_BUDGET_MB free.
          3. From the remaining tasks, select wave candidates whose
             memory fits the wave budget and whose runtime
             x min_anchor_ratio is no longer than the anchor window.
          4. Greedy-pack candidates into successive waves (largest-memory
             first), then cap the wave count at
             max(1, floor(1.1 * anchor_window / avg_wave_runtime))
             so we never schedule more waves than fit in the anchor's
             runtime shadow.
          5. Emit a TemporalWaveBatch and repeat until no more anchor
             groups can be formed.

        Returns (temporal_batches, remaining_tasks, next_batch_id);
        remaining_tasks is fed back into _greedy_skeleton for
        regular packing.
        """
        MIN_WAVE_BUDGET_MB = 500
        sorted_tasks = sorted(all_tasks, key=lambda t: t.estimated_runtime, reverse=True)
        task_by_id = {t.task_id: t for t in sorted_tasks}
        consumed = set()
        temporal_batches = []
        batch_id = batch_id_start

        for i, seed in enumerate(sorted_tasks):
            if seed.task_id in consumed:
                continue
            tentative_group = [seed]
            tentative_ids = {seed.task_id}
            group_mem = seed.estimated_memory
            seed_rt = seed.estimated_runtime

            for candidate in sorted_tasks[i + 1:]:
                if candidate.task_id in consumed or candidate.task_id in tentative_ids:
                    continue
                if candidate.estimated_runtime < seed_rt / max_anchor_group_ratio:
                    break
                if group_mem + candidate.estimated_memory > self.available_memory_mb - MIN_WAVE_BUDGET_MB:
                    continue
                tentative_group.append(candidate)
                tentative_ids.add(candidate.task_id)
                group_mem += candidate.estimated_memory

            anchor_window = max(t.estimated_runtime for t in tentative_group)
            wave_budget = self.available_memory_mb - group_mem
            if wave_budget < MIN_WAVE_BUDGET_MB:
                continue

            wave_candidates = [
                t for t in task_by_id.values()
                if t.task_id not in consumed
                and t.task_id not in tentative_ids
                and t.estimated_memory <= wave_budget
                and t.estimated_runtime * min_anchor_ratio <= anchor_window
            ]
            if not wave_candidates:
                continue

            sorted_cands = sorted(wave_candidates, key=lambda t: t.estimated_memory, reverse=True)
            waves = []
            wave_tasks_curr = []
            wave_mem_curr = 0
            wave_rt_curr = 0.0
            wave_num = 0

            for task in sorted_cands:
                if wave_mem_curr + task.estimated_memory <= wave_budget:
                    wave_tasks_curr.append(task)
                    wave_mem_curr += task.estimated_memory
                    wave_rt_curr = max(wave_rt_curr, task.estimated_runtime)
                else:
                    if wave_tasks_curr:
                        waves.append(TaskWave(tasks=wave_tasks_curr.copy(),
                                              total_memory=wave_mem_curr,
                                              estimated_max_runtime=wave_rt_curr,
                                              wave_id=f"wave_{wave_num:02d}"))
                        wave_num += 1
                    wave_tasks_curr = [task]
                    wave_mem_curr = task.estimated_memory
                    wave_rt_curr = task.estimated_runtime

            if wave_tasks_curr:
                waves.append(TaskWave(tasks=wave_tasks_curr.copy(),
                                      total_memory=wave_mem_curr,
                                      estimated_max_runtime=wave_rt_curr,
                                      wave_id=f"wave_{wave_num:02d}"))
            if not waves:
                continue

            avg_wave_rt = sum(w.estimated_max_runtime for w in waves) / len(waves)
            if avg_wave_rt > 0:
                max_useful = max(1, int(anchor_window / avg_wave_rt * 1.1))
                if len(waves) > max_useful:
                    waves = waves[:max_useful]

            consumed.update(tentative_ids)
            for w in waves:
                for t in w.tasks:
                    consumed.add(t.task_id)

            tb = TemporalWaveBatch(
                anchor_tasks=tentative_group, waves=waves,
                anchor_group_memory=group_mem, wave_memory_budget=wave_budget,
                estimated_anchor_runtime=anchor_window,
                batch_id=f"twbatch_{batch_id:03d}",
            )
            temporal_batches.append(tb)
            batch_id += 1

            total_wt = sum(len(w.tasks) for w in waves)
            if len(tentative_group) == 1:
                anc_desc = (f"anchor={tentative_group[0].task_id} "
                            f"({anchor_window:.0f}s, {group_mem}MB)")
            else:
                ids = '+'.join(t.task_id for t in tentative_group)
                anc_desc = (f"anchors=[{ids}] ({len(tentative_group)} tasks, "
                            f"{group_mem}MB, window={anchor_window:.0f}s)")
            info(f"[TemporalWave] {tb.batch_id}: {anc_desc} | "
                 f"{len(waves)} waves x ~{len(waves[0].tasks)} tasks = "
                 f"{total_wt} wave tasks | wave_budget={wave_budget}MB")

        remaining = [t for t in all_tasks if t.task_id not in consumed]
        return temporal_batches, remaining, batch_id

    def create_optimal_batches(self, tasks, min_anchor_ratio=2.0,
                               use_temporal_waves=True, max_anchor_group_ratio=1.5):
        """Build the full per-GPU batch plan.

        Stages:
          (i)   timeout-risk tasks   -> solo batches (one per task);
          (ii)  VRAM-overflow tasks  -> solo batches routed through CPU
                memory offloading;
          (iii) remaining tasks      -> Multi-Anchor Temporal Wave Batches
                (when use_temporal_waves is set and >=2 normal tasks);
          (iv)  whatever is left     -> regular Next-Fit Decreasing batches
                via _greedy_skeleton.
        """
        batches = []
        batch_id = 1

        risk_tasks     = [t for t in tasks if t.timeout_risk]
        overflow_tasks = [t for t in tasks if t.vram_overflow and not t.timeout_risk]
        normal_tasks   = [t for t in tasks if not t.timeout_risk and not t.vram_overflow]

        for task in risk_tasks:
            overflow_tag = " [VRAM OVERFLOW]" if task.vram_overflow else ""
            batches.append(TaskBatch(tasks=[task], total_memory=task.estimated_memory,
                                     estimated_max_runtime=task.estimated_runtime,
                                     batch_id=f"batch_{batch_id:03d}_timeout_risk"))
            batch_id += 1
            warning(f"Task {task.task_id} placed in solo batch (timeout risk{overflow_tag})")

        for task in overflow_tasks:
            batches.append(TaskBatch(tasks=[task], total_memory=task.estimated_memory,
                                     estimated_max_runtime=task.estimated_runtime,
                                     batch_id=f"batch_{batch_id:03d}_vram_overflow"))
            batch_id += 1
            warning(f"Task {task.task_id} ({task.estimated_memory}MB) exceeds GPU VRAM, "
                    f"placed in solo batch (will use CPU memory buffer)")

        if not normal_tasks:
            return batches

        remaining = normal_tasks
        if use_temporal_waves and len(normal_tasks) >= 2:
            temporal_batches, remaining, batch_id = self._build_temporal_wave_batches(
                normal_tasks, batch_id, min_anchor_ratio, max_anchor_group_ratio
            )
            if temporal_batches:
                batches.extend(temporal_batches)
                info(f"TemporalWave: {len(temporal_batches)} wave batch(es) scheduled, "
                     f"{len(remaining)} task(s) remain for regular packing")

        if remaining:
            remaining_sorted = sorted(remaining,
                                      key=lambda t: (t.estimated_memory, t.estimated_runtime),
                                      reverse=True)
            regular_batches, batch_id = self._greedy_skeleton(remaining_sorted, batch_id)
            batches.extend(regular_batches)

        return batches


# ---------------------------------------------
#  Multi-GPU Task Distribution
# ---------------------------------------------

def distribute_tasks_by_tokens(tasks, num_gpus):
    """Distribute tasks across GPUs using Longest Processing Time (LPT) scheduling.

    Tasks are sorted by descending token count and assigned, one at a
    time, to the GPU currently carrying the smallest accumulated token
    load (tracked in a min-heap). Complexity is O(N log N) for the sort
    plus O(log G) per assignment, dominated by O(N log N) overall.
    The token-balanced assignment minimises the makespan across GPUs.
    """
    if num_gpus <= 0:
        return [tasks] if tasks else []
    if len(tasks) <= num_gpus:
        result = [[] for _ in range(num_gpus)]
        for i, task in enumerate(tasks):
            result[i % num_gpus].append(task)
        return result

    heap = [(0, i) for i in range(num_gpus)]
    heapq.heapify(heap)
    gpu_tasks = [[] for _ in range(num_gpus)]

    for task in sorted(tasks, key=lambda t: t.json_info.get('token_count', 0), reverse=True):
        tok, gid = heapq.heappop(heap)
        gpu_tasks[gid].append(task)
        heapq.heappush(heap, (tok + task.json_info.get('token_count', 0), gid))

    return gpu_tasks


def create_gpu_workers(tasks_by_gpu, gpu_ids, optimizer, workspace_root,
                        min_anchor_ratio=2.0, use_temporal_waves=True,
                        max_anchor_group_ratio=1.5):
    """Build a GPUWorker for each GPU, attaching its tasks, batch plan, and per-GPU working directory."""
    workers = []
    for gpu_id, gpu_task_list in zip(gpu_ids, tasks_by_gpu):
        if not gpu_task_list:
            continue
        total_tokens = sum(t.json_info.get('token_count', 0) for t in gpu_task_list)
        batches = optimizer.create_optimal_batches(
            gpu_task_list, min_anchor_ratio=min_anchor_ratio,
            use_temporal_waves=use_temporal_waves,
            max_anchor_group_ratio=max_anchor_group_ratio,
        )
        gpu_dir = workspace_root / f"gpu_{gpu_id}_work"
        worker = GPUWorker(gpu_id=gpu_id, tasks=gpu_task_list, total_tokens=total_tokens,
                           batches=batches, working_dir=gpu_dir)
        workers.append(worker)
    return workers


# ---------------------------------------------
#  Output helpers
# ---------------------------------------------

def check_output_exists(output_dir, json_info, verbose=False):
    """Return True if a non-empty output folder for this job already exists
    (sanitised by lowercasing and replacing whitespace and path separators)."""
    output_path = Path(output_dir)
    job_name = json_info.get('name', 'Unknown')
    sanitized_name = job_name.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
    try:
        for folder in output_path.iterdir():
            if folder.is_dir() and folder.name.lower() == sanitized_name:
                if any(folder.iterdir()):
                    if verbose:
                        debug(f"Found completed output folder for '{job_name}' ({folder})")
                    return True
    except (PermissionError, FileNotFoundError):
        pass
    return False


def filter_harmless_warnings(stderr_text):
    """Drop benign JAX/XLA/TensorFlow backend-init noise from stderr so that
    real errors stand out. Used by is_task_successful and by the
    foreground error logger."""
    harmless_patterns = [
        "Unable to initialize backend 'rocm'", "Unable to initialize backend 'tpu'",
        "module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'",
        "Failed to open libtpu.so", "libtpu.so:", "INTERNAL: Failed to open libtpu.so",
        "No GPU/TPU found, falling back to CPU", "XLA service", "UserWarning:",
        "FutureWarning:", "DeprecationWarning:", "xla_bridge.py:", "GpuAllocatorConfig",
        "backend initialization", "JAX backend", "tensorflow/compiler/xla",
        "WARNING:absl:", "WARNING:tensorflow:", "backend 'rocm'", "backend 'tpu'",
        "libtpu initialization", "XLA_FLAGS", "No devices found", "Backend initialization warning"
    ]
    _tf_log_prefix_re = re.compile(r'^[IW]\d{4}\s')

    lines = stderr_text.split('\n')
    filtered_lines = []
    for line in lines:
        if not line.strip():
            continue
        line_lower = line.lower()
        is_harmless = any(pattern.lower() in line_lower for pattern in harmless_patterns)
        if not is_harmless and _tf_log_prefix_re.match(line):
            is_harmless = True
        if not is_harmless:
            if not any(x in line_lower for x in ['error', 'exception', 'failed', 'abort', 'crash', 'killed']):
                if any(x in line_lower for x in ['tensorflow', 'jax', 'xla']):
                    is_harmless = True
        if not is_harmless:
            filtered_lines.append(line)
    return '\n'.join(filtered_lines)


def is_task_successful(output_dir, result, strict_errors=False, task=None):
    """Decide whether an AF3 subprocess call actually produced a valid result.

    Decision order:
      1. If the per-job output folder exists and contains any of the
         standard AF3 artefacts (*.cif, confidence_*.json,
         summary_confidences_*.json, fold_*.pdb, ranked_*.pdb,
         *.pkl) -> success, regardless of exit code.
      2. If strict_errors is set -> trust returncode.
      3. Otherwise filter benign warnings from stderr; if a real-error
         pattern survives -> failure, else trust returncode.
    """
    output_path = Path(output_dir)
    success_indicators = [
        "fold_*.pdb", "confidence_*.json", "summary_confidences_*.json",
        "*.cif", "result_*.pdb", "ranked_*.pdb", "*.pkl"
    ]


    candidate_dirs: List[Path] = []
    if task is not None:
        job_name = task.json_info.get('name', '') if isinstance(task.json_info, dict) else ''
        if job_name:
            sanitized = (job_name.replace(' ', '_')
                                  .replace('/', '_')
                                  .replace('\\', '_').lower())
            job_dir = output_path / sanitized
            if job_dir.exists() and job_dir.is_dir():
                candidate_dirs.append(job_dir)

    for cdir in candidate_dirs:
        for pattern in success_indicators:
            if any(cdir.rglob(pattern)):
                return True

    if strict_errors:
        return result.returncode == 0

    if result.stderr:
        filtered_stderr = filter_harmless_warnings(result.stderr)
        if not filtered_stderr.strip():
            return True
        real_error_patterns = [
            "CUDA out of memory", "OutOfMemoryError", "RuntimeError",
            "FileNotFoundError", "Permission denied", "No such file or directory",
            "Segmentation fault", "Killed", "Fatal error", "Traceback (most recent call last):",
            "ImportError", "ModuleNotFoundError", "OSError", "MemoryError",
            "SystemError", "KeyboardInterrupt", "BrokenPipeError", "ConnectionError", "TimeoutError"
        ]
        for pattern in real_error_patterns:
            if pattern in filtered_stderr:
                return False

    return result.returncode == 0


# ---------------------------------------------
#  JSON Parsing
# ---------------------------------------------

def _parse_json_core(data, json_path):
    """Compute task-level statistics (token count, residue/nucleotide
    lengths, ligand count) from an already-parsed AF3 JSON dict.
    Accepts both the modern "type": "..." and the legacy nested
    ({"protein": {...}}) JSON shapes."""
    protein_length = 0
    ligand_count   = 0
    rna_length     = 0
    dna_length     = 0
    sequences = data.get('sequences', [])

    for seq_entry in sequences:
        seq_type = seq_entry.get('type', '').lower()
        count = max(1, int(seq_entry.get('count', 1)))

        if seq_type == 'protein' or 'protein' in seq_entry:
            p = seq_entry if seq_type == 'protein' else seq_entry['protein']
            protein_length += len(p.get('sequence', '')) * count
        elif seq_type == 'rna' or 'rna' in seq_entry:
            r = seq_entry if seq_type == 'rna' else seq_entry['rna']
            rna_length += len(r.get('sequence', '')) * count
        elif seq_type == 'dna' or 'dna' in seq_entry:
            d = seq_entry if seq_type == 'dna' else seq_entry['dna']
            dna_length += len(d.get('sequence', '')) * count
        elif seq_type == 'ligand' or 'ligand' in seq_entry:
            ligand_count += count

    token_count = count_tokens_from_af3_json(data)

    return {
        'name':             data.get('name', 'Unknown'),
        'token_count':      token_count,
        'protein_length':   protein_length,
        'rna_length':       rna_length,
        'dna_length':       dna_length,
        'ligand_count':     ligand_count,
        'total_sequences':  len(sequences),
        'dialect':          data.get('dialect', 'unknown'),
        'version':          data.get('version', 0),
        'json_input_file':  json_path.name,
        'json_filepath':    str(json_path),
        '_json_path':       str(json_path),
    }


def parse_json_file(json_path):
    """Read an AF3 input JSON from disk and return its task statistics, or
    None and a warning on parse failure."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return _parse_json_core(data, json_path)
    except Exception as e:
        warning(f"Failed to parse {json_path}: {e}")
        return None


# ---------------------------------------------
#  File Collection (Parallel)
# ---------------------------------------------

def _parse_single_json_for_process(json_path_str):
    """Multiprocessing-safe variant of parse_json_file (silently
    returns None on failure so a worker error never poisons the pool)."""
    try:
        json_path = Path(json_path_str)
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return _parse_json_core(data, json_path)
    except Exception:
        return None


def collect_json_files(input_dir, output_dir, skip_existing=True,
                       verbose=False, num_workers=None):
    """Scan input_dir for *.json files and split them into pending
    vs. already-completed tasks.

    For <=10 files, parsing runs single-threaded; for larger inputs, JSON
    parsing is offloaded to a multiprocessing pool (fork on Linux,
    spawn elsewhere) with a live progress + ETA readout. When
    skip_existing is set, jobs whose output folder already exists are
    redirected into the skipped_files list so they can be recorded in
    the result TSV without rerunning AF3.
    """
    info(f"Scanning JSON files in: {input_dir}")
    json_files = list(input_dir.glob("*.json"))
    if not json_files:
        error(f"No JSON files found in {input_dir}")
        sys.exit(1)
    info(f"Found {len(json_files)} JSON files")

    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    num_workers = min(num_workers, len(json_files), multiprocessing.cpu_count())

    if skip_existing:
        info("Checking for existing outputs...")
        existing_names = set()
        output_path = Path(output_dir)
        try:
            for folder in output_path.iterdir():
                if folder.is_dir() and any(folder.iterdir()):
                    existing_names.add(folder.name.lower())
        except (PermissionError, FileNotFoundError):
            pass
        if existing_names:
            info(f"Found {len(existing_names)} existing output folders")

    if len(json_files) <= 10:
        info("Small file count - using single-threaded parsing")
        pending_files = []
        skipped_files = []
        for json_file in json_files:
            json_info = parse_json_file(json_file)
            if json_info is not None:
                if skip_existing and check_output_exists(output_dir, json_info, verbose):
                    skipped_files.append((json_file, json_info))
                else:
                    pending_files.append((json_file, json_info))
        info(f"Pending tasks: {len(pending_files)}")
        if skipped_files:
            info(f"Skipped tasks: {len(skipped_files)}")
        return pending_files, skipped_files

    info(f"Parsing JSON files with {num_workers} parallel processes...")
    json_paths_str = [str(jf) for jf in json_files]
    pending_files = []
    skipped_files = []
    parse_errors = 0
    start_time = time.monotonic()

  
    import platform as _platform
    _preferred_ctx = 'fork' if _platform.system() == 'Linux' else 'spawn'
    try:
        ctx = multiprocessing.get_context(_preferred_ctx)
    except ValueError:
        ctx = multiprocessing.get_context('spawn')
    info(f"Multiprocessing start method: {_preferred_ctx}")

    with ctx.Pool(processes=num_workers) as pool:
        completed = 0
        total = len(json_paths_str)
        last_progress = -1

        for json_info in pool.imap_unordered(_parse_single_json_for_process, json_paths_str, chunksize=10):
            completed += 1
            progress = int(completed / total * 100)
            current_time = time.monotonic()
            if progress >= last_progress + 2 or completed % 5 == 0 or completed == total:
                elapsed = current_time - start_time
                if completed > 0:
                    rate = completed / elapsed
                    eta = (total - completed) / rate if rate > 0 else 0
                    print(f"\r  Parsing: {completed}/{total} ({progress:3d}%) | "
                          f"Elapsed: {elapsed:.1f}s | Speed: {rate:.1f} files/s | ETA: {eta:.1f}s   ", end='', flush=True)
                last_progress = progress

            if json_info is None:
                parse_errors += 1
                continue
            json_file = Path(json_info['_json_path'])
            if skip_existing:
                job_name = json_info.get('name', 'Unknown')
                sanitized_name = job_name.replace(' ', '_').replace('/', '_').replace('\\', '_').lower()
                if sanitized_name in existing_names:
                    skipped_files.append((json_file, json_info))
                    continue
            pending_files.append((json_file, json_info))

    elapsed = time.monotonic() - start_time
    rate = total / elapsed if elapsed > 0 else 0
    print(f"\r  Parsing: {completed}/{total} (100%) | Time: {elapsed:.1f}s | Avg Speed: {rate:.1f} files/s          ")

    if parse_errors > 0:
        warning(f"Failed to parse {parse_errors} file(s)")
    info(f"Pending tasks: {len(pending_files)}")
    if skipped_files:
        info(f"Skipped tasks: {len(skipped_files)}")
    return pending_files, skipped_files


# ---------------------------------------------
#  Task Execution
# ---------------------------------------------

def run_alphafold3_task(task, singularity_image, af3_db_path, models_path,
                        output_path, extra_args=None, strict_errors=False,
                        task_timeout=7200, gpu_id=0):
    """Run AlphaFold 3 inference for a single task inside the singularity image.

    Pins the subprocess to gpu_id via CUDA_VISIBLE_DEVICES, bind-
    mounts the AF3 database, models directory and per-GPU working
    directory into the container, runs run_alphafold.py, and tracks
    peak GPU memory through a GPUMonitor polling at 30-second
    intervals.

    Honours the global concurrency cap (_TASK_CONCURRENCY_SEMAPHORE) when
    set by --max-concurrent-tasks, so many small-token tasks cannot
    collectively exhaust system RAM.

    Returns (success, runtime_seconds, peak_memory_mb).
    """
    json_file = task.json_file
    json_filename = json_file.name
    input_dir = os.path.abspath(json_file.parent)
    output_dir = os.path.abspath(output_path)
    af3_db_abs = os.path.abspath(af3_db_path)
    models_abs = os.path.abspath(models_path)

    command = [
        "singularity", "exec", "--nv",
        "--env", f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "--bind", f"{af3_db_abs}:/root/public_databases",
        "--bind", f"{input_dir}:/root/af_input",
        "--bind", f"{output_dir}:/root/af_output",
        "--bind", f"{models_abs}:/root/models",


        singularity_image,
        "python", "run_alphafold.py",
        f"--json_path=/root/af_input/{json_filename}",
        "--model_dir=/root/models",
        "--db_dir=/root/public_databases",
        "--output_dir=/root/af_output"
    ]
    if extra_args:
        command.extend(extra_args)


    _sem = _TASK_CONCURRENCY_SEMAPHORE
    if _sem is not None:
        if not _sem.acquire(timeout=0):  # non-blocking probe for the debug log
            debug(f"[GPU{gpu_id}] {task.task_id} waiting for global concurrency slot...")
            _sem.acquire()

    try:
        info(f"[GPU{gpu_id}] Running {task.task_id}: {json_filename} "
             f"(tokens={task.json_info.get('token_count','?')}, "
             f"est.mem={task.estimated_memory}MB)")
        debug(f"Command: {' '.join(command)}")
        os.makedirs(output_dir, exist_ok=True)


        gpu_monitor = GPUMonitor(interval=30, gpu_id=gpu_id)
        initial_memory = gpu_monitor.get_current_memory_usage()
        gpu_monitor.start_monitoring()

        start_time = time.monotonic()

        try:
            result = subprocess.run(command, capture_output=True, text=True,
                                    timeout=task_timeout, cwd=os.getcwd())
            end_time = time.monotonic()
            runtime = end_time - start_time
            gpu_monitor.stop_monitoring()

            if result.returncode != 0:
                warning(f"[GPU{gpu_id}] Task {task.task_id} exited with code {result.returncode}")
                if result.stderr:
                    filtered = filter_harmless_warnings(result.stderr)
                    if filtered:
                        error(f"[GPU{gpu_id}] Task {task.task_id} errors:\n{filtered[-1500:]}")


            success_status = is_task_successful(output_dir, result, strict_errors, task=task)
            status_word = "completed" if success_status else "FAILED"
            (info if success_status else warning)(f"[GPU{gpu_id}] {task.task_id} {status_word} in {runtime:.1f}s")

            peak_memory = max(gpu_monitor.peak_memory - initial_memory, 0)
            return success_status, runtime, peak_memory

        except subprocess.TimeoutExpired:
            warning(f"[GPU{gpu_id}] Task {task.task_id} timed out after {task_timeout}s")
            try:
                gpu_monitor.stop_monitoring()
                peak_memory = max(gpu_monitor.peak_memory - initial_memory, 0)
            except Exception:
                peak_memory = 0
            return False, float(task_timeout), peak_memory

        except Exception as e:
            error(f"[GPU{gpu_id}] Task {task.task_id} error: {e}")
            try:
                gpu_monitor.stop_monitoring()
                peak_memory = max(gpu_monitor.peak_memory - initial_memory, 0)
            except Exception:
                peak_memory = 0
            return False, 0.0, peak_memory

    finally:
        if _sem is not None:
            _sem.release()


def run_batch_parallel(batch, singularity_image, af3_db_path, models_path,
                       output_path, extra_args=None, strict_errors=False,
                       max_workers=None, task_timeout=7200, gpu_id=0,
                       on_task_complete=None):
    """Run all tasks of a regular TaskBatch concurrently on one GPU.

    Spawns one ThreadPoolExecutor worker per task (capped by
    max_workers); each worker invokes run_alphafold3_task and
    streams its result through on_task_complete so the per-task TSV
    log is updated live."""
    workers = len(batch.tasks) if max_workers is None else min(max_workers, len(batch.tasks))
    info(f"[GPU{gpu_id}] Batch {batch.batch_id}: {len(batch.tasks)} tasks, "
         f"{batch.total_memory}MB est. memory")

    results = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                run_alphafold3_task, task, singularity_image,
                af3_db_path, models_path, output_path, extra_args,
                strict_errors, task_timeout, gpu_id
            ): task for task in batch.tasks
        }
        for future in as_completed(futures):
            task = futures[future]
            try:
                ok, runtime, peak_mem = future.result()
                results.append((task, ok, runtime, peak_mem))
                if on_task_complete:
                    on_task_complete(task, ok, runtime, peak_mem, gpu_id, batch, '')
            except Exception as e:
                error(f"[GPU{gpu_id}] Task {task.task_id} exception: {e}")
                results.append((task, False, 0.0, 0))
                if on_task_complete:
                    on_task_complete(task, False, 0.0, 0, gpu_id, batch, '')

    success_count = sum(1 for _, ok, _, _ in results if ok)
    info(f"[GPU{gpu_id}] Batch {batch.batch_id} done: {success_count}/{len(results)} succeeded")
    return results


def run_temporal_wave_batch(batch, singularity_image, af3_db_path, models_path,
                            output_path, extra_args=None, strict_errors=False,
                            max_workers=None, task_timeout=7200, gpu_id=0,
                            on_task_complete=None):
    """Execute one TemporalWaveBatch: launch the anchor task(s), then
    dispatch each successive wave of lightweight tasks while the anchors
    are still running. Once all anchors finish, any waves still queued
    are NOT silently dropped: their tasks are returned as a separate
    list of "deferred wave tasks" so the caller can re-run them in a
    later parallel pass (rather than losing them). The number of
    concurrent threads is sized for len(anchors) + max_wave_size.

    Returns:
        (all_results, deferred_wave_tasks)
        - all_results: list of (task, ok, runtime, peak_mem) for tasks that
          actually ran in this batch (anchors + completed waves).
        - deferred_wave_tasks: list of PredictionTask objects from waves that
          were skipped because all anchors finished before those waves were
          dispatched. These tasks have NOT been run yet and must be
          re-scheduled by the caller.
    """
    anchor_tasks = batch.anchor_tasks
    waves = batch.waves
    total_wt = sum(len(w.tasks) for w in waves)

    if len(anchor_tasks) == 1:
        anc_desc = (f"anchor={anchor_tasks[0].task_id} "
                    f"(tokens={anchor_tasks[0].json_info.get('token_count','?')}, "
                    f"est.rt={batch.estimated_anchor_runtime:.0f}s, "
                    f"mem={batch.anchor_group_memory}MB)")
    else:
        ids = '+'.join(t.task_id for t in anchor_tasks)
        anc_desc = (f"anchors=[{ids}] "
                    f"({len(anchor_tasks)} tasks, "
                    f"window={batch.estimated_anchor_runtime:.0f}s, "
                    f"group_mem={batch.anchor_group_memory}MB)")

    info(f"[GPU{gpu_id}] TemporalWave {batch.batch_id}: {anc_desc} | "
         f"{len(waves)} waves, {total_wt} wave tasks | "
         f"wave_budget={batch.wave_memory_budget}MB")

    max_wave_size = max((len(w.tasks) for w in waves), default=0)
    needed = len(anchor_tasks) + max_wave_size

    if max_workers is None:
        workers = needed
    else:
        workers = min(needed, max(len(anchor_tasks), max_workers))
        if workers < needed:
            info(f"[GPU{gpu_id}] {batch.batch_id}: capping concurrency "
                 f"{needed} -> {workers} (--max-workers={max_workers})")

    all_results = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        anchor_futures = {
            executor.submit(
                run_alphafold3_task, anc, singularity_image,
                af3_db_path, models_path, output_path, extra_args,
                strict_errors, task_timeout, gpu_id
            ): anc for anc in anchor_tasks
        }

        waves_completed = 0
        waves_skipped = 0
        deferred_wave_tasks: List[PredictionTask] = []

        for wave_idx, wave in enumerate(waves):
            if all(f.done() for f in anchor_futures):
                # All anchors finished before we got to dispatch this wave.
                # Collect every task from waves[wave_idx:] as DEFERRED so the
                # caller can re-run them later (instead of silently losing
                # them, which is the original bug).
                remaining_waves = waves[wave_idx:]
                waves_skipped = len(remaining_waves)
                for rw in remaining_waves:
                    deferred_wave_tasks.extend(rw.tasks)
                info(f"[GPU{gpu_id}] {batch.batch_id}: all anchors done early, "
                     f"deferring {waves_skipped} remaining wave(s) "
                     f"({len(deferred_wave_tasks)} task(s)) for end-of-GPU "
                     f"parallel rerun")
                break

            info(f"[GPU{gpu_id}] {batch.batch_id} {wave.wave_id}: "
                 f"launching {len(wave.tasks)} tasks "
                 f"({wave.total_memory}MB, ~{wave.estimated_max_runtime:.0f}s)")

            wave_futures = {
                executor.submit(
                    run_alphafold3_task, task, singularity_image,
                    af3_db_path, models_path, output_path, extra_args,
                    strict_errors, task_timeout, gpu_id
                ): task for task in wave.tasks
            }

            for future in as_completed(wave_futures):
                task = wave_futures[future]
                try:
                    ok, runtime, peak_mem = future.result()
                    all_results.append((task, ok, runtime, peak_mem))
                    if on_task_complete:
                        on_task_complete(task, ok, runtime, peak_mem, gpu_id, batch, wave.wave_id)
                except Exception as e:
                    error(f"[GPU{gpu_id}] Wave task {task.task_id} error: {e}")
                    all_results.append((task, False, 0.0, 0))
                    if on_task_complete:
                        on_task_complete(task, False, 0.0, 0, gpu_id, batch, wave.wave_id)

            wave_ok = sum(1 for t, ok, _, _ in all_results
                         if ok and t.task_id in {x.task_id for x in wave.tasks})
            info(f"[GPU{gpu_id}] {batch.batch_id} {wave.wave_id}: {wave_ok}/{len(wave.tasks)} ok")
            waves_completed += 1

        for anchor_future, anc in anchor_futures.items():
            try:
                ok, runtime, peak_mem = anchor_future.result()
                all_results.append((anc, ok, runtime, peak_mem))
                status = "OK" if ok else "FAILED"
                info(f"[GPU{gpu_id}] {batch.batch_id} anchor {anc.task_id}: {status} in {runtime:.1f}s")
                if on_task_complete:
                    on_task_complete(anc, ok, runtime, peak_mem, gpu_id, batch, 'anchor')
            except Exception as e:
                error(f"[GPU{gpu_id}] Anchor task {anc.task_id} error: {e}")
                all_results.append((anc, False, 0.0, 0))
                if on_task_complete:
                    on_task_complete(anc, False, 0.0, 0, gpu_id, batch, 'anchor')

    total_ok = sum(1 for _, ok, _, _ in all_results if ok)
    info(f"[GPU{gpu_id}] TemporalWave {batch.batch_id} DONE: "
         f"{total_ok}/{len(all_results)} succeeded | "
         f"{waves_completed} waves run, {waves_skipped} skipped"
         + (f" ({len(deferred_wave_tasks)} task(s) deferred)" if deferred_wave_tasks else ""))
    return all_results, deferred_wave_tasks


def run_gpu_worker(worker, singularity_image, af3_db_path, models_path,
                    output_path, extra_args, strict_errors, max_workers,
                    task_timeout, input_dir=None, result_writer=None,
                    optimizer=None):
    """Drive one GPU through its full batch queue, then a two-stage retry pass.

    Stages:
      1. Stage every assigned JSON into the per-GPU working directory so
         the singularity bind mount sees only this GPU's inputs (avoids
         cross-GPU JSON contention).
      2. Walk the batch list sequentially, invoking either
         run_temporal_wave_batch or run_batch_parallel per batch.
         Collect:
           - failed_tasks: tasks that ran and failed.
           - deferred_wave_tasks: tasks from temporal-wave batches whose
             waves were skipped because all anchors finished before the
             waves were dispatched (these tasks have NOT been run yet).
      3. Parallel rerun pass: combine deferred_wave_tasks + failed_tasks
         into a single rerun pool, repack into regular Next-Fit
         Decreasing batches via the optimizer (with use_temporal_waves=
         False so we cannot lose tasks again), and run those rerun
         batches in parallel on this GPU. Anything that still fails
         becomes a "still_failed" candidate for stage 4.
      4. Final per-task retry pass: rerun each still-failed task
         individually as a last resort.
      5. On the way out, restore staged JSON files to the original input
         directory and remove the working directory if empty.

    Returns a per-GPU summary dict consumed by the top-level reporter.
    """
    gpu_id = worker.gpu_id
    info(f"\n{'='*60}")
    info(f"[GPU{gpu_id}] Starting with {len(worker.tasks)} tasks in {len(worker.batches)} batches")
    info(f"[GPU{gpu_id}] Total tokens: {worker.total_tokens:,}")
    info(f"{'='*60}")

    worker_dir = worker.working_dir
    worker_dir.mkdir(parents=True, exist_ok=True)

    original_json_paths = {}
    moved_files = []
    for task in worker.tasks:
        dest = worker_dir / task.json_file.name
        if task.json_file != dest and not dest.exists():
            original_path = str(task.json_file)
            shutil.move(original_path, str(dest))
            task.json_file = dest
            original_json_paths[dest] = original_path
            moved_files.append(task)

    if moved_files:
        info(f"[GPU{gpu_id}] Moved {len(moved_files)} JSON files to working directory")

    batch_trackers = {}

    def on_task_complete(task, ok, runtime, peak_mem, gid, batch, wave_id=''):
        if batch.batch_id not in batch_trackers:
            batch_trackers[batch.batch_id] = {
                'start_time': time.monotonic(), 'peak_memory': 0, 'tasks': []
            }
        tracker = batch_trackers[batch.batch_id]
        tracker['peak_memory'] = max(tracker['peak_memory'], peak_mem)
        tracker['tasks'].append((task, ok, runtime, peak_mem))
        batch_runtime = time.monotonic() - tracker['start_time']

        if isinstance(batch, TemporalWaveBatch):
            btype = 'temporal_anchor' if wave_id == 'anchor' else 'temporal_wave'
        else:
            btype = 'normal'

        if result_writer:
            result_writer.write_task_result(
                task=task, ok=ok, runtime=runtime, peak_mem=peak_mem,
                gpu_id=gid, batch_id=batch.batch_id,
                batch_peak_memory=tracker['peak_memory'],
                batch_runtime=batch_runtime, is_retry=False,
                batch_type=btype, wave_id=wave_id
            )

    overall_start = time.monotonic()
    total_ok = 0
    total_fail = 0
    failed_tasks = []
    deferred_wave_tasks: List[PredictionTask] = []
    all_results = []
    batch_metrics_map = {}

    for i, batch in enumerate(worker.batches, 1):
        batch_start = time.monotonic()
        is_temporal = isinstance(batch, TemporalWaveBatch)
        batch_type_label = "[TEMPORAL WAVE]" if is_temporal else "[NORMAL]"
        info(f"[GPU{gpu_id}] Batch {i}/{len(worker.batches)}: {batch.batch_id} {batch_type_label}")

        if is_temporal:
            batch_results, batch_deferred = run_temporal_wave_batch(
                batch, singularity_image, af3_db_path, models_path, output_path,
                extra_args, strict_errors, max_workers, task_timeout, gpu_id,
                on_task_complete=on_task_complete
            )
            if batch_deferred:
                deferred_wave_tasks.extend(batch_deferred)
                info(f"[GPU{gpu_id}] Batch {batch.batch_id}: deferred "
                     f"{len(batch_deferred)} wave task(s) for parallel rerun")
        else:
            batch_results = run_batch_parallel(
                batch, singularity_image, af3_db_path, models_path, output_path,
                extra_args, strict_errors, max_workers, task_timeout, gpu_id,
                on_task_complete=on_task_complete
            )

        n_ok = sum(1 for _, ok, _, _ in batch_results if ok)
        n_fail = len(batch_results) - n_ok
        total_ok += n_ok
        total_fail += n_fail

        for task, ok, _, _ in batch_results:
            if not ok:
                failed_tasks.append(task)
        all_results.extend(batch_results)
        batch_time = time.monotonic() - batch_start

        batch_peak_memory = max((r[3] for r in batch_results if r[3] > 0), default=0)
        batch_metrics = BatchResultMetrics(
            batch_id=batch.batch_id, gpu_id=gpu_id,
            batch_peak_memory_mb=batch_peak_memory, batch_runtime_seconds=batch_time,
            task_count=len(batch_results), success_count=n_ok, fail_count=n_fail
        )
        batch_metrics_map[batch.batch_id] = batch_metrics

        info(f"[GPU{gpu_id}] Batch {batch.batch_id}: {n_ok} ok / {n_fail} failed | "
             f"peak_mem={batch_peak_memory}MB | time={batch_time:.1f}s")

        if i < len(worker.batches):
            time.sleep(1)

    # =========================================================
    # Stage 3: Parallel rerun pass for deferred + failed tasks
    # =========================================================
    # Only triggered when at least one TemporalWaveBatch deferred wave
    # tasks (i.e. its anchor finished before the wave was dispatched).
    # In that case the deferred wave tasks would otherwise be silently
    # lost; we combine them with any tasks that ran but failed,
    # repack the union into regular NFD parallel batches via the
    # optimizer (NO temporal waves here so we cannot lose any task
    # again), and run those batches in parallel on this GPU. Anything
    # that still fails after this stage drops down to the per-task
    # retry below.
    #
    # When NO wave tasks were deferred, behavior is unchanged: failed
    # tasks (if any) go straight to the per-task retry pass below.
    rerun_recovered = 0
    rerun_still_fail = 0
    rerun_batches_run = 0
    still_failed_tasks: List[PredictionTask] = []

    deferred_ids: Set[str] = {t.task_id for t in deferred_wave_tasks}
    rerun_pool: List[PredictionTask] = []
    if deferred_wave_tasks:
        # Deduplicate by task_id (a task can't be both deferred AND failed,
        # but we de-dup defensively in case of any future logic change).
        seen_ids: Set[str] = set()
        for t in itertools.chain(deferred_wave_tasks, failed_tasks):
            if t.task_id not in seen_ids:
                rerun_pool.append(t)
                seen_ids.add(t.task_id)

    if rerun_pool:
        # Pre-account: deferred tasks were never counted in total_ok/total_fail
        # (they didn't run). Treat them as "currently failing" for the duration
        # of the rerun stage so the running counters stay sensible; we'll
        # decrement total_fail for each one that succeeds below. Tasks coming
        # from `failed_tasks` are already counted in total_fail, so no extra
        # bookkeeping is needed for them.
        total_fail += len(deferred_wave_tasks)

        warning(f"[GPU{gpu_id}] Parallel rerun pool: "
                f"{len(deferred_wave_tasks)} deferred + {len(failed_tasks)} failed = "
                f"{len(rerun_pool)} task(s)")

        if optimizer is not None:
            # Reset transient flags so the optimizer doesn't keep these tasks
            # locked into solo "timeout_risk" / "vram_overflow" batches when
            # those flags don't apply at rerun time. We preserve the
            # original flags (which travel with the PredictionTask
            # dataclass) - the optimizer reads them as-is.
            try:
                rerun_batches = optimizer.create_optimal_batches(
                    rerun_pool, use_temporal_waves=False
                )
            except Exception as e:
                error(f"[GPU{gpu_id}] Rerun batch packing failed: {e}; "
                      f"falling back to one solo batch per task")
                rerun_batches = [
                    TaskBatch(tasks=[t], total_memory=t.estimated_memory,
                              estimated_max_runtime=t.estimated_runtime,
                              batch_id=f"rerun_solo_{t.task_id}")
                    for t in rerun_pool
                ]
        else:
            # No optimizer was provided (older callers): fall back to
            # one task per batch so we still parallel-rerun safely on
            # this GPU.
            warning(f"[GPU{gpu_id}] No optimizer passed to run_gpu_worker; "
                    f"running rerun pool as solo batches")
            rerun_batches = [
                TaskBatch(tasks=[t], total_memory=t.estimated_memory,
                          estimated_max_runtime=t.estimated_runtime,
                          batch_id=f"rerun_solo_{t.task_id}")
                for t in rerun_pool
            ]

        # Tag rerun batch ids so they don't collide with original ids in TSV.
        for rb_idx, rb in enumerate(rerun_batches, 1):
            if not rb.batch_id.startswith('rerun_'):
                rb.batch_id = f"rerun_{rb_idx:03d}_{rb.batch_id}"

        info(f"[GPU{gpu_id}] Parallel rerun: {len(rerun_batches)} batch(es) "
             f"across {len(rerun_pool)} task(s)")

        def on_rerun_task_complete(task, ok, runtime, peak_mem, gid, batch, wave_id=''):
            # Mirror on_task_complete bookkeeping but stamp batch_type='rerun_parallel'
            # and is_retry=True for tasks that previously failed, False for
            # tasks that were merely deferred (their first actual run).
            if batch.batch_id not in batch_trackers:
                batch_trackers[batch.batch_id] = {
                    'start_time': time.monotonic(), 'peak_memory': 0, 'tasks': []
                }
            tracker = batch_trackers[batch.batch_id]
            tracker['peak_memory'] = max(tracker['peak_memory'], peak_mem)
            tracker['tasks'].append((task, ok, runtime, peak_mem))
            batch_runtime = time.monotonic() - tracker['start_time']

            is_retry_flag = task.task_id not in deferred_ids
            if result_writer:
                result_writer.write_task_result(
                    task=task, ok=ok, runtime=runtime, peak_mem=peak_mem,
                    gpu_id=gid, batch_id=batch.batch_id,
                    batch_peak_memory=tracker['peak_memory'],
                    batch_runtime=batch_runtime, is_retry=is_retry_flag,
                    batch_type='rerun_parallel', wave_id=''
                )

        for rb_idx, rb in enumerate(rerun_batches, 1):
            rb_start = time.monotonic()
            info(f"[GPU{gpu_id}] Rerun batch {rb_idx}/{len(rerun_batches)}: "
                 f"{rb.batch_id} ({len(rb.tasks)} tasks, "
                 f"~{rb.total_memory}MB, ~{rb.estimated_max_runtime/60:.1f}min)")
            rb_results = run_batch_parallel(
                rb, singularity_image, af3_db_path, models_path, output_path,
                extra_args, strict_errors, max_workers, task_timeout, gpu_id,
                on_task_complete=on_rerun_task_complete
            )
            rerun_batches_run += 1
            rb_time = time.monotonic() - rb_start
            rb_peak = max((r[3] for r in rb_results if r[3] > 0), default=0)
            batch_metrics_map[rb.batch_id] = BatchResultMetrics(
                batch_id=rb.batch_id, gpu_id=gpu_id,
                batch_peak_memory_mb=rb_peak, batch_runtime_seconds=rb_time,
                task_count=len(rb_results),
                success_count=sum(1 for _, ok, _, _ in rb_results if ok),
                fail_count=sum(1 for _, ok, _, _ in rb_results if not ok),
            )
            for task, ok, runtime, peak_mem in rb_results:
                all_results.append((task, ok, runtime, peak_mem))
                if ok:
                    total_ok += 1
                    total_fail -= 1
                    rerun_recovered += 1
                else:
                    still_failed_tasks.append(task)

            n_rb_ok = sum(1 for _, ok, _, _ in rb_results if ok)
            n_rb_fail = len(rb_results) - n_rb_ok
            info(f"[GPU{gpu_id}] Rerun batch {rb.batch_id}: "
                 f"{n_rb_ok} ok / {n_rb_fail} failed | "
                 f"peak_mem={rb_peak}MB | time={rb_time:.1f}s")

            if rb_idx < len(rerun_batches):
                time.sleep(1)

        info(f"[GPU{gpu_id}] Parallel rerun complete: "
             f"{rerun_recovered} recovered / {len(still_failed_tasks)} still failing")

    # =========================================================
    # Stage 4: Final per-task retry pass (last resort)
    # =========================================================
    # Anything that failed in the parallel rerun (or, if the rerun stage
    # was skipped because there was nothing to defer / no optimizer, the
    # original failed_tasks list) is rerun ONE BY ONE serially. This is
    # the last-resort safety net to recover transient failures.
    retry_ok = 0
    retry_still_fail = 0

    final_retry_pool: List[PredictionTask]
    if rerun_pool:
        final_retry_pool = still_failed_tasks
    else:
        final_retry_pool = failed_tasks

    if final_retry_pool:
        warning(f"[GPU{gpu_id}] {len(final_retry_pool)} task(s) still failing - "
                f"retrying individually (one at a time)...")
        for r_idx, task in enumerate(final_retry_pool, 1):
            info(f"[GPU{gpu_id}] Retry {r_idx}/{len(final_retry_pool)}: {task.task_id}")

            ok, runtime, peak_mem = run_alphafold3_task(
                task, singularity_image, af3_db_path, models_path, output_path,
                extra_args, strict_errors, task_timeout=task_timeout, gpu_id=gpu_id
            )
            if result_writer:
                result_writer.write_task_result(
                    task=task, ok=ok, runtime=runtime, peak_mem=peak_mem,
                    gpu_id=gpu_id, batch_id=f"retry_{task.task_id}",
                    batch_peak_memory=peak_mem, batch_runtime=runtime,
                    is_retry=True, batch_type='retry', wave_id=''
                )
            if ok:
                retry_ok += 1
                total_ok += 1
                total_fail -= 1
                success(f"[GPU{gpu_id}] Retry succeeded: {task.task_id}")
            else:
                retry_still_fail += 1
                warning(f"[GPU{gpu_id}] Retry failed: {task.task_id}")

    total_time = time.monotonic() - overall_start

    restored_count = 0
    if original_json_paths and worker_dir.exists():
        try:
            for moved_path, orig_path in original_json_paths.items():
                if moved_path.exists():
                    orig_dest = Path(orig_path)
                    if orig_dest.exists():
                        info(f"[GPU{gpu_id}] Original file exists, removing copy: {moved_path.name}")
                    else:
                        shutil.move(str(moved_path), str(orig_dest))
                        restored_count += 1
            if restored_count > 0:
                info(f"[GPU{gpu_id}] Restored {restored_count} JSON files to input directory")
            remaining = list(worker_dir.glob("*.json"))
            if not remaining:
                try:
                    shutil.rmtree(worker_dir)
                    info(f"[GPU{gpu_id}] Removed temporary directory: {worker_dir}")
                except Exception as e:
                    warning(f"[GPU{gpu_id}] Could not remove temp dir: {e}")
            else:
                info(f"[GPU{gpu_id}] {len(remaining)} JSON files remain in {worker_dir}")
        except Exception as e:
            warning(f"[GPU{gpu_id}] Cleanup error: {e}")

    return {
        'gpu_id': gpu_id, 'total_ok': total_ok, 'total_fail': total_fail,
        'retry_ok': retry_ok, 'retry_still_fail': retry_still_fail,
        'rerun_recovered': rerun_recovered,
        'rerun_pool_size': len(rerun_pool),
        'rerun_batches_run': rerun_batches_run,
        'deferred_wave_tasks': len(deferred_wave_tasks),
        'total_time': total_time, 'all_results': all_results,
        'batch_metrics': batch_metrics_map, 'worker_dir': str(worker_dir)
    }


@dataclass
class BatchResultMetrics:
    """Aggregated metrics for one completed batch (peak VRAM, wall-clock
    runtime, success/failure counts), used for the post-run summary."""

    batch_id: str
    gpu_id: int
    batch_peak_memory_mb: int = 0
    batch_runtime_seconds: float = 0.0
    task_count: int = 0
    success_count: int = 0
    fail_count: int = 0


# ---------------------------------------------
#  Summary Printing
# ---------------------------------------------

def print_gpu_distribution_summary(workers, total_tasks):
    """Pretty-print per-GPU task counts, token loads, and deviation from
    the cross-GPU average (a quick sanity check on LPT balancing)."""
    print_colored("\n" + "="*80, Colors.CYAN)
    print_colored("MULTI-GPU TASK DISTRIBUTION SUMMARY", Colors.CYAN)
    print_colored("="*80, Colors.CYAN)
    total_tokens = sum(w.total_tokens for w in workers)
    avg_tokens = total_tokens / len(workers) if workers else 0
    info(f"Total GPUs used: {len(workers)}")
    info(f"Total tasks: {total_tasks}")
    info(f"Total tokens: {total_tokens:,}")
    info(f"Average tokens per GPU: {avg_tokens:,.0f}")
    print_colored("\nPer-GPU Distribution:", Colors.CYAN)
    for worker in workers:
        deviation = ((worker.total_tokens - avg_tokens) / avg_tokens * 100) if avg_tokens > 0 else 0
        print(f"  GPU {worker.gpu_id}: {len(worker.tasks):3d} tasks, "
              f"{worker.total_tokens:>8,} tokens ({deviation:+6.2f}% from avg), "
              f"{len(worker.batches):2d} batches")
    print_colored("="*80 + "\n", Colors.CYAN)


def print_optimization_summary(workers, total_tasks, max_memory_mb,
                               skipped_count=0, vram_overflow_count=0,
                               vram_overflow_token=None):
    """Pretty-print the per-GPU batch plan, distinguishing temporal-wave
    batches (with anchor / wave breakdown) from regular ones."""
    print_colored("\n" + "="*80, Colors.MAGENTA)
    print_colored("BATCH OPTIMIZATION SUMMARY (Multi-GPU)", Colors.MAGENTA)
    print_colored("="*80, Colors.MAGENTA)
    info(f"Total tasks to process: {total_tasks}")
    if skipped_count > 0:
        info(f"Skipped tasks: {skipped_count}")
    if vram_overflow_count > 0:
        warning(f"VRAM overflow tasks: {vram_overflow_count} (token >= {vram_overflow_token})")
    total_batches = sum(len(w.batches) for w in workers)
    total_tw = sum(1 for w in workers for b in w.batches if isinstance(b, TemporalWaveBatch))
    info(f"Total batches: {total_batches} ({total_tw} temporal-wave, {total_batches - total_tw} normal)")
    info(f"Available GPU memory: {max_memory_mb}MB per GPU")

    for worker in workers:
        if not worker.batches:
            continue
        print_colored(f"\nGPU {worker.gpu_id} Batches:", Colors.BLUE)
        for batch in worker.batches:
            if isinstance(batch, TemporalWaveBatch):
                anchor_tasks = batch.anchor_tasks
                wave_task_count = batch.wave_task_count
                wave_tokens_all = [t.json_info['token_count'] for w in batch.waves for t in w.tasks]
                wave_token_range = (f"{min(wave_tokens_all):,}-{max(wave_tokens_all):,}"
                                    if wave_tokens_all else "N/A")
                avg_wave_rt = (sum(w.estimated_max_runtime for w in batch.waves)
                               / max(len(batch.waves), 1))
                if len(anchor_tasks) == 1:
                    anc = anchor_tasks[0]
                    anc_desc = (f"anchor={anc.task_id} "
                                f"(tokens={anc.json_info['token_count']:,}, "
                                f"mem={batch.anchor_group_memory}MB, "
                                f"~{batch.estimated_anchor_runtime/60:.1f}min)")
                else:
                    ids = '+'.join(t.task_id for t in anchor_tasks)
                    anc_desc = (f"anchors=[{ids}] "
                                f"({len(anchor_tasks)} tasks, "
                                f"group_mem={batch.anchor_group_memory}MB, "
                                f"window~{batch.estimated_anchor_runtime/60:.1f}min)")
                print_colored(
                    f"  {batch.batch_id}: [TEMPORAL WAVE] {anc_desc} | "
                    f"wave_budget={batch.wave_memory_budget}MB | "
                    f"{len(batch.waves)} waves x ~{len(batch.waves[0].tasks) if batch.waves else 0} tasks "
                    f"= {wave_task_count} wave tasks | "
                    f"wave tokens {wave_token_range} | "
                    f"avg_wave_rt~{avg_wave_rt:.0f}s",
                    Colors.GREEN
                )
                for wave in batch.waves[:3]:
                    print(f"      {wave.wave_id}: {len(wave.tasks):2d} tasks | "
                          f"mem={wave.total_memory:>5}MB | ~{wave.estimated_max_runtime:.0f}s")
                if len(batch.waves) > 3:
                    print(f"      ... ({len(batch.waves) - 3} more waves)")
            else:
                tokens = [t.json_info['token_count'] for t in batch.tasks]
                vram_tag = ""
                if any(t.vram_overflow for t in batch.tasks):
                    vram_tag = " [VRAM OVERFLOW - CPU BUFFER]"
                print(f"  {batch.batch_id}: {len(batch.tasks):2d} tasks | "
                      f"mem={batch.total_memory:>5}MB | "
                      f"wall~{batch.estimated_max_runtime/60:>5.1f}min | "
                      f"tokens {min(tokens):>5,}-{max(tokens):>5,}{vram_tag}")
    print_colored("="*80 + "\n", Colors.MAGENTA)


def print_memory_step_summary(profile_loader):
    """Pretty-print the loaded memory-step table, marking which steps
    exceed the effective VRAM and will trigger CPU memory offloading."""
    step_summary = profile_loader.get_memory_step_summary()
    if not step_summary:
        return
    print_colored("\n" + "="*80, Colors.CYAN)
    print_colored(f"MEMORY ALLOCATION STEP SUMMARY ({profile_loader.profile_source_label})", Colors.CYAN)
    print_colored("="*80, Colors.CYAN)
    print(f"{'Step':<5} {'Token Range':<20} {'Memory (MB)':<15} {'Memory (GB)':<12} {'Status'}")
    print("-"*80)
    for i, (min_token, max_token, mem, exceeds) in enumerate(step_summary, 1):
        if max_token is None:
            token_range = f">= {min_token}"
        else:
            token_range = f"{min_token} - {max_token - 1}"
        status = "[!] CPU BUFFER" if exceeds else "[OK] GPU"
        memory_gb = mem / 1024
        print(f"{i:<5} {token_range:<20} {mem:<15,} {memory_gb:<12.1f} {status}")
    vram_threshold = profile_loader.get_vram_overflow_threshold()
    if vram_threshold:
        print("-"*80)
        warning(f"VRAM Overflow Threshold: token >= {vram_threshold}")
    print_colored("="*80 + "\n", Colors.CYAN)


# ---------------------------------------------
#  Singularity Test
# ---------------------------------------------

def test_singularity_command(singularity_image, af3_db_path, models_path,
                             input_dir, output_dir, gpu_id=0):
    """Smoke-test the singularity image and bind mounts by launching
    python --version inside the container; aborts the run early if
    the image, database, or models directory is misconfigured."""
    info(f"Testing singularity command on GPU {gpu_id}...")
    test_cmd = [
        "singularity", "exec", "--nv",
        "--env", f"CUDA_VISIBLE_DEVICES={gpu_id}",
        "--bind", f"{af3_db_path}:/root/public_databases",
        "--bind", f"{input_dir}:/root/af_input",
        "--bind", f"{output_dir}:/root/af_output",
        "--bind", f"{models_path}:/root/models",
        "--bind", af3_db_path,
        singularity_image,
        "python", "--version"
    ]
    try:
        result = subprocess.run(test_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            success(f"Singularity test passed on GPU {gpu_id}. Python: {result.stdout.strip()}")
            return True
        else:
            error(f"Singularity test failed (exit {result.returncode}): {result.stderr}")
            return False
    except Exception as e:
        error(f"Singularity test failed: {e}")
        return False


# ---------------------------------------------
#  Main
# ---------------------------------------------

def main():
    """CLI entry point. Validates arguments, loads the memory profile,
    smoke-tests the singularity image, parses input JSONs in parallel,
    distributes tasks across GPUs (LPT), launches one worker thread per
    GPU, and writes a streaming TSV log of all per-task results."""
    parser = argparse.ArgumentParser(
        description=(
            "AlphaFold3 Multi-GPU Parallel Executor\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -i ./af_input -o results.tsv --sif alphafold3.sif
  %(prog)s -i ./af_input -o results.tsv --sif alphafold3.sif --gpus 0,1,2,3
  %(prog)s -i ./af_input -o results.tsv --sif alphafold3.sif --min-anchor-ratio 3.0
  %(prog)s -i ./af_input -o results.tsv --sif alphafold3.sif --no-temporal-waves
        """
    )

    io_group = parser.add_argument_group('Input/Output Options')
    io_group.add_argument('-i', '--input-dir', type=Path, required=True, metavar='DIR')
    io_group.add_argument('-o', '--output-file', type=Path, required=True, metavar='FILE')
    io_group.add_argument('--output-dir', type=str, default='./af_output_parallel', metavar='DIR')
    io_group.add_argument('--no-skip-existing', action='store_true')
    io_group.add_argument('--skip-vram-overflow', action='store_true')
    io_group.add_argument('--max-tokens', type=int, default=None, metavar='N')
    io_group.add_argument('--temp-dir', type=str, default=None, metavar='DIR')

    af3_group = parser.add_argument_group('AlphaFold3 Configuration')
    af3_group.add_argument('--sif', '--singularity-image', type=str, required=True, metavar='FILE')
    af3_group.add_argument('--af3-db', type=str, default='./af3_DB', metavar='DIR')
    af3_group.add_argument('--models', type=str, default='./models', metavar='DIR')
    af3_group.add_argument('--norun-data-pipeline', action='store_true')
    af3_group.add_argument('--af3-extra-args', type=str, nargs='*', metavar='ARG')

    gpu_group = parser.add_argument_group('GPU Configuration')
    gpu_group.add_argument('--gpus', type=str, default=None, metavar='LIST')
    gpu_group.add_argument('--gpu-preset', type=str, default=None, metavar='PRESET', dest='gpu_preset')
    gpu_group.add_argument('--gpu-memory', type=int, default=DEFAULT_GPU_VRAM_MB, metavar='MB')
    gpu_group.add_argument('--vram-margin', type=float, default=0.95, metavar='RATIO')
    gpu_group.add_argument('--safety-margin', type=float, default=0.1, metavar='RATIO')
    gpu_group.add_argument('--max-workers', type=int, default=None, metavar='N',
                           help='Per-batch concurrency cap on a single GPU. ')
    gpu_group.add_argument('--max-concurrent-tasks', type=int, default=None, metavar='N',
                           dest='max_concurrent_tasks',
                           help='GLOBAL cap on concurrent AF3 subprocesses across ALL GPUs. '
                                'Prevents CPU RAM / swap exhaustion when many small-token '
                                'tasks fit in GPU VRAM but collectively overflow system memory. '
                                'Default: auto-derived from CPU RAM (see --cpu-memory-per-task-mb).')
    gpu_group.add_argument('--cpu-memory-per-task-mb', type=int, default=4500, metavar='MB',
                           dest='cpu_memory_per_task_mb',
                           help='Estimated CPU RAM per AF3 subprocess (default 4500MB, '
                                'matches observed RES on A800 nodes). Used for '
                                '--max-concurrent-tasks auto-derivation when not set explicitly.')
    gpu_group.add_argument('--cpu-memory-reserve-mb', type=int, default=8192, metavar='MB',
                           dest='cpu_memory_reserve_mb',
                           help='CPU RAM reserved for OS + python parent + nvidia-smi etc. '
                                '(default 8192MB). Used for --max-concurrent-tasks auto-derivation.')
    gpu_group.add_argument('--no-cpu-memory-autocap', action='store_true',
                           dest='no_cpu_memory_autocap',
                           help='Disable automatic --max-concurrent-tasks derivation. ')
    gpu_group.add_argument('--max-batch-runtime', type=float, default=7200.0, metavar='SECONDS')
    gpu_group.add_argument('--task-timeout', type=int, default=7200, metavar='SECONDS')

    sched_group = parser.add_argument_group('Temporal Wave Scheduling')
    sched_group.add_argument('--no-temporal-waves', action='store_true')
    sched_group.add_argument('--min-anchor-ratio', type=float, default=2.0, metavar='RATIO')
    sched_group.add_argument('--max-anchor-group-ratio', type=float, default=1.5, metavar='RATIO',
                             dest='max_anchor_group_ratio')

    profile_group = parser.add_argument_group('Memory Profiling Options')
    profile_group.add_argument('--memory-profile', type=Path, metavar='FILE')
    profile_group.add_argument('--no-profile-gap-fill', action='store_true', dest='no_profile_gap_fill')
    profile_group.add_argument('--memory-estimation-factor', type=float, default=1.0, metavar='FACTOR')

    monitor_group = parser.add_argument_group('Monitoring & Debug Options')
    monitor_group.add_argument('--monitor-interval', type=int, default=5, metavar='SECONDS')
    monitor_group.add_argument('--cpu-workers', type=int, default=None, metavar='N')
    monitor_group.add_argument('--verbose', '-v', action='store_true')
    monitor_group.add_argument('--strict-errors', action='store_true')
    monitor_group.add_argument('--test-only', action='store_true')

    args = parser.parse_args()


    set_verbose(args.verbose)

    preset_profile_key = 'a800'
    preset_label = ''

    if args.gpu_preset is not None:
        key = args.gpu_preset.lower().strip()
        if key not in GPU_PRESETS:
            preset_names = ', '.join(sorted(GPU_PRESETS.keys()))
            error(f"Unknown --gpu-preset '{key}'.  Available presets: {preset_names}")
            sys.exit(1)
        preset = GPU_PRESETS[key]
        preset_profile_key = preset['profile']
        preset_label = preset['label']
        if args.gpu_memory == DEFAULT_GPU_VRAM_MB:
            args.gpu_memory = preset['vram_mb']
        info(f"GPU preset '{key}' ({preset_label}): profile={preset_profile_key}, vram={args.gpu_memory}MB")

    if not (0.0 <= args.safety_margin <= 0.5):
        error("Safety margin must be between 0.0 and 0.5"); sys.exit(1)
    if not (0.5 <= args.vram_margin <= 1.0):
        error("VRAM margin must be between 0.5 and 1.0"); sys.exit(1)
    if args.gpu_memory < 1000:
        error("GPU memory must be at least 1000MB"); sys.exit(1)
    if not (0.5 <= args.memory_estimation_factor <= 5.0):
        error("Memory estimation factor must be between 0.5 and 5.0"); sys.exit(1)
    if not args.input_dir.exists() or not args.input_dir.is_dir():
        error(f"Input directory invalid: {args.input_dir}"); sys.exit(1)
    if not os.path.exists(args.sif):
        error(f"Singularity image not found: {args.sif}"); sys.exit(1)
    if args.min_anchor_ratio < 1.1:
        error("--min-anchor-ratio must be >= 1.1"); sys.exit(1)
    if args.max_anchor_group_ratio < 1.0:
        error("--max-anchor-group-ratio must be >= 1.0"); sys.exit(1)
    if args.max_tokens is not None and args.max_tokens < 1:
        error("--max-tokens must be a positive integer"); sys.exit(1)
    if args.task_timeout is not None and args.task_timeout < 60:
        error("--task-timeout must be at least 60 seconds"); sys.exit(1)
    if args.max_batch_runtime < 60:
        error("--max-batch-runtime must be at least 60 seconds"); sys.exit(1)

    available_gpus = detect_available_gpus()
    if not available_gpus:
        error("No GPUs detected! Please check nvidia-smi."); sys.exit(1)

    selected_gpus = parse_gpu_list(args.gpus)
    invalid_gpus = [g for g in selected_gpus if g not in available_gpus]
    if invalid_gpus:
        warning(f"GPUs {invalid_gpus} not found. Available: {available_gpus}")
        selected_gpus = [g for g in selected_gpus if g in available_gpus]
    if not selected_gpus:
        error("No valid GPUs selected!"); sys.exit(1)

    info(f"Available GPUs: {available_gpus}")
    info(f"Selected GPUs: {selected_gpus}")

    if args.gpu_preset is None and args.gpu_memory == DEFAULT_GPU_VRAM_MB:
        detected_vram = detect_gpu_vram_mb(selected_gpus[0])
        if detected_vram and detected_vram != DEFAULT_GPU_VRAM_MB:
            matched = _guess_gpu_preset_from_vram(detected_vram)
            if matched:
                preset_profile_key = GPU_PRESETS[matched]['profile']
                preset_label = GPU_PRESETS[matched]['label']
                info(f"Auto-detected GPU VRAM: {detected_vram} MB -> matched preset '{matched}' ({preset_label})")
            else:
                warning(f"Auto-detected GPU VRAM: {detected_vram} MB - no matching preset found.")
            args.gpu_memory = detected_vram

    af3_db_path = os.path.abspath(args.af3_db)
    models_path = os.path.abspath(args.models)
    output_path = os.path.abspath(args.output_dir)
    singularity_image = os.path.abspath(args.sif)
    input_dir = os.path.abspath(args.input_dir)

    if args.temp_dir:
        workspace_root = Path(args.temp_dir).resolve()
    else:
        workspace_root = Path(input_dir) / 'gpu_work'
    workspace_root.mkdir(parents=True, exist_ok=True)

    for path, name in [(af3_db_path, "AF3 database"), (models_path, "Models")]:
        if not os.path.exists(path):
            error(f"{name} directory not found: {path}"); sys.exit(1)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    gpu_monitor = GPUMonitor(args.monitor_interval)
    if not gpu_monitor.check_nvidia_smi():
        error("nvidia-smi not available"); sys.exit(1)
    try:

        subprocess.run(['singularity', '--version'],
                       capture_output=True, check=True, timeout=10)
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        error("Singularity not available in PATH"); sys.exit(1)

    use_temporal_waves = not args.no_temporal_waves
    min_anchor_ratio = args.min_anchor_ratio

    profile_loader = TokenMemoryProfileLoader(
        args.memory_profile, vram_margin=args.vram_margin,
        builtin_profile=preset_profile_key,
        profile_gap_fill=not getattr(args, 'no_profile_gap_fill', False),
    )
    profile_loader.set_gpu_vram(args.gpu_memory)

    print_colored("\n" + "="*80, Colors.MAGENTA)
    print_colored("ALPHAFOLD3 MULTI-GPU PARALLEL EXECUTOR", Colors.MAGENTA)
    print_colored(f"Token-Aware Distribution across {len(selected_gpus)} GPUs", Colors.MAGENTA)
    print_colored(f"Step-wise Memory Model ({len(profile_loader._memory_steps)} Discrete Token Ranges)", Colors.MAGENTA)
    print_colored(f"Profile: {profile_loader.profile_source_label}", Colors.MAGENTA)
    print_colored("Multi-Anchor TemporalWaveBatch", Colors.MAGENTA)
    print_colored("Profile Gap-Fill Interpolation", Colors.MAGENTA)
    print_colored("GPU VRAM Overflow Detection & Auto-isolation", Colors.MAGENTA)
    print_colored("True CPU Parallelism (ProcessPoolExecutor)", Colors.MAGENTA)
    print_colored("Streaming Write - Results Visible in Real-time", Colors.MAGENTA)
    if use_temporal_waves:
        print_colored(f"Temporal Wave Scheduling ENABLED (min_anchor={min_anchor_ratio:.1f}, "
                      f"max_group={args.max_anchor_group_ratio:.1f})", Colors.GREEN)
    else:
        print_colored("  Temporal Wave Scheduling DISABLED (--no-temporal-waves)", Colors.YELLOW)
    if preset_label:
        print_colored(f"GPU Preset: {preset_label}", Colors.CYAN)
    print_colored("="*80, Colors.MAGENTA)
    info(f"Input directory   : {input_dir}")
    info(f"Output file       : {args.output_file}")
    info(f"Output directory  : {output_path}")
    info(f"Workspace dir     : {workspace_root}")
    info(f"Skip existing     : {'DISABLED' if args.no_skip_existing else 'ENABLED'}")
    if args.skip_vram_overflow:
        info(f"Skip VRAM overflow: ENABLED")
    if args.max_tokens is not None:
        info(f"Max token filter  : tasks with token_count > {args.max_tokens:,} will be SKIPPED")
    info(f"Singularity image : {singularity_image}")
    info(f"AF3 database      : {af3_db_path}")
    info(f"Models directory  : {models_path}")
    info(f"GPUs used          : {selected_gpus}")
    info(f"GPU memory        : {args.gpu_memory}MB ({args.gpu_memory/1024:.1f}GB) per GPU")
    info(f"VRAM margin       : {args.vram_margin*100:.0f}%")
    info(f"Safety margin     : {args.safety_margin*100:.1f}%")
    info(f"Max batch runtime : {args.max_batch_runtime/60:.0f}min")
    info(f"Task hard timeout : {args.task_timeout/60:.0f}min")
    if use_temporal_waves:
        info(f"Temporal waves    : ENABLED (min_anchor_ratio={min_anchor_ratio:.1f})")
        info(f"  -> max_anchor_group_ratio={args.max_anchor_group_ratio:.1f}")
    else:
        info(f"Temporal waves    : DISABLED")
    info(f"Memory profile    : {profile_loader.profile_source_label}")

    print_memory_step_summary(profile_loader)

    extra_args = []
    if args.norun_data_pipeline:
        extra_args.append('--norun_data_pipeline')
    if args.af3_extra_args:
        extra_args.extend(args.af3_extra_args)
    if extra_args:
        info(f"Extra AF3 arguments: {' '.join(extra_args)}")

    if not test_singularity_command(singularity_image, af3_db_path, models_path,
                                    input_dir, output_path, selected_gpus[0]):
        error("Singularity test failed. Check your configuration."); sys.exit(1)

    cpu_count = multiprocessing.cpu_count()
    if args.cpu_workers is not None:
        cpu_workers = args.cpu_workers
        info(f"CPU workers for JSON parsing: {cpu_workers} (manually specified; machine has {cpu_count} logical CPUs)")
    else:
        cpu_workers = cpu_count
        info(f"CPU workers for JSON parsing: {cpu_workers} (auto-detected)")

    skip_existing = not args.no_skip_existing
    json_files, skipped_files = collect_json_files(
        args.input_dir, output_path, skip_existing, args.verbose, cpu_workers
    )

    if not json_files:
        if skipped_files:
            success("All tasks already have output. Nothing to do.")
        else:
            error("No tasks to process.")
        sys.exit(0)

    tasks = []
    token_skipped_files = []
    vram_overflow_count = 0
    for i, (json_file, json_info) in enumerate(json_files, 1):
        token_count = json_info['token_count']
        if args.max_tokens is not None and token_count > args.max_tokens:
            token_skipped_files.append((json_file, json_info))
            continue

        raw_memory = profile_loader.estimate_memory_mb(token_count)
        estimated_memory = int(raw_memory * args.memory_estimation_factor)
        estimated_runtime = profile_loader.estimate_runtime_seconds(token_count)
        timeout_risk = profile_loader.is_timeout_risk(token_count, args.task_timeout)
        over_vram = profile_loader.is_over_gpu_vram(token_count)

        if over_vram:
            vram_overflow_count += 1
            if args.skip_vram_overflow:
                token_skipped_files.append((json_file, json_info))
                continue

        task = PredictionTask(
            json_file=json_file, json_info=json_info,
            estimated_memory=estimated_memory, estimated_runtime=estimated_runtime,
            task_id=f"task_{i:04d}", timeout_risk=timeout_risk, vram_overflow=over_vram
        )
        tasks.append(task)

    info(f"Created {len(tasks)} prediction tasks")
    if token_skipped_files:
        warning(f"Token/VRAM filter: {len(token_skipped_files)} task(s) skipped")
    if not tasks:
        warning("No tasks remain after applying filters.")
        result_writer = StreamingResultWriter(args.output_file)
        result_writer.write_header()
        for json_file, ji in skipped_files:
            result_writer.write_skipped_task(json_file, ji, reason='skipped_existing_output')
        for json_file, ji in token_skipped_files:
            result_writer.write_skipped_task(json_file, ji, reason='skipped_token_limit')
        result_writer.close()
        sys.exit(0)

    if vram_overflow_count and not args.skip_vram_overflow:
        warning(f"VRAM overflow: {vram_overflow_count} task(s) exceed GPU VRAM ({args.gpu_memory}MB).")
        warning(f"  -> To skip them instead, add: --skip-vram-overflow")

    optimizer = DualDimensionTaskOptimizer(
        max_memory_mb=args.gpu_memory, safety_margin=args.safety_margin,
        max_batch_runtime_seconds=args.max_batch_runtime
    )

    tasks_by_gpu = distribute_tasks_by_tokens(tasks, len(selected_gpus))
    gpu_workers = create_gpu_workers(
        tasks_by_gpu, selected_gpus, optimizer, workspace_root,
        min_anchor_ratio=min_anchor_ratio, use_temporal_waves=use_temporal_waves,
        max_anchor_group_ratio=args.max_anchor_group_ratio,
    )

    _cleanup_state['gpu_workers'] = gpu_workers
    _cleanup_state['input_dir'] = input_dir

    vram_overflow_token = profile_loader.get_vram_overflow_threshold()
    print_gpu_distribution_summary(gpu_workers, len(tasks))
    print_optimization_summary(gpu_workers, len(tasks), args.gpu_memory,
                               len(skipped_files), vram_overflow_count, vram_overflow_token)


    global _TASK_CONCURRENCY_SEMAPHORE, _TASK_CONCURRENCY_CAP
    effective_cap = args.max_concurrent_tasks
    if effective_cap is None and not args.no_cpu_memory_autocap:
        try:
            import psutil
            total_ram_mb = psutil.virtual_memory().total // (1024 * 1024)
            usable_ram_mb = max(total_ram_mb - args.cpu_memory_reserve_mb,
                                args.cpu_memory_per_task_mb)
            auto_cap = max(1, usable_ram_mb // args.cpu_memory_per_task_mb)
            info(f"CPU-RAM auto-cap: total={total_ram_mb}MB, "
                 f"reserve={args.cpu_memory_reserve_mb}MB, "
                 f"per-task={args.cpu_memory_per_task_mb}MB -> "
                 f"max-concurrent-tasks={auto_cap}")
            effective_cap = auto_cap
        except ImportError:
            warning("psutil not installed - skipping CPU-RAM auto-cap. "
                    "Install with 'pip install psutil' or pass "
                    "--max-concurrent-tasks N explicitly to prevent "
                    "CPU RAM exhaustion on many-small-task workloads.")

    if effective_cap is not None and effective_cap > 0:
        _TASK_CONCURRENCY_SEMAPHORE = threading.Semaphore(effective_cap)
        _TASK_CONCURRENCY_CAP = effective_cap
        info(f"Global concurrency cap ACTIVE: at most {effective_cap} "
             f"AF3 subprocess(es) will run simultaneously across all GPUs.")
    else:
        info("Global concurrency cap DISABLED - tasks limited only by per-GPU VRAM.")

    if args.test_only:
        info("Test-only mode - exiting without running tasks.")
        sys.exit(0)

    result_writer = StreamingResultWriter(args.output_file)
    result_writer.write_header()

    if skipped_files:
        for json_file, ji in skipped_files:
            result_writer.write_skipped_task(json_file, ji, reason='skipped_existing_output')
        info(f"Recorded {len(skipped_files)} skipped tasks (existing output)")
    if token_skipped_files:
        for json_file, ji in token_skipped_files:
            token_count = ji['token_count']
            if args.skip_vram_overflow and profile_loader.is_over_gpu_vram(token_count):
                reason = 'skipped_vram_overflow'
            else:
                reason = 'skipped_token_limit'
            result_writer.write_skipped_task(json_file, ji, reason=reason)

    tw_count = sum(1 for w in gpu_workers for b in w.batches if isinstance(b, TemporalWaveBatch))
    tw_wave_tasks = sum(b.wave_task_count for w in gpu_workers
                        for b in w.batches if isinstance(b, TemporalWaveBatch))
    info(f"\nStarting multi-GPU execution with {len(gpu_workers)} workers...")
    if use_temporal_waves and tw_count > 0:
        info(f"{tw_count} TemporalWaveBatch(es) will process "
             f"{tw_wave_tasks} wave tasks in anchor VRAM shadow")
    info("Streaming mode - results written immediately as tasks complete")

    overall_start = time.monotonic()

    with ThreadPoolExecutor(max_workers=len(gpu_workers)) as executor:
        futures = {
            executor.submit(
                run_gpu_worker, worker, singularity_image,
                af3_db_path, models_path, output_path, extra_args,
                args.strict_errors, args.max_workers, args.task_timeout,
                input_dir, result_writer, optimizer
            ): worker for worker in gpu_workers
        }
        gpu_results = []
        for future in as_completed(futures):
            worker = futures[future]
            try:
                result = future.result()
                gpu_results.append(result)
                info(f"\n{'='*60}")
                success(f"GPU {worker.gpu_id} COMPLETED: "
                       f"{result['total_ok']} ok / {result['total_fail']} failed "
                       f"in {result['total_time']/3600:.2f}h")
                info(f"{'='*60}")
            except Exception as e:
                error(f"GPU {worker.gpu_id} execution error: {e}")
                gpu_results.append({
                    'gpu_id': worker.gpu_id, 'total_ok': 0,
                    'total_fail': len(worker.tasks), 'total_time': 0
                })

    total_time = time.monotonic() - overall_start
    result_writer.close()

    total_ok = sum(r['total_ok'] for r in gpu_results)
    total_fail = sum(r['total_fail'] for r in gpu_results)
    total_retry_ok = sum(r.get('retry_ok', 0) for r in gpu_results)
    total_rerun_recovered = sum(r.get('rerun_recovered', 0) for r in gpu_results)
    total_rerun_pool = sum(r.get('rerun_pool_size', 0) for r in gpu_results)
    total_deferred = sum(r.get('deferred_wave_tasks', 0) for r in gpu_results)

    print_colored("\n" + "="*80, Colors.GREEN)
    print_colored("MULTI-GPU EXECUTION COMPLETED", Colors.GREEN)
    print_colored("="*80, Colors.GREEN)
    success(f"Total runtime      : {total_time:.1f}s ({total_time/3600:.2f}h)")
    success(f"GPUs used          : {len(gpu_workers)}")
    success(f"Tasks processed    : {total_ok + total_fail}")
    if skipped_files:
        info(f"Tasks skipped      : {len(skipped_files)} (existing output)")
    success(f"Successful (total) : {total_ok}")
    if total_fail:
        warning(f"Final failures     : {total_fail}")
    if total_deferred:
        info(f"Wave tasks deferred: {total_deferred} (anchor finished early)")
    if total_rerun_pool:
        info(f"Parallel rerun pool: {total_rerun_pool} task(s)")
        if total_rerun_recovered:
            info(f"  -> Recovered      : {total_rerun_recovered} (parallel rerun)")
    if total_retry_ok:
        info(f"  -> Recovered      : {total_retry_ok} (per-task retry)")
    if total_ok + total_fail > 0:
        success(f"Success rate       : {total_ok/(total_ok+total_fail)*100:.1f}%")
    success(f"Results saved      : {args.output_file}")
    print_colored("="*80 + "\n", Colors.GREEN)


_cleanup_state = {
    'workers': [], 'input_dir': None, 'gpu_workers': [],
}

def restore_json_files_from_gpu_work(gpu_work_dir, input_dir):
    """Move JSON files from per-GPU working directories back to the
    original input directory after an interrupt, then remove the now-
    empty working subdirectories. Returns the count of files restored."""
    if not gpu_work_dir.exists():
        return 0
    restored = 0
    try:
        for gpu_subdir in gpu_work_dir.iterdir():
            if gpu_subdir.is_dir() and gpu_subdir.name.startswith('gpu_'):
                for json_file in gpu_subdir.glob('*.json'):
                    dest = input_dir / json_file.name
                    if not dest.exists():
                        shutil.move(str(json_file), str(dest))
                        restored += 1
                    else:
                        json_file.unlink()
        for gpu_subdir in gpu_work_dir.iterdir():
            if gpu_subdir.is_dir() and gpu_subdir.name.startswith('gpu_'):
                try:
                    if not any(gpu_subdir.iterdir()):
                        gpu_subdir.rmdir()
                except OSError:
                    pass
        if not any(gpu_work_dir.iterdir()):
            gpu_work_dir.rmdir()
    except Exception as e:
        warning(f"Error during restore: {e}")
    return restored

def signal_handler(signum, frame):
    """SIGINT/SIGTERM handler: best-effort restore of staged JSON files
    to the input directory before exiting, so an interrupt never strands
    user data inside the per-GPU working directories."""
    warning("\nInterrupt received - attempting to restore files...")
    try:
        if _cleanup_state.get('input_dir') and _cleanup_state.get('gpu_workers'):
            input_dir = Path(_cleanup_state['input_dir'])
            for worker in _cleanup_state['gpu_workers']:
                if hasattr(worker, 'working_dir') and worker.working_dir.exists():
                    restored = restore_json_files_from_gpu_work(worker.working_dir, input_dir)
                    if restored > 0:
                        info(f"Restored {restored} files from {worker.working_dir.name}")
    except Exception as e:
        warning(f"Cleanup error: {e}")
    warning("Cleanup attempted. Exiting...")
    sys.exit(130)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    main()
