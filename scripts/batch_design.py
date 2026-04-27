#!/usr/bin/env python3
"""
Batch Protein Design Pipeline

Multi-protein parallel design with task queue management, cluster support (SLURM/LSF),
result aggregation, error recovery, and progress tracking.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ClusterType(Enum):
    """Supported cluster schedulers."""
    LOCAL = "local"
    SLURM = "slurm"
    LSF = "lsf"


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class DesignTask:
    """Single protein design task."""
    task_id: str
    pdb_path: Path
    chain_id: str = ""
    protein_name: str = ""
    homologs_fasta: Optional[Path] = None
    asr_fasta: Optional[Path] = None
    family_manifest: Optional[Path] = None
    top_design_positions: int = 12
    report_positions: int = 24
    top_k: int = 12
    status: TaskStatus = TaskStatus.PENDING
    job_id: Optional[str] = None
    error_message: Optional[str] = None
    output_dir: Optional[Path] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class BatchConfig:
    """Batch processing configuration."""
    input_csv: Optional[Path] = None
    input_dir: Optional[Path] = None
    input_pattern: str = "*.pdb"
    output_root: Path = ROOT / "outputs" / "batch_design"
    max_parallel: int = 4
    cluster_type: ClusterType = ClusterType.LOCAL
    cluster_queue: str = "default"
    cluster_account: str = ""
    cluster_time: str = "24:00:00"
    cluster_mem: str = "16G"
    cluster_cpus: int = 8
    cluster_gpu: bool = True
    slurm_extra: str = ""
    lsf_extra: str = ""
    design_mode: bool = True
    resume: bool = False
    verbose: bool = False


@dataclass
class BatchResult:
    """Aggregated batch processing results."""
    batch_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    pending: int = 0
    running: int = 0
    total_duration: Optional[timedelta] = None
    tasks: list[DesignTask] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_tasks": self.total_tasks,
            "completed": self.completed,
            "failed": self.failed,
            "cancelled": self.cancelled,
            "pending": self.pending,
            "running": self.running,
            "total_duration_seconds": self.total_duration.total_seconds() if self.total_duration else None,
            "success_rate": self.completed / self.total_tasks if self.total_tasks > 0 else 0.0,
        }


class ProgressTracker:
    """Tracks and displays batch processing progress."""

    def __init__(self, total: int, verbose: bool = False):
        self.total = total
        self.verbose = verbose
        self.completed = 0
        self.failed = 0
        self.running = 0
        self.start_time = time.time()
        self._last_update = 0

    def update(self, completed: int = 0, failed: int = 0, running: int = 0) -> None:
        """Update progress counters."""
        self.completed = completed
        self.failed = failed
        self.running = running
        self._display()

    def _display(self) -> None:
        """Display progress bar."""
        current = time.time()
        if current - self._last_update < 2.0 and not self.verbose:
            return
        self._last_update = current

        elapsed = current - self.start_time
        rate = self.completed / elapsed if elapsed > 0 else 0
        eta = (self.total - self.completed - self.failed) / rate if rate > 0 else 0

        bar_width = 40
        filled = int(bar_width * self.completed / self.total) if self.total > 0 else 0
        bar = "#" * filled + "-" * (bar_width - filled)

        status = f"\r[{bar}] {self.completed}/{self.total} "
        status += f"(failed:{self.failed}, running:{self.running}) "
        status += f"ETA:{eta:.0f}s"

        sys.stdout.write(status)
        sys.stdout.flush()

    def finish(self) -> None:
        """Complete the progress display."""
        self._display()
        sys.stdout.write("\n")
        sys.stdout.flush()


class TaskQueue:
    """Manages the task queue for batch processing."""

    def __init__(self, tasks: list[DesignTask], max_parallel: int):
        self.tasks = tasks
        self.max_parallel = max_parallel
        self._completed: list[DesignTask] = []
        self._failed: list[DesignTask] = []
        self._lock_file: Optional[Path] = None

    def get_pending(self) -> list[DesignTask]:
        """Get pending tasks that can be started."""
        running = [t for t in self.tasks if t.status == TaskStatus.RUNNING]
        if len(running) >= self.max_parallel:
            return []
        pending = [t for t in self.tasks if t.status == TaskStatus.PENDING]
        return pending[: self.max_parallel - len(running)]

    def get_running(self) -> list[DesignTask]:
        """Get all running tasks."""
        return [t for t in self.tasks if t.status == TaskStatus.RUNNING]

    def mark_completed(self, task: DesignTask) -> None:
        """Mark a task as completed."""
        task.status = TaskStatus.COMPLETED
        task.end_time = datetime.now()
        self._completed.append(task)

    def mark_failed(self, task: DesignTask, error: str) -> None:
        """Mark a task as failed with error message."""
        task.status = TaskStatus.FAILED
        task.error_message = error
        task.end_time = datetime.now()
        self._failed.append(task)

    def retry_task(self, task: DesignTask) -> bool:
        """Retry a failed task if retries remain."""
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.PENDING
            task.error_message = None
            return True
        return False

    def save_state(self, state_file: Path) -> None:
        """Save queue state for recovery."""
        state = {
            "tasks": [
                {
                    "task_id": t.task_id,
                    "status": t.status.value,
                    "job_id": t.job_id,
                    "error_message": t.error_message,
                    "output_dir": str(t.output_dir) if t.output_dir else None,
                    "retry_count": t.retry_count,
                }
                for t in self.tasks
            ]
        }
        state_file.write_text(json.dumps(state, indent=2), encoding="utf-8")

    @classmethod
    def load_state(cls, state_file: Path, original_tasks: list[DesignTask]) -> "TaskQueue":
        """Load queue state from file."""
        state = json.loads(state_file.read_text(encoding="utf-8"))
        task_map = {t.task_id: t for t in original_tasks}
        for t_state in state["tasks"]:
            task = task_map.get(t_state["task_id"])
            if task:
                task.status = TaskStatus(t_state["status"])
                task.job_id = t_state.get("job_id")
                task.error_message = t_state.get("error_message")
                task.output_dir = Path(t_state["output_dir"]) if t_state.get("output_dir") else None
                task.retry_count = t_state.get("retry_count", 0)
        pending_tasks = [t for t in original_tasks if t.status == TaskStatus.PENDING]
        queue = cls(pending_tasks, max_parallel=4)
        queue._completed = [t for t in original_tasks if t.status == TaskStatus.COMPLETED]
        queue._failed = [t for t in original_tasks if t.status == TaskStatus.FAILED]
        return queue


class ClusterJobManager:
    """Manages cluster job submission and monitoring."""

    def __init__(self, cluster_type: ClusterType):
        self.cluster_type = cluster_type

    def submit_job(self, task: DesignTask, script_path: Path, config: BatchConfig) -> str:
        """Submit a job to the cluster scheduler."""
        if self.cluster_type == ClusterType.SLURM:
            return self._submit_slurm(task, script_path, config)
        elif self.cluster_type == ClusterType.LSF:
            return self._submit_lsf(task, script_path, config)
        else:
            raise ValueError(f"Unsupported cluster type: {self.cluster_type}")

    def _submit_slurm(self, task: DesignTask, script_path: Path, config: BatchConfig) -> str:
        """Submit job to SLURM."""
        cmd = [
            "sbatch",
            "--job-name", f"mars_design_{task.task_id[:8]}",
            "--output", str(task.output_dir / "slurm_out.txt"),
            "--error", str(task.output_dir / "slurm_err.txt"),
            "--time", config.cluster_time,
            "--mem", config.cluster_mem,
            "--cpus-per-task", str(config.cluster_cpus),
            "--account", config.cluster_account if config.cluster_account else None,
            "--partition", config.cluster_queue,
        ]
        if config.cluster_gpu:
            cmd.extend(["--gres", "gpu:1"])
        if config.slurm_extra:
            cmd.extend(config.slurm_extra.split())
        cmd.append(str(script_path))

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[-1]
        return job_id

    def _submit_lsf(self, task: DesignTask, script_path: Path, config: BatchConfig) -> str:
        """Submit job to LSF."""
        cmd = [
            "bsub",
            "-J", f"mars_design_{task.task_id[:8]}",
            "-o", str(task.output_dir / "lsf_out.txt"),
            "-e", str(task.output_dir / "lsf_err.txt"),
            "-W", config.cluster_time,
            "-M", config.cluster_mem,
            "-n", str(config.cluster_cpus),
            "-q", config.cluster_queue,
        ]
        if config.cluster_account:
            cmd.extend(["-P", config.cluster_account])
        if config.cluster_gpu:
            cmd.extend(["-R", "rusage[gpu=1]"])
        if config.lsf_extra:
            cmd.extend(config.lsf_extra.split())
        cmd.extend(["bash", str(script_path)])

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        job_id = result.stdout.strip().split()[1].strip("<>")
        return job_id

    def check_job_status(self, job_id: str) -> TaskStatus:
        """Check status of a cluster job."""
        if self.cluster_type == ClusterType.SLURM:
            return self._check_slurm_status(job_id)
        elif self.cluster_type == ClusterType.LSF:
            return self._check_lsf_status(job_id)
        return TaskStatus.UNKNOWN

    def _check_slurm_status(self, job_id: str) -> TaskStatus:
        """Check SLURM job status."""
        try:
            result = subprocess.run(
                ["squeue", "--job", job_id, "--noheader", "-o", "%T"],
                capture_output=True, text=True, check=True
            )
            status_map = {
                "RUNNING": TaskStatus.RUNNING,
                "PENDING": TaskStatus.PENDING,
                "COMPLETED": TaskStatus.COMPLETED,
                "FAILED": TaskStatus.FAILED,
                "CANCELLED": TaskStatus.CANCELLED,
            }
            return status_map.get(result.stdout.strip(), TaskStatus.PENDING)
        except subprocess.CalledProcessError:
            return TaskStatus.COMPLETED

    def _check_lsf_status(self, job_id: str) -> TaskStatus:
        """Check LSF job status."""
        try:
            result = subprocess.run(
                ["bjobs", "-noheader", "-o", "STAT", job_id],
                capture_output=True, text=True, check=True
            )
            status_map = {
                "RUN": TaskStatus.RUNNING,
                "PEND": TaskStatus.PENDING,
                "DONE": TaskStatus.COMPLETED,
                "EXIT": TaskStatus.FAILED,
                "USP": TaskStatus.FAILED,
            }
            return status_map.get(result.stdout.strip(), TaskStatus.PENDING)
        except subprocess.CalledProcessError:
            return TaskStatus.COMPLETED

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a cluster job."""
        try:
            if self.cluster_type == ClusterType.SLURM:
                subprocess.run(["scancel", job_id], check=True)
            elif self.cluster_type == ClusterType.LSF:
                subprocess.run(["bkill", job_id], check=True)
            return True
        except subprocess.CalledProcessError:
            return False


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Configure logging for the batch processor."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("batch_design")


def parse_input_csv(csv_path: Path) -> list[dict[str, Any]]:
    """Parse input CSV file containing protein design tasks."""
    import csv
    tasks = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_spec = {
                "pdb_path": Path(row["pdb_path"]),
                "chain_id": row.get("chain_id", ""),
                "protein_name": row.get("protein_name", ""),
                "homologs_fasta": Path(row["homologs_fasta"]) if row.get("homologs_fasta") else None,
                "asr_fasta": Path(row["asr_fasta"]) if row.get("asr_fasta") else None,
                "family_manifest": Path(row["family_manifest"]) if row.get("family_manifest") else None,
            }
            tasks.append(task_spec)
    return tasks


def discover_pdb_files(input_dir: Path, pattern: str = "*.pdb") -> list[Path]:
    """Discover PDB files in input directory."""
    return sorted(input_dir.glob(pattern))


def create_task_from_spec(spec: dict[str, Any], batch_id: str, output_root: Path) -> DesignTask:
    """Create a DesignTask from specification."""
    pdb_path = spec["pdb_path"]
    protein_name = spec.get("protein_name") or pdb_path.stem
    task_id = f"{batch_id}_{uuid.uuid4().hex[:8]}"

    output_dir = output_root / task_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return DesignTask(
        task_id=task_id,
        pdb_path=pdb_path,
        chain_id=spec.get("chain_id", ""),
        protein_name=protein_name,
        homologs_fasta=spec.get("homologs_fasta"),
        asr_fasta=spec.get("asr_fasta"),
        family_manifest=spec.get("family_manifest"),
        output_dir=output_dir,
    )


def generate_job_script(task: DesignTask, config: BatchConfig) -> Path:
    """Generate shell script for task execution."""
    script_path = task.output_dir / "run_task.sh"

    script_lines = [
        "#!/bin/bash",
        f"# Batch task: {task.task_id}",
        f"# PDB: {task.pdb_path}",
        "",
        'set -euo pipefail',
        "",
    ]

    autodesign_args = [
        str(ROOT / "scripts" / "run_mars_autodesign.py"),
        "design" if config.design_mode else "analyze",
        "--pdb", str(task.pdb_path),
        "--output-root", str(config.output_root),
        "--top-design-positions", str(task.top_design_positions),
        "--report-positions", str(task.report_positions),
        "--top-k", str(task.top_k),
    ]

    if task.chain_id:
        autodesign_args.extend(["--chain", task.chain_id])
    if task.protein_name:
        autodesign_args.extend(["--protein-name", task.protein_name])
    if task.homologs_fasta:
        autodesign_args.extend(["--homologs-fasta", str(task.homologs_fasta)])
    if task.asr_fasta:
        autodesign_args.extend(["--asr-fasta", str(task.asr_fasta)])
    if task.family_manifest:
        autodesign_args.extend(["--family-manifest", str(task.family_manifest)])

    script_lines.extend([
        f'echo "Starting task {task.task_id} at $(date)"',
        "cd " + str(ROOT),
        "",
        " ".join(autodesign_args),
        "",
        f'echo "Task {task.task_id} completed at $(date)"',
    ])

    script_path.write_text("\n".join(script_lines), encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def run_local_task(task: DesignTask, config: BatchConfig, logger: logging.Logger) -> bool:
    """Execute a design task locally."""
    logger.info(f"Starting local task: {task.task_id}")
    task.status = TaskStatus.RUNNING
    task.start_time = datetime.now()

    try:
        script_path = generate_job_script(task, config)
        logger.debug(f"Generated script: {script_path}")

        result = subprocess.run(
            ["bash", str(script_path)],
            capture_output=True,
            text=True,
            cwd=str(ROOT),
            timeout=7200,
        )

        if result.returncode == 0:
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            logger.info(f"Task {task.task_id} completed successfully")
            return True
        else:
            error_msg = result.stderr or f"Exit code: {result.returncode}"
            task.error_message = error_msg
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            logger.error(f"Task {task.task_id} failed: {error_msg}")
            return False

    except subprocess.TimeoutExpired:
        task.error_message = "Task timed out after 2 hours"
        task.status = TaskStatus.FAILED
        task.end_time = datetime.now()
        logger.error(f"Task {task.task_id} timed out")
        return False
    except Exception as e:
        task.error_message = str(e)
        task.status = TaskStatus.FAILED
        task.end_time = datetime.now()
        logger.exception(f"Task {task.task_id} encountered error")
        return False


def run_cluster_task(
    task: DesignTask,
    config: BatchConfig,
    job_manager: ClusterJobManager,
    logger: logging.Logger,
) -> bool:
    """Execute a design task on a cluster."""
    logger.info(f"Submitting cluster task: {task.task_id}")
    task.status = TaskStatus.RUNNING
    task.start_time = datetime.now()

    try:
        script_path = generate_job_script(task, config)
        job_id = job_manager.submit_job(task, script_path, config)
        task.job_id = job_id
        logger.info(f"Task {task.task_id} submitted as job {job_id}")
        return True

    except Exception as e:
        task.error_message = str(e)
        task.status = TaskStatus.FAILED
        task.end_time = datetime.now()
        logger.exception(f"Failed to submit task {task.task_id}")
        return False


def check_cluster_tasks(
    tasks: list[DesignTask],
    job_manager: ClusterJobManager,
    logger: logging.Logger,
) -> tuple[list[DesignTask], list[DesignTask]]:
    """Check status of cluster tasks and return completed/failed lists."""
    completed = []
    failed = []

    for task in tasks:
        if task.status != TaskStatus.RUNNING or not task.job_id:
            continue

        status = job_manager.check_job_status(task.job_id)

        if status == TaskStatus.COMPLETED:
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            completed.append(task)
            logger.info(f"Task {task.task_id} (job {task.job_id}) completed")

        elif status == TaskStatus.FAILED or status == TaskStatus.CANCELLED:
            task.status = TaskStatus.FAILED
            task.error_message = f"Job {task.job_id} {status.value}"
            task.end_time = datetime.now()
            failed.append(task)
            logger.warning(f"Task {task.task_id} (job {task.job_id}) failed")

    return completed, failed


def aggregate_results(tasks: list[DesignTask], output_root: Path) -> BatchResult:
    """Aggregate results from all tasks."""
    result = BatchResult(
        batch_id=tasks[0].task_id.split("_")[0] if tasks else "unknown",
        start_time=min((t.start_time for t in tasks if t.start_time), default=datetime.now()),
        end_time=datetime.now(),
        total_tasks=len(tasks),
    )

    for task in tasks:
        if task.status == TaskStatus.COMPLETED:
            result.completed += 1
        elif task.status == TaskStatus.FAILED:
            result.failed += 1
        elif task.status == TaskStatus.CANCELLED:
            result.cancelled += 1
        elif task.status == TaskStatus.PENDING:
            result.pending += 1
        elif task.status == TaskStatus.RUNNING:
            result.running += 1

    result.tasks = tasks
    if result.end_time:
        result.total_duration = result.end_time - result.start_time

    summary_path = output_root / "batch_summary.json"
    summary_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")

    report_path = output_root / "batch_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("BATCH DESIGN PIPELINE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Batch ID: {result.batch_id}\n")
        f.write(f"Start Time: {result.start_time}\n")
        f.write(f"End Time: {result.end_time}\n")
        f.write(f"Duration: {result.total_duration}\n\n")
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total Tasks:   {result.total_tasks}\n")
        f.write(f"Completed:     {result.completed}\n")
        f.write(f"Failed:        {result.failed}\n")
        f.write(f"Cancelled:     {result.cancelled}\n")
        f.write(f"Pending:       {result.pending}\n")
        f.write(f"Running:       {result.running}\n")
        f.write(f"Success Rate:  {result.to_dict()['success_rate']:.1%}\n\n")
        f.write("FAILED TASKS\n")
        f.write("-" * 40 + "\n")
        for task in tasks:
            if task.status == TaskStatus.FAILED:
                f.write(f"  {task.task_id}: {task.error_message}\n")

    return result


def run_local_batch(
    tasks: list[DesignTask],
    config: BatchConfig,
    logger: logging.Logger,
) -> BatchResult:
    """Run batch processing locally with parallel execution."""
    queue = TaskQueue(tasks, config.max_parallel)
    progress = ProgressTracker(len(tasks), config.verbose)

    state_file = config.output_root / ".batch_state.json"
    if config.resume and state_file.exists():
        queue = TaskQueue.load_state(state_file, tasks)
        logger.info(f"Resumed batch from state file: {len(queue.tasks)} pending tasks")

    while True:
        pending = queue.get_pending()
        for task in pending:
            if run_local_task(task, config, logger):
                queue.mark_completed(task)
            else:
                if not queue.retry_task(task):
                    queue.mark_failed(task, task.error_message or "Unknown error")

        completed = sum(1 for t in tasks if t.status == TaskStatus.COMPLETED)
        failed = sum(1 for t in tasks if t.status == TaskStatus.FAILED)
        running = len(queue.get_running())
        progress.update(completed, failed, running)
        queue.save_state(state_file)

        if completed + failed >= len(tasks):
            break

        time.sleep(10)

    progress.finish()
    return aggregate_results(tasks, config.output_root)


def run_cluster_batch(
    tasks: list[DesignTask],
    config: BatchConfig,
    logger: logging.Logger,
) -> BatchResult:
    """Run batch processing on a cluster scheduler."""
    job_manager = ClusterJobManager(config.cluster_type)
    progress = ProgressTracker(len(tasks), config.verbose)

    state_file = config.output_root / ".batch_state.json"
    if config.resume and state_file.exists():
        pending_tasks = [t for t in tasks if t.status == TaskStatus.PENDING]
        tasks = pending_tasks
        logger.info(f"Resumed batch from state file: {len(tasks)} pending tasks")

    all_tasks = tasks[:]
    while True:
        pending = [t for t in all_tasks if t.status == TaskStatus.PENDING]
        for task in pending[: config.max_parallel]:
            run_cluster_task(task, config, job_manager, logger)

        running = [t for t in all_tasks if t.status == TaskStatus.RUNNING]
        completed, failed = check_cluster_tasks(running, job_manager, logger)

        for task in completed:
            job_manager.cancel_job(task.job_id) if task.job_id else None

        for task in failed:
            job_manager.cancel_job(task.job_id) if task.job_id else None

        completed_count = sum(1 for t in all_tasks if t.status == TaskStatus.COMPLETED)
        failed_count = sum(1 for t in all_tasks if t.status == TaskStatus.FAILED)
        running_count = len(running)
        progress.update(completed_count, failed_count, running_count)

        if completed_count + failed_count >= len(all_tasks):
            break

        time.sleep(30)

    progress.finish()
    return aggregate_results(all_tasks, config.output_root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch protein design pipeline with parallel execution and cluster support"
    )

    input_group = parser.add_argument_group("Input Options")
    input_group.add_argument("--input-csv", type=Path, help="CSV file with task specifications")
    input_group.add_argument("--input-dir", type=Path, help="Directory containing PDB files")
    input_group.add_argument("--input-pattern", default="*.pdb", help="Glob pattern for PDB files")

    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "batch_design")
    output_group.add_argument("--resume", action="store_true", help="Resume from previous run")

    execution_group = parser.add_argument_group("Execution Options")
    execution_group.add_argument("--max-parallel", type=int, default=4, help="Max parallel tasks")
    execution_group.add_argument("--cluster", choices=["local", "slurm", "lsf"], default="local")
    execution_group.add_argument("--queue", default="default", help="Cluster queue name")
    execution_group.add_argument("--account", default="", help="Cluster account")
    execution_group.add_argument("--time", default="24:00:00", help="Job time limit")
    execution_group.add_argument("--mem", default="16G", help="Memory per job")
    execution_group.add_argument("--cpus", type=int, default=8, help="CPUs per job")
    execution_group.add_argument("--no-gpu", action="store_true", help="Disable GPU usage")

    design_group = parser.add_argument_group("Design Options")
    design_group.add_argument("--analyze-only", action="store_true", help="Run analysis only, skip design")
    design_group.add_argument("--top-positions", type=int, default=12)
    design_group.add_argument("--report-positions", type=int, default=24)
    design_group.add_argument("--top-k", type=int, default=12)

    parser.add_argument("--slurm-extra", default="", help="Extra SLURM sbatch flags")
    parser.add_argument("--lsf-extra", default="", help="Extra LSF bsub flags")
    parser.add_argument("-v", "--verbose", action="store_true")

    args = parser.parse_args()

    logger = setup_logging(args.verbose)
    config = BatchConfig(
        input_csv=args.input_csv,
        input_dir=args.input_dir,
        input_pattern=args.input_pattern,
        output_root=args.output_root,
        max_parallel=args.max_parallel,
        cluster_type=ClusterType(args.cluster),
        cluster_queue=args.queue,
        cluster_account=args.account,
        cluster_time=args.time,
        cluster_mem=args.mem,
        cluster_cpus=args.cpus,
        cluster_gpu=not args.no_gpu,
        design_mode=not args.analyze_only,
        resume=args.resume,
        verbose=args.verbose,
        slurm_extra=args.slurm_extra,
        lsf_extra=args.lsf_extra,
    )

    config.output_root.mkdir(parents=True, exist_ok=True)
    batch_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    task_specs: list[dict[str, Any]] = []

    if args.input_csv:
        task_specs = parse_input_csv(args.input_csv)
        logger.info(f"Loaded {len(task_specs)} tasks from CSV")
    elif args.input_dir:
        pdb_files = discover_pdb_files(args.input_dir, args.input_pattern)
        task_specs = [{"pdb_path": p, "chain_id": "", "protein_name": ""} for p in pdb_files]
        logger.info(f"Discovered {len(task_specs)} PDB files")
    else:
        parser.error("Either --input-csv or --input-dir must be specified")

    tasks = [create_task_from_spec(spec, batch_id, config.output_root) for spec in task_specs]
    logger.info(f"Created {len(tasks)} tasks")

    if config.cluster_type == ClusterType.LOCAL:
        result = run_local_batch(tasks, config, logger)
    else:
        result = run_cluster_batch(tasks, config, logger)

    logger.info(f"Batch complete: {result.completed}/{result.total_tasks} completed, "
                f"{result.failed} failed, success rate: {result.to_dict()['success_rate']:.1%}")
    print(f"\nResults saved to: {config.output_root}")

    sys.exit(0 if result.failed == 0 else 1)


if __name__ == "__main__":
    main()
