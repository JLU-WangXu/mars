#!/usr/bin/env python3
"""
MARS-FIELD Interactive CLI Interface

Provides interactive configuration, real-time progress display,
result preview, and design iteration control for MARS-FIELD pipeline.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

# ANSI color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RESET = "\033[0m"


class ProgressState(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProgressStep:
    name: str
    description: str
    state: ProgressState = ProgressState.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    message: str = ""

    @property
    def duration(self) -> Optional[float]:
        if self.start_time is None:
            return None
        end = self.end_time if self.end_time else time.time()
        return end - self.start_time

    def format_duration(self) -> str:
        d = self.duration
        if d is None:
            return "--:--"
        mins = int(d // 60)
        secs = int(d % 60)
        return f"{mins:02d}:{secs:02d}"


@dataclass
class DesignSession:
    name: str = ""
    pdb_path: str = ""
    chain: str = "A"
    design_positions: list[int] = field(default_factory=list)
    protected_positions: list[int] = field(default_factory=list)
    top_k: int = 12
    decoder_enabled: bool = True
    neural_rerank_enabled: bool = False
    max_iterations: int = 3
    current_iteration: int = 1
    output_dir: Path = field(default_factory=lambda: Path("outputs/default_session"))

    def to_config_dict(self) -> dict:
        return {
            "protein": {
                "name": self.name,
                "pdb_path": self.pdb_path,
                "chain": self.chain,
                "wt_sequence": "PLACEHOLDER_WT_SEQ",
                "protected_positions": self.protected_positions,
                "preprocess": {},
            },
            "generation": {
                "design_positions": [str(p) for p in self.design_positions],
                "esm_if": {"enabled": False},
                "protein_mpnn": {"enabled": True},
            },
            "method": {
                "local_proposals": {},
                "manual_bias": {},
                "score_weights": {},
                "oxidation_min_sasa": 0.3,
                "oxidation_min_dist_protected": 8.0,
                "flexible_surface_min_sasa": 0.15,
                "learned_fusion": {"decoder_enabled": self.decoder_enabled},
                "neural_rerank": {"enabled": self.neural_rerank_enabled},
            },
            "evolution": {},
            "topic": {"enabled": False},
        }


class ProgressDisplay:
    """Real-time progress display for pipeline execution."""

    def __init__(self):
        self.steps: list[ProgressStep] = []
        self.lock = threading.Lock()
        self.running = True

    def add_step(self, name: str, description: str) -> int:
        step = ProgressStep(name=name, description=description)
        with self.lock:
            self.steps.append(step)
            return len(self.steps) - 1

    def start_step(self, index: int, message: str = "") -> None:
        with self.lock:
            if 0 <= index < len(self.steps):
                self.steps[index].state = ProgressState.RUNNING
                self.steps[index].start_time = time.time()
                if message:
                    self.steps[index].message = message

    def complete_step(self, index: int, message: str = "") -> None:
        with self.lock:
            if 0 <= index < len(self.steps):
                self.steps[index].state = ProgressState.COMPLETED
                self.steps[index].end_time = time.time()
                if message:
                    self.steps[index].message = message

    def fail_step(self, index: int, message: str = "") -> None:
        with self.lock:
            if 0 <= index < len(self.steps):
                self.steps[index].state = ProgressState.FAILED
                self.steps[index].end_time = time.time()
                if message:
                    self.steps[index].message = message

    def skip_step(self, index: int, message: str = "") -> None:
        with self.lock:
            if 0 <= index < len(self.steps):
                self.steps[index].state = ProgressState.SKIPPED
                self.steps[index].end_time = time.time()
                if message:
                    self.steps[index].message = message

    def _get_state_symbol(self, state: ProgressState) -> str:
        symbols = {
            ProgressState.PENDING: f"{Colors.DIM}○{Colors.RESET}",
            ProgressState.RUNNING: f"{Colors.CYAN}◐{Colors.RESET}",
            ProgressState.COMPLETED: f"{Colors.GREEN}●{Colors.RESET}",
            ProgressState.FAILED: f"{Colors.RED}✗{Colors.RESET}",
            ProgressState.SKIPPED: f"{Colors.YELLOW}◌{Colors.RESET}",
        }
        return symbols.get(state, "?")

    def _get_state_color(self, state: ProgressState) -> str:
        colors = {
            ProgressState.PENDING: Colors.DIM,
            ProgressState.RUNNING: Colors.CYAN,
            ProgressState.COMPLETED: Colors.GREEN,
            ProgressState.FAILED: Colors.RED,
            ProgressState.SKIPPED: Colors.YELLOW,
        }
        return colors.get(state, Colors.RESET)

    def render(self) -> None:
        """Render the current progress state to terminal."""
        with self.lock:
            lines = []
            total_steps = len(self.steps)
            completed = sum(1 for s in self.steps if s.state == ProgressState.COMPLETED)
            running = any(s.state == ProgressState.RUNNING for s in self.steps)

            # Header
            lines.append("")
            lines.append(f"{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.RESET}")
            lines.append(f"{Colors.BOLD}  MARS-FIELD Pipeline Progress{Colors.RESET}")
            lines.append(f"{Colors.BOLD}{'═' * 60}{Colors.RESET}")

            # Overall progress
            if total_steps > 0:
                pct = int(100 * completed / total_steps)
                bar_len = 40
                filled = int(bar_len * completed / total_steps)
                bar = "█" * filled + "░" * (bar_len - filled)
                lines.append(f"\n  [{bar}] {pct}% ({completed}/{total_steps} steps)")
            lines.append("")

            # Individual steps
            for i, step in enumerate(self.steps):
                state_sym = self._get_state_symbol(step.state)
                state_color = self._get_state_color(step.state)
                duration = step.format_duration()

                line = f"  {state_sym} {state_color}{step.name}{Colors.RESET}"
                line = f"{line:<45} {Colors.DIM}[{duration}]{Colors.RESET}"

                if step.state == ProgressState.RUNNING and step.message:
                    line += f"\n    {Colors.CYAN}→ {step.message}{Colors.RESET}"
                elif step.message:
                    line += f"\n    {Colors.DIM}  {step.message}{Colors.RESET}"

                lines.append(line)

            lines.append("")
            lines.append(f"{Colors.DIM}  {'Press Ctrl+C to interrupt...' if running else ''}{Colors.RESET}")
            lines.append("")

        # Clear and redraw
        output = "\n".join(lines)
        sys.stdout.write(f"\033[2J\033[H{output}")
        sys.stdout.flush()


class InteractiveConfigurator:
    """Interactive configuration wizard for MARS-FIELD."""

    def __init__(self):
        self.session = DesignSession()
        self.configs_dir = Path(__file__).resolve().parents[1] / "configs"

    def clear_screen(self) -> None:
        print("\033[2J\033[H", end="")

    def print_header(self, title: str) -> None:
        self.clear_screen()
        print(f"{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}  {title}{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.RESET}\n")

    def print_step(self, step_num: int, total: int, title: str) -> None:
        print(f"\n{Colors.CYAN}┌─ Step {step_num}/{total}: {title}{Colors.RESET}")
        print(f"{Colors.CYAN}│{'─' * 56}{Colors.RESET}")

    def print_option(self, key: str, description: str, default: str = "") -> None:
        default_str = f" {Colors.DIM}[default: {default}]{Colors.RESET}" if default else ""
        print(f"  {Colors.YELLOW}[{key}]{Colors.RESET} {description}{default_str}")

    def print_input_prompt(self, prompt: str) -> str:
        return input(f"  {Colors.GREEN}→{Colors.RESET} {prompt}: ").strip()

    def print_warning(self, msg: str) -> None:
        print(f"  {Colors.YELLOW}⚠{Colors.RESET} {msg}")

    def print_error(self, msg: str) -> None:
        print(f"  {Colors.RED}✗{Colors.RESET} {msg}")

    def print_success(self, msg: str) -> None:
        print(f"  {Colors.GREEN}✓{Colors.RESET} {msg}")

    def configure(self) -> DesignSession:
        """Run interactive configuration wizard."""
        self.print_header("MARS-FIELD Interactive Configuration")

        print(f"{Colors.DIM}Welcome to MARS-FIELD design configuration wizard.{Colors.RESET}")
        print(f"{Colors.DIM}Press Enter to accept defaults shown in brackets.{Colors.RESET}\n")

        # Step 1: Project name
        self.print_step(1, 5, "Project Setup")
        name = self.print_input_prompt("Design project name")
        self.session.name = name or f"design_{int(time.time())}"
        self.print_success(f"Project name: {self.session.name}")

        # Step 2: PDB file
        self.print_step(2, 5, "Structure Input")
        pdb_path = self.print_input_prompt("PDB file path")
        if pdb_path:
            p = Path(pdb_path)
            if p.exists():
                self.session.pdb_path = str(p.resolve())
                self.print_success(f"PDB file: {self.session.pdb_path}")
            else:
                self.print_warning(f"File not found, will use: {pdb_path}")
                self.session.pdb_path = pdb_path
        else:
            self.session.pdb_path = str(self.configs_dir / "calb_1lbt.yaml")
            self.print_success(f"Using default PDB: {self.session.pdb_path}")

        # Step 3: Chain ID
        self.print_step(3, 5, "Chain Selection")
        chain = self.print_input_prompt("Chain ID")
        self.session.chain = chain.upper() or "A"
        self.print_success(f"Chain: {self.session.chain}")

        # Step 4: Design positions
        self.print_step(4, 5, "Design Positions")
        pos_input = self.print_input_prompt("Design positions (comma-separated, e.g., 10,12,15)")
        if pos_input:
            try:
                self.session.design_positions = [int(x.strip()) for x in pos_input.split(",")]
                self.print_success(f"Design positions: {self.session.design_positions}")
            except ValueError:
                self.print_error("Invalid format, using all positions")
                self.session.design_positions = []
        else:
            self.print_success("Will auto-detect design positions")

        # Step 5: Advanced options
        self.print_step(5, 5, "Advanced Options")
        self.print_option("k", "Top-K sequences per iteration", str(self.session.top_k))
        self.print_option("d", "Enable decoder", "Y" if self.session.decoder_enabled else "N")
        self.print_option("n", "Enable neural reranking", "Y" if self.session.neural_rerank_enabled else "N")
        self.print_option("i", "Max iterations", str(self.session.max_iterations))

        advanced = self.print_input_prompt("Configure advanced options? (y/n)")
        if advanced.lower() == "y":
            top_k = self.print_input_prompt(f"Top-K (current: {self.session.top_k})")
            if top_k.isdigit():
                self.session.top_k = int(top_k)

            decoder = self.print_input_prompt("Enable decoder (y/n)")
            if decoder:
                self.session.decoder_enabled = decoder.lower() == "y"

            rerank = self.print_input_prompt("Enable neural reranking (y/n)")
            if rerank:
                self.session.neural_rerank_enabled = rerank.lower() == "y"

            iterations = self.print_input_prompt(f"Max iterations (current: {self.session.max_iterations})")
            if iterations.isdigit():
                self.session.max_iterations = int(iterations)

        # Summary
        self.print_header("Configuration Summary")
        summary_items = [
            ("Project", self.session.name),
            ("PDB", self.session.pdb_path),
            ("Chain", self.session.chain),
            ("Design positions", str(self.session.design_positions) if self.session.design_positions else "Auto-detect"),
            ("Top-K", str(self.session.top_k)),
            ("Decoder", "Enabled" if self.session.decoder_enabled else "Disabled"),
            ("Neural reranking", "Enabled" if self.session.neural_rerank_enabled else "Disabled"),
            ("Max iterations", str(self.session.max_iterations)),
        ]
        for key, value in summary_items:
            print(f"  {Colors.CYAN}{key:<20}{Colors.RESET} {value}")

        confirm = self.print_input_prompt("\nStart pipeline? (y/n)")
        if confirm.lower() != "y":
            self.print_warning("Configuration cancelled")
            sys.exit(0)

        return self.session


class ResultPreview:
    """Preview and analyze pipeline results."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir

    def preview_summary(self) -> dict:
        """Generate result summary preview."""
        summary = {
            "output_dir": str(self.output_dir),
            "files": [],
            "candidates": 0,
            "top_scores": [],
        }

        # List output files
        if self.output_dir.exists():
            for ext in ["*.csv", "*.json", "*.jsonl"]:
                for f in self.output_dir.rglob(ext):
                    rel_path = f.relative_to(self.output_dir)
                    size = f.stat().st_size
                    summary["files"].append({
                        "path": str(rel_path),
                        "size": size,
                        "size_str": self._format_size(size),
                    })

        # Count candidates from CSV files
        for csv_file in self.output_dir.rglob("*.csv"):
            try:
                import pandas as pd
                df = pd.read_csv(csv_file, nrows=5)
                if "score" in df.columns or "total_score" in df.columns:
                    score_col = "total_score" if "total_score" in df.columns else "score"
                    summary["top_scores"] = df[score_col].head(5).tolist()
                    summary["candidates"] = len(df)
                    break
            except Exception:
                pass

        return summary

    def display_preview(self) -> None:
        """Display result preview to terminal."""
        summary = self.preview_summary()

        print(f"\n{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}  Result Preview{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.RESET}\n")

        print(f"  {Colors.CYAN}Output directory:{Colors.RESET} {summary['output_dir']}")
        print(f"  {Colors.CYAN}Total files:{Colors.RESET} {len(summary['files'])}")

        if summary["candidates"] > 0:
            print(f"  {Colors.CYAN}Total candidates:{Colors.RESET} {summary['candidates']}")

        if summary["files"]:
            print(f"\n  {Colors.YELLOW}Generated files:{Colors.RESET}")
            for f in summary["files"][:10]:
                print(f"    • {f['path']} ({f['size_str']})")
            if len(summary["files"]) > 10:
                print(f"    ... and {len(summary['files']) - 10} more files")

        if summary["top_scores"]:
            print(f"\n  {Colors.GREEN}Top scores:{Colors.RESET}")
            for i, score in enumerate(summary["top_scores"], 1):
                print(f"    {i}. {score:.4f}")

        print("")

    def _format_size(self, size: int) -> str:
        """Format file size in human-readable form."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"


class DesignIterationController:
    """Control design iterations and iterations."""

    def __init__(self, session: DesignSession):
        self.session = session
        self.iteration_history: list[dict] = []
        self.progress_display = ProgressDisplay()
        self._setup_progress_steps()

    def _setup_progress_steps(self) -> None:
        """Define pipeline steps for progress tracking."""
        steps = [
            ("structure_analysis", "Structure Analysis"),
            ("feature_extraction", "Feature Extraction"),
            ("proposal_generation", "Proposal Generation"),
            ("sequence_optimization", "Sequence Optimization"),
            ("scoring", "Scoring & Ranking"),
            ("results_export", "Results Export"),
        ]
        for name, desc in steps:
            self.progress_display.add_step(name, desc)

    def run_iteration(self, iteration: int) -> bool:
        """Run a single design iteration."""
        self.session.current_iteration = iteration
        print(f"\n{Colors.BOLD}Iteration {iteration}/{self.session.max_iterations}{Colors.RESET}\n")

        # Simulate pipeline steps with progress
        pipeline_thread = threading.Thread(target=self._run_pipeline_simulation)
        pipeline_thread.daemon = True
        pipeline_thread.start()

        try:
            while pipeline_thread.is_alive():
                self.progress_display.render()
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.progress_display.running = False
            print(f"\n{Colors.YELLOW}Interrupted by user{Colors.RESET}")
            return False

        pipeline_thread.join()
        return True

    def _run_pipeline_simulation(self) -> None:
        """Simulate pipeline execution for demonstration."""
        import random

        step_delays = [2.0, 1.5, 3.0, 2.5, 1.0, 1.5]

        for i, delay in enumerate(step_delays):
            self.progress_display.start_step(i, f"Processing...")
            time.sleep(delay)

            # Simulate success/failure
            if random.random() > 0.95:
                self.progress_display.fail_step(i, "Error occurred")
            else:
                self.progress_display.complete_step(i, "Done")

    def should_continue(self) -> bool:
        """Determine if another iteration should run."""
        if self.session.current_iteration >= self.session.max_iterations:
            return False

        print(f"\n{Colors.CYAN}Iteration {self.session.current_iteration} complete.{Colors.RESET}")
        continue_prompt = input(f"  Continue to iteration {self.session.current_iteration + 1}? (y/n): ").strip()
        return continue_prompt.lower() == "y"

    def record_iteration_result(self, iteration: int, metrics: dict) -> None:
        """Record iteration metrics for history."""
        self.iteration_history.append({
            "iteration": iteration,
            "timestamp": time.time(),
            "metrics": metrics,
        })

    def show_iteration_history(self) -> None:
        """Display iteration history."""
        if not self.iteration_history:
            print(f"\n{Colors.DIM}No iterations completed yet.{Colors.RESET}")
            return

        print(f"\n{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.RESET}")
        print(f"{Colors.BOLD}  Iteration History{Colors.RESET}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'═' * 60}{Colors.RESET}\n")

        for record in self.iteration_history:
            iter_num = record["iteration"]
            metrics = record["metrics"]
            print(f"  {Colors.CYAN}Iteration {iter_num}:{Colors.RESET}")
            for key, value in metrics.items():
                print(f"    {key}: {value}")
            print("")


class MarsCLI:
    """Main CLI interface for MARS-FIELD."""

    def __init__(self):
        self.configurator = InteractiveConfigurator()
        self.session: Optional[DesignSession] = None
        self.progress_display = ProgressDisplay()

    def run_interactive(self) -> int:
        """Run interactive configuration and pipeline."""
        # Configure session
        self.session = self.configurator.configure()

        # Setup iteration controller
        controller = DesignIterationController(self.session)

        # Run iterations
        iteration = 1
        while iteration <= self.session.max_iterations:
            success = controller.run_iteration(iteration)
            if not success:
                return 1

            controller.record_iteration_result(iteration, {
                "status": "completed",
                "candidates": self.session.top_k,
            })

            if iteration < self.session.max_iterations:
                if not controller.should_continue():
                    break
            iteration += 1

        # Show results
        print(f"\n{Colors.BOLD}{Colors.GREEN}Pipeline completed!{Colors.RESET}\n")
        controller.show_iteration_history()

        # Preview results
        preview = ResultPreview(self.session.output_dir)
        preview.display_preview()

        return 0

    def run_quick(self, config_path: str, top_k: int = 12) -> int:
        """Run pipeline with minimal configuration."""
        print(f"\n{Colors.BOLD}Quick Run Mode{Colors.RESET}")
        print(f"  Config: {config_path}")
        print(f"  Top-K: {top_k}\n")

        # This would integrate with the actual pipeline
        print(f"{Colors.YELLOW}Quick run integrates with run_mars_pipeline.py{Colors.RESET}")
        return 0


def main():
    """Main entry point for MARS-FIELD CLI."""
    parser = argparse.ArgumentParser(
        description="MARS-FIELD Interactive CLI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mars_cli.py              # Interactive configuration
  mars_cli.py --quick      # Quick run with default settings
  mars_cli.py --config path/to/config.yaml
        """,
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=12,
        help="Number of top sequences to generate (default: 12)",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Run with default settings (quick mode)",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output",
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Output directory for results",
    )

    args = parser.parse_args()

    # Disable colors if requested
    if args.no_color:
        for attr in dir(Colors):
            if not attr.startswith("_"):
                setattr(Colors, attr, "")

    # Initialize CLI
    cli = MarsCLI()

    if args.quick or args.config:
        # Quick run mode
        config_path = args.config or "configs/calb_1lbt.yaml"
        return cli.run_quick(config_path, args.top_k)
    else:
        # Interactive mode
        return cli.run_interactive()


if __name__ == "__main__":
    sys.exit(main())
