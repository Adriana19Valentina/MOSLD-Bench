#!/usr/bin/env python3
# run_full_pipeline.py - Master script to run the complete Bengali CL pipeline

import subprocess
import sys
import os
import time
import json
from datetime import datetime

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Pipeline steps
PIPELINE_STEPS = [
    # Baseline
    ("Train Baseline", "train_baseline.py", "Training baseline model for OOD detection"),

    # Test_1
    ("Pipeline T1", "pipeline_t1.py", "OOD detection + Clustering new classes"),
    ("Train T1", "train_t1.py", "Training model on baseline + discovered"),
    ("Evaluate T1", "evaluate_t1.py", "Evaluating on test_1"),

    # Test_2
    ("Pipeline T2", "pipeline_t2.py", "OOD detection + Clustering new classes"),
    ("Train T2", "train_t2.py", "Incremental training from T1"),
    ("Evaluate T2", "evaluate_t2.py", "Evaluating on test_2"),

    # Test_3
    ("Pipeline T3", "pipeline_t3.py", "OOD detection + Clustering new classes"),
    ("Train T3", "train_t3.py", "Incremental training from T2"),
    ("Evaluate T3", "evaluate_t3.py", "Evaluating on test_3"),

    # Final Summary
    ("Final Summary", "generate_final_summary.py", "Generating final results JSON"),
]


def print_header():
    """Print pipeline header."""
    print("=" * 80)
    print("     ROMANIAN CONTINUAL LEARNING - FULL PIPELINE EXECUTION")
    print("=" * 80)
    print(f"\n📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📁 Working directory: {SCRIPT_DIR}")
    print(f"\n📋 Pipeline steps:")
    for i, (name, script, desc) in enumerate(PIPELINE_STEPS, 1):
        print(f"   {i}. {name}: {desc}")
    print("=" * 80)


def run_step(step_num, name, script, description):
    """Run a single pipeline step."""
    print(f"\n{'=' * 80}")
    print(f"STEP {step_num}/{len(PIPELINE_STEPS)}: {name}")
    print(f"Description: {description}")
    print(f"Script: {script}")
    print("=" * 80)

    script_path = os.path.join(SCRIPT_DIR, script)

    if not os.path.exists(script_path):
        print(f"❌ ERROR: Script not found: {script_path}")
        return False, 0

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=SCRIPT_DIR,
            capture_output=False,  # Show output in real-time
            text=True
        )

        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✅ {name} completed successfully in {elapsed_time:.1f}s")
            return True, elapsed_time
        else:
            print(f"\n❌ {name} failed with return code {result.returncode}")
            return False, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"\n❌ ERROR running {name}: {e}")
        return False, elapsed_time


def run_pipeline(start_step=1, end_step=None):
    """Run the complete pipeline or a subset of steps."""

    print_header()

    if end_step is None:
        end_step = len(PIPELINE_STEPS)

    # Validate step numbers
    if start_step < 1 or start_step > len(PIPELINE_STEPS):
        print(f"❌ Invalid start_step: {start_step}")
        return False

    if end_step < start_step or end_step > len(PIPELINE_STEPS):
        print(f"❌ Invalid end_step: {end_step}")
        return False

    print(f"\n🚀 Running steps {start_step} to {end_step}")

    results = []
    total_time = 0
    all_success = True

    for i, (name, script, desc) in enumerate(PIPELINE_STEPS, 1):
        if i < start_step:
            print(f"\n⏭️  Skipping step {i}: {name}")
            continue
        if i > end_step:
            print(f"\n⏹️  Stopping at step {i}")
            break

        success, elapsed = run_step(i, name, script, desc)
        results.append({
            'step': i,
            'name': name,
            'script': script,
            'success': success,
            'time': elapsed
        })
        total_time += elapsed

        if not success:
            all_success = False
            print(f"\n⚠️  Pipeline stopped due to failure at step {i}")
            break

    # Print summary
    print(f"\n{'=' * 80}")
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)

    print(f"\n📊 Results:")
    for r in results:
        status = "✅" if r['success'] else "❌"
        print(f"   {status} Step {r['step']}: {r['name']} ({r['time']:.1f}s)")

    print(f"\n⏱️  Total time: {total_time:.1f}s ({total_time / 60:.1f} minutes)")
    print(f"📅 Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if all_success:
        print(f"\n🎉 ALL STEPS COMPLETED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  PIPELINE FAILED - Check errors above")

    print("=" * 80)

    # Save execution log
    log_file = os.path.join(SCRIPT_DIR, 'russian_cl_outputs', 'pipeline_execution_log.json')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    execution_log = {
        'timestamp': datetime.now().isoformat(),
        'total_time_seconds': total_time,
        'all_success': all_success,
        'results': results
    }

    with open(log_file, 'w') as f:
        json.dump(execution_log, f, indent=2)

    print(f"\n📝 Execution log saved to: {log_file}")

    return all_success


def run_test_step(step_name):
    """Run a specific test step by name."""
    step_name_lower = step_name.lower()

    # Find matching step
    for i, (name, script, desc) in enumerate(PIPELINE_STEPS, 1):
        if step_name_lower in name.lower() or step_name_lower in script.lower():
            print(f"Found matching step: {name}")
            return run_pipeline(start_step=i, end_step=i)

    print(f"❌ No matching step found for: {step_name}")
    print("Available steps:")
    for i, (name, script, desc) in enumerate(PIPELINE_STEPS, 1):
        print(f"   {i}. {name} ({script})")
    return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Bengali Continual Learning Pipeline")
    parser.add_argument('--start', type=int, default=1, help='Start from step N (1-9)')
    parser.add_argument('--end', type=int, default=None, help='End at step N (1-9)')
    parser.add_argument('--step', type=str, default=None, help='Run specific step by name (e.g., "t1", "train_t2")')
    parser.add_argument('--t1', action='store_true', help='Run only Test_1 (steps 1-3)')
    parser.add_argument('--t2', action='store_true', help='Run only Test_2 (steps 4-6)')
    parser.add_argument('--t3', action='store_true', help='Run only Test_3 (steps 7-9)')
    parser.add_argument('--list', action='store_true', help='List all pipeline steps')

    args = parser.parse_args()

    if args.list:
        print("Pipeline steps:")
        for i, (name, script, desc) in enumerate(PIPELINE_STEPS, 1):
            print(f"   {i}. {name}: {script}")
            print(f"      {desc}")
        sys.exit(0)

    if args.step:
        success = run_test_step(args.step)
    elif args.t1:
        success = run_pipeline(start_step=1, end_step=3)
    elif args.t2:
        success = run_pipeline(start_step=4, end_step=6)
    elif args.t3:
        success = run_pipeline(start_step=7, end_step=9)
    else:
        success = run_pipeline(start_step=args.start, end_step=args.end)

    sys.exit(0 if success else 1)