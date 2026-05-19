import os
import sys
import time
import argparse
import subprocess
from datetime import datetime


def run_script(script_name, description):
    print(f"\n{'=' * 70}")
    print(f" {description}")
    print(f"   Script: {script_name}")
    print('=' * 70)

    start = time.time()
    result = subprocess.run([sys.executable, script_name])
    elapsed = time.time() - start

    if result.returncode == 0:
        print(f"\n {description} completed in {elapsed:.1f}s")
        return True, elapsed
    else:
        print(f"\n {description} FAILED after {elapsed:.1f}s")
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(description='Run Contrastive Learning Pipeline')
    parser.add_argument('--step', type=int, choices=[1, 2, 3],
                        help='Run only specific step (1=T1, 2=T2, 3=T3)')
    args = parser.parse_args()

    print("=" * 70)
    print("CONTRASTIVE LEARNING PIPELINE")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    steps = [
        (1, 'train_t1_contrastive.py', 'T1 Training (CE + Contrastive)'),
        (2, 'train_t2_contrastive.py', 'T2 Training (CE + Contrastive)'),
        (3, 'train_t3_contrastive.py', 'T3 Training (CE + Contrastive)'),
    ]

    if args.step:
        steps = [(n, s, d) for n, s, d in steps if n == args.step]

    results = []
    total_time = 0

    for step_num, script, desc in steps:
        success, elapsed = run_script(script, desc)
        results.append((step_num, desc, success, elapsed))
        total_time += elapsed

        if not success:
            print(f"\n Pipeline stopped at step {step_num}")
            break

    print(f"\n{'=' * 70}")
    print("PIPELINE SUMMARY")
    print('=' * 70)

    for num, desc, success, elapsed in results:
        status = "ok" if success else "not ok"
        print(f"  {status} Step {num}: {desc} ({elapsed:.1f}s)")

    print(f"\n Total time: {total_time:.1f}s ({total_time / 60:.1f} min)")
    print(f" Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    all_success = all(r[2] for r in results)
    if all_success:
        print("\nPIPELINE COMPLETED SUCCESSFULLY!")
    else:
        print("\n PIPELINE COMPLETED WITH ERRORS")
        sys.exit(1)


if __name__ == '__main__':
    main()
