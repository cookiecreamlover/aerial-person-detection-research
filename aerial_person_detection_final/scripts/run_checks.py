"""Run publication-repository regression tests and compile checks."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TESTS = [
    'tests/test_metrics.py',
    'tests/test_review_fixes.py',
    'tests/test_bootstrap_and_labels.py',
    'tests/test_protocol_controls.py',
    'tests/test_uav_simulation.py',
]


def main():
    for test in TESTS:
        subprocess.run([sys.executable, test], cwd=ROOT, env={**__import__('os').environ, 'PYTHONPATH': str(ROOT)}, check=True)
    subprocess.run([sys.executable, '-m', 'compileall', '-q', '.'], cwd=ROOT, check=True)
    print('PUBLICATION REPOSITORY CHECKS PASSED')


if __name__ == '__main__':
    main()
