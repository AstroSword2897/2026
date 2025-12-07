#!/usr/bin/env python3
"""
Test Runner with Status Reporting (GREEN/ORANGE/RED)
Executes tests and generates status reports aligned with daily checkpoint system.
"""

import subprocess
import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def determine_status(
    passed: int,
    failed: int,
    skipped: int,
    total: int,
    performance_ok: bool = True,
    coverage_ok: bool = True
) -> str:
    """
    Determine GREEN/ORANGE/RED status based on test results.
    
    GREEN: All tests pass, performance within targets, coverage >80%
    ORANGE: Most tests pass (>80%), 1 critical issue, or moderate performance/coverage issues
    RED: Multiple failures (>20%), critical functionality broken, or major gaps
    """
    if total == 0:
        return "RED"
    
    pass_rate = passed / total if total > 0 else 0.0
    
    # RED conditions
    if failed > total * 0.2:  # >20% failure rate
        return "RED"
    if not performance_ok and failed > 0:
        return "RED"
    if not coverage_ok and pass_rate < 0.6:
        return "RED"
    
    # ORANGE conditions
    if failed > 0:  # Any failures
        return "ORANGE"
    if pass_rate < 0.95:  # <95% pass rate
        return "ORANGE"
    if not performance_ok:
        return "ORANGE"
    if not coverage_ok:
        return "ORANGE"
    
    # GREEN - all good
    return "GREEN"


def run_pytest(test_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Run pytest on a specific test file or directory."""
    cmd = ["pytest", test_path, "-v", "--tb=short"]
    if not verbose:
        cmd = ["pytest", test_path, "-q"]
    
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=Path(__file__).parent.parent
    )
    elapsed = time.time() - start_time
    
    # Parse output
    output_lines = result.stdout.split('\n')
    
    # Extract test counts from pytest output
    passed = 0
    failed = 0
    skipped = 0
    
    # Look for summary line like "27 passed, 16 warnings in 29.98s"
    for line in output_lines:
        line_lower = line.lower()
        if "passed" in line_lower or "failed" in line_lower:
            # Try to extract numbers from summary line
            import re
            # Match pattern like "27 passed" or "2 failed"
            passed_match = re.search(r'(\d+)\s+passed', line_lower)
            failed_match = re.search(r'(\d+)\s+failed', line_lower)
            skipped_match = re.search(r'(\d+)\s+skipped', line_lower)
            
            if passed_match:
                passed = int(passed_match.group(1))
            if failed_match:
                failed = int(failed_match.group(1))
            if skipped_match:
                skipped = int(skipped_match.group(1))
            
            # If we found a summary line, break
            if passed_match or failed_match:
                break
    
    total = passed + failed + skipped
    
    return {
        'passed': passed,
        'failed': failed,
        'skipped': skipped,
        'total': total,
        'elapsed': elapsed,
        'returncode': result.returncode,
        'stdout': result.stdout,
        'stderr': result.stderr
    }


def run_all_tests() -> Dict[str, Any]:
    """Run all tests and collect results."""
    test_files = [
        'tests/test_model.py',
        'tests/test_comprehensive_system.py',
        'tests/test_condition_specific.py',
        'tests/test_export_validation.py',
    ]
    
    results = {}
    
    for test_file in test_files:
        print(f"\n{'='*60}")
        print(f"Running: {test_file}")
        print(f"{'='*60}")
        
        results[test_file] = run_pytest(test_file, verbose=True)
        
        status = determine_status(
            results[test_file]['passed'],
            results[test_file]['failed'],
            results[test_file]['skipped'],
            results[test_file]['total']
        )
        results[test_file]['status'] = status
        
        print(f"\nStatus: {status}")
        print(f"Passed: {results[test_file]['passed']}, Failed: {results[test_file]['failed']}, Skipped: {results[test_file]['skipped']}")
    
    # Run full suite
    print(f"\n{'='*60}")
    print("Running Full Test Suite")
    print(f"{'='*60}")
    
    full_results = run_pytest('tests/', verbose=True)
    full_status = determine_status(
        full_results['passed'],
        full_results['failed'],
        full_results['skipped'],
        full_results['total']
    )
    full_results['status'] = full_status
    results['full_suite'] = full_results
    
    return results


def generate_status_report(results: Dict[str, Any]) -> str:
    """Generate markdown status report."""
    report = []
    report.append("# Test Execution Status Report")
    report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("\n## Summary\n")
    
    # Overall summary
    full = results.get('full_suite', {})
    overall_status = full.get('status', 'UNKNOWN')
    
    report.append(f"### Overall Status: {overall_status}")
    report.append(f"- **Total Tests:** {full.get('total', 0)}")
    report.append(f"- **Passed:** {full.get('passed', 0)}")
    report.append(f"- **Failed:** {full.get('failed', 0)}")
    report.append(f"- **Skipped:** {full.get('skipped', 0)}")
    report.append(f"- **Execution Time:** {full.get('elapsed', 0):.2f}s")
    
    report.append("\n## Test Suite Results\n")
    
    for test_file, result in results.items():
        if test_file == 'full_suite':
            continue
        
        status = result.get('status', 'UNKNOWN')
        report.append(f"### {test_file}")
        report.append(f"**Status:** {status}")
        report.append(f"- Passed: {result.get('passed', 0)}")
        report.append(f"- Failed: {result.get('failed', 0)}")
        report.append(f"- Skipped: {result.get('skipped', 0)}")
        report.append(f"- Time: {result.get('elapsed', 0):.2f}s")
        report.append("")
    
    return "\n".join(report)


if __name__ == "__main__":
    print("Running Test Suite with Status Reporting")
    print("=" * 60)
    
    results = run_all_tests()
    
    report = generate_status_report(results)
    
    # Save report
    report_path = Path(__file__).parent.parent / "docs" / "TEST_EXECUTION_REPORT.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    
    print("\n" + "=" * 60)
    print("Test Execution Complete")
    print(f"Report saved to: {report_path}")
    print("=" * 60)
    
    # Print summary
    print("\n" + report)
    
    # Exit with appropriate code
    full = results.get('full_suite', {})
    if full.get('failed', 0) > 0:
        sys.exit(1)
    sys.exit(0)

