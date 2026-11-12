"""Summarize a nightly test run's JUnit reports as the body of a GitHub issue.

Usage:
    python .github/scripts/nightly_report.py <artifacts-dir> --run-url <url> --tests-result <result>

Prints the issue body (Markdown) and writes ``failed=true|false`` to ``$GITHUB_OUTPUT``.
"""

import argparse
import os
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path


def failures(artifacts: Path) -> dict[str, list[tuple[str, str]]]:
    """Test file -> [(test name, first line of the error)] for every failed or errored test."""
    by_file = defaultdict(list)
    for report in sorted(artifacts.rglob("junit-*.xml")):
        for case in ET.parse(report).getroot().iter("testcase"):
            problem = case.find("failure")
            if problem is None:
                problem = case.find("error")
            if problem is None:
                continue
            parts = (case.get("classname") or "?").split(".")
            while len(parts) > 1 and parts[-1].startswith("Test"):  # drop test class names
                parts.pop()
            test_file = "/".join(parts) + ".py"
            message = (problem.get("message") or problem.text or "").strip().splitlines()
            by_file[test_file].append((case.get("name", "?"), message[0][:200] if message else ""))
    return dict(by_file)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifacts", type=Path)
    parser.add_argument("--run-url", required=True)
    parser.add_argument("--tests-result", required=True, help="result of the Test job: success, failure, ...")
    args = parser.parse_args()

    failed = failures(args.artifacts)
    lines = [f"The [nightly test run]({args.run_url}) failed.", ""]
    if failed:
        lines += [
            f"{sum(map(len, failed.values()))} tests failed in {len(failed)} test files. "
            "A failing download test usually means the dataset's host is down or has moved.",
            "",
            "| Test file | Test | Error |",
            "|---|---|---|",
        ]
        for test_file, cases in sorted(failed.items()):
            for name, message in cases:
                message = message.replace("|", "\\|")
                lines.append(f"| `{test_file}` | `{name}` | {message} |")
    elif args.tests_result != "success":
        lines.append(
            f"No test failures were recorded, but the test jobs ended with `{args.tests_result}`. See the run log."
        )

    is_failure = bool(failed) or args.tests_result != "success"
    print("\n".join(lines))
    if output := os.environ.get("GITHUB_OUTPUT"):
        with open(output, "a") as f:
            f.write(f"failed={'true' if is_failure else 'false'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
