"""Exit with code 1 if blocking security issues are found.

Blocking conditions:
  - pip-audit: any vulnerability that has a known fix version available

Non-blocking (printed as informational, no failure):
  - pip-audit: vulnerabilities with no fix version available yet

Note: bandit HIGH severity blocking is handled by PyCQA/bandit-action directly.

Reads:
  pip-audit-report.json — produced by: pip-audit -f json -o pip-audit-report.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def check_pip_audit() -> bool:
    """Return True if fixable pip-audit vulnerabilities are found."""
    try:
        deps: list[dict] = json.loads(Path("pip-audit-report.json").read_text()).get(
            "dependencies", []
        )
    except Exception as exc:
        print(f"Could not read pip-audit report: {exc}", file=sys.stderr)
        return True

    fixable: list[tuple[str, str]] = []
    unfixable: list[tuple[str, str]] = []
    for dep in deps:
        for vuln in dep.get("vulns", []):
            entry = (dep["name"], vuln["id"])
            (fixable if vuln.get("fix_versions") else unfixable).append(entry)

    if fixable:
        print(f"pip-audit: {len(fixable)} fixable vulnerability/ies — blocking")
    if unfixable:
        print(f"pip-audit: {len(unfixable)} vulnerability/ies with no fix available — non-blocking")

    return bool(fixable)


def main() -> None:
    sys.exit(1 if check_pip_audit() else 0)


if __name__ == "__main__":
    main()
