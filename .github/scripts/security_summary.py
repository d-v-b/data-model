"""Build a markdown security report from pip-audit JSON output.

Reads:
  pip-audit-report.json — produced by: pip-audit -f json -o pip-audit-report.json

Writes:
  security-summary.md          — used by the Post PR comment step
  $GITHUB_STEP_SUMMARY (env)   — appended for the Actions job summary tab

Note: bandit results are reported via the GitHub Security tab (SARIF), not here.

Environment variables:
  GITHUB_STEP_SUMMARY — path to the step summary file (set by Actions runner)
"""

from __future__ import annotations

import json
import os
from pathlib import Path

MARKER = "<!-- security-scan-results -->"


def _pip_audit_section() -> list[str]:
    lines: list[str] = ["\n### pip-audit — Dependency Vulnerability Scan\n"]
    try:
        deps: list[dict] = json.loads(Path("pip-audit-report.json").read_text()).get(
            "dependencies", []
        )
    except Exception as exc:
        lines.append(f":warning: Could not read pip-audit report: {exc}")
        return lines

    vulnerable = [d for d in deps if d.get("vulns")]
    if not vulnerable:
        lines.append(":white_check_mark: No vulnerabilities found")
        return lines

    fixable: list[tuple[str, str, dict]] = []
    unfixable: list[tuple[str, str, dict]] = []
    for dep in vulnerable:
        for vuln in dep["vulns"]:
            entry = (dep["name"], dep["version"], vuln)
            (fixable if vuln.get("fix_versions") else unfixable).append(entry)

    if fixable:
        lines += [
            f"**:x: {len(fixable)} fixable vulnerability/ies found**\n",
            "| Package | Version | ID | Fix versions | Description |",
            "|---------|---------|-----|-------------|-------------|",
        ]
        for name, version, vuln in fixable:
            fixes = ", ".join(vuln["fix_versions"])
            desc = vuln.get("description", "")[:120]
            lines.append(f"| {name} | {version} | {vuln['id']} | {fixes} | {desc} |")

    if unfixable:
        items = ", ".join(f"{name}@{ver} ({v['id']})" for name, ver, v in unfixable)
        lines.append(
            f"\n> :warning: {len(unfixable)} vulnerability/ies have no fix available"
            f" yet (non-blocking): {items}"
        )

    return lines


def build_markdown() -> str:
    lines = ["## Security Scan Results\n"]
    lines += _pip_audit_section()
    return "\n".join(lines) + "\n"


def main() -> None:
    step_summary_path = os.environ.get("GITHUB_STEP_SUMMARY", "")
    body = MARKER + "\n" + build_markdown()
    Path("security-summary.md").write_text(body)
    if step_summary_path:
        with open(step_summary_path, "a") as f:
            f.write(body)


if __name__ == "__main__":
    main()
