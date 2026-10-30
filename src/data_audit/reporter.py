"""
Reporter module for saving audit logs, JSON reports, CSV offending lists, and Markdown summaries.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import pandas as pd

from .detectors import DiscrepancyResult

log = logging.getLogger("data_audit.reporter")


def _json_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    if hasattr(obj, "tolist"):
        return obj.tolist()
    return str(obj)


class AuditReporter:
    """Formats and exports audit findings to disk and logs."""

    def __init__(self, out_dir: Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.csv_dir = self.out_dir / "offending_records"
        self.csv_dir.mkdir(parents=True, exist_ok=True)

    def log_results_to_console(self, results: List[DiscrepancyResult]) -> None:
        """Emits structured log messages for each finding."""
        log.info("=" * 70)
        log.info("DATA AUDIT DISCREPANCY SUMMARY")
        log.info("=" * 70)

        for r in results:
            prefix = f"[{r.severity}]"
            msg = f"{prefix:<9} {r.target_dataset:<32} | {r.check_name:<40} | Issues: {r.discrepancy_count:>5} ({r.discrepancy_pct:>6.2f}%)"
            if r.severity == "ERROR":
                log.error(msg)
                log.error("   └─ %s", r.description)
            elif r.severity == "WARNING":
                log.warning(msg)
                log.warning("   └─ %s", r.description)
            else:
                log.info(msg)

        log.info("=" * 70)

    def export_json(self, results: List[DiscrepancyResult], filename: str = "discrepancy_report.json") -> Path:
        """Exports full discrepancy report as JSON."""
        out_path = self.out_dir / filename
        data = {
            "summary": {
                "total_checks": len(results),
                "errors": sum(1 for r in results if r.severity == "ERROR"),
                "warnings": sum(1 for r in results if r.severity == "WARNING"),
                "info": sum(1 for r in results if r.severity == "INFO"),
            },
            "findings": [r.to_dict() for r in results],
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=_json_default)
        log.info("Exported JSON discrepancy report to %s", out_path)
        return out_path

    def export_offending_csvs(self, results: List[DiscrepancyResult]) -> List[Path]:
        """Saves offending row slices to separate CSVs for detailed triage."""
        exported = []
        for r in results:
            if r.offending_data is not None and len(r.offending_data) > 0:
                safe_name = f"{r.check_name.lower().replace(' ', '_')}.csv"
                dest = self.csv_dir / safe_name
                r.offending_data.to_csv(dest, index=False)
                exported.append(dest)
                log.info("Saved %d offending records for '%s' to %s", len(r.offending_data), r.check_name, dest)
        return exported

    def generate_markdown_report(
        self, results: List[DiscrepancyResult], filename: str = "DISCREPANCY_AUDIT_REPORT.md"
    ) -> Path:
        """Generates an executive Markdown report."""
        out_path = self.out_dir / filename

        n_errors = sum(1 for r in results if r.severity == "ERROR")
        n_warnings = sum(1 for r in results if r.severity == "WARNING")
        n_info = sum(1 for r in results if r.severity == "INFO")

        lines = [
            "# Dataset Discrepancy & Integrity Audit Report",
            "",
            "## 1. Executive Summary",
            "",
            f"- **Total Checks Evaluated:** {len(results)}",
            f"- **Errors (Breaking / Data-Corruption Risks):** {n_errors}",
            f"- **Warnings (Semantic Nuances & Fallback Triggers):** {n_warnings}",
            f"- **Informational (Normal Data Reductions & Milestones):** {n_info}",
            "",
            "---",
            "",
            "## 2. Findings by Severity",
            "",
            "| Severity | Target Dataset | Check Name | Category | Issues Count | Issues % | Impact |",
            "|---|---|---|---|---|---|---|",
        ]

        # Sort order: ERROR first, then WARNING, then INFO
        severity_order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        sorted_results = sorted(results, key=lambda x: (severity_order.get(x.severity, 9), x.check_name))

        for r in sorted_results:
            badge = f"**{r.severity}**" if r.severity in ("ERROR", "WARNING") else r.severity
            lines.append(
                f"| {badge} | `{r.target_dataset}` | `{r.check_name}` | {r.category} | {r.discrepancy_count:,} | {r.discrepancy_pct:.2f}% | {r.impact} |"
            )

        lines.extend([
            "",
            "---",
            "",
            "## 3. Detailed Descriptions & Remediation Guidance",
            "",
        ])

        for r in sorted_results:
            lines.extend([
                f"### `{r.check_name}` ({r.severity})",
                f"- **Target Dataset:** {r.target_dataset}",
                f"- **Category:** {r.category}",
                f"- **Total Rows / Entities Evaluated:** {r.total_evaluated:,}",
                f"- **Discrepancies Identified:** {r.discrepancy_count:,} ({r.discrepancy_pct:.2f}%)",
                f"- **Description:** {r.description}",
                f"- **Downstream Impact:** {r.impact}",
                f"- **Recommended Remediation:** {r.remediation}",
            ])
            if r.offending_data is not None and len(r.offending_data) > 0:
                csv_ref = f"offending_records/{r.check_name.lower().replace(' ', '_')}.csv"
                lines.append(f"- **Sample Offending Data Export:** [`{csv_ref}`]({csv_ref}) ({len(r.offending_data)} rows)")
            lines.append("")

        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        log.info("Generated Markdown audit report to %s", out_path)
        return out_path
