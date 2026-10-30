"""
Data Audit and Discrepancy Detection Package
============================================
Comprehensive suite for identifying, isolating, and logging schema, semantic,
referential, temporal, and distributional discrepancies in ecological and
temporal graph datasets (SA02601_v6 visitation and SA02602_v5 flowering).
"""

from .audit import DatasetAuditor, run_full_audit
from .detectors import (
    DistributionalDetector,
    GranularityDetector,
    ReferentialIntegrityDetector,
    SchemaAndDateDetector,
    SemanticStatusDetector,
    TemporalContinuityDetector,
)
from .reporter import AuditReporter

__all__ = [
    "DatasetAuditor",
    "run_full_audit",
    "SchemaAndDateDetector",
    "SemanticStatusDetector",
    "ReferentialIntegrityDetector",
    "GranularityDetector",
    "DistributionalDetector",
    "TemporalContinuityDetector",
    "AuditReporter",
]
