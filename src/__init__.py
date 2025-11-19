"""
OFI utilities ported from replication project.

This module provides core infrastructure for:
- Reading TAQ .rda files
- Processing NBBO quote data
- Computing Order Flow Imbalance (OFI)
- Timestamp normalization and RTH filtering
"""

from .ofi_utils import (
    ColumnMap,
    resolve_columns,
    read_rda,
    filter_crossed,
    parse_trading_day_from_filename,
    build_tob_series_1s,
    compute_ofi_top_of_book,
    run_regression_with_hc,
)

__all__ = [
    "ColumnMap",
    "resolve_columns",
    "read_rda",
    "filter_crossed",
    "parse_trading_day_from_filename",
    "build_tob_series_1s",
    "compute_ofi_top_of_book",
    "run_regression_with_hc",
]
