"""
Unit tests for scripts/run_sentiment.py.

Covers the new --input-file mode and the _parse_input_filename helper,
while verifying that existing Finnhub discovery behaviour is unchanged.

Test classes
------------
TestParseInputFilename    — pure filename → (ticker, output_tag) logic
TestFindInputFiles        — filesystem-based CSV discovery
TestArgParser             — --input-file argument is wired into argparse
TestMainInputFileBranch   — end-to-end main() with --input-file (mocked I/O)
TestMainDiscoveryBranch   — dry-run guard in existing discovery branch
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Bootstrap: load the script as a module without a real FINNHUB_API_KEY.
# We pre-populate sys.modules with a mock for src.utils.config so that
# Settings validation never runs during the import.
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT_PATH  = _PROJECT_ROOT / "scripts" / "run_sentiment.py"

_mock_settings = MagicMock()
_mock_settings.finbert_model      = "ProsusAI/finbert"
_mock_settings.finbert_batch_size = 32
_mock_settings.finbert_device     = "auto"
_mock_settings.log_level          = "INFO"

_mock_config_module         = MagicMock()
_mock_config_module.settings = _mock_settings

# Ensure the project root is on sys.path so `src.*` imports resolve.
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

with patch.dict("sys.modules", {"src.utils.config": _mock_config_module}):
    spec = importlib.util.spec_from_file_location("run_sentiment", _SCRIPT_PATH)
    _rs = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(_rs)  # type: ignore[union-attr]

# Convenience aliases for the names we test directly.
_parse_input_filename = _rs._parse_input_filename
_find_input_files     = _rs._find_input_files
_parse_args           = _rs._parse_args


# ── _parse_input_filename ─────────────────────────────────────────────────────


class TestParseInputFilename:
    """Tests for the _parse_input_filename helper (pure function)."""

    # ── GDELT filenames ──────────────────────────────────────────────────────

    def test_gdelt_full_year_returns_ticker_and_year(self) -> None:
        p = Path("NVDA_gdelt_2025-01-01_2025-12-31.csv")
        ticker, tag = _parse_input_filename(p)
        assert ticker == "NVDA"
        assert tag == "2025"

    def test_gdelt_partial_year_returns_start_year(self) -> None:
        p = Path("AMZN_gdelt_2025-01-01_2025-03-31.csv")
        ticker, tag = _parse_input_filename(p)
        assert ticker == "AMZN"
        assert tag == "2025"

    def test_gdelt_multi_char_ticker(self) -> None:
        p = Path("MSFT_gdelt_2024-06-01_2024-12-31.csv")
        ticker, tag = _parse_input_filename(p)
        assert ticker == "MSFT"
        assert tag == "2024"

    def test_gdelt_cross_year_range_uses_start_year(self) -> None:
        p = Path("TSLA_gdelt_2024-10-01_2025-03-31.csv")
        ticker, tag = _parse_input_filename(p)
        assert ticker == "TSLA"
        assert tag == "2024"

    def test_gdelt_single_letter_ticker(self) -> None:
        p = Path("F_gdelt_2025-01-01_2025-06-30.csv")
        ticker, tag = _parse_input_filename(p)
        assert ticker == "F"
        assert tag == "2025"

    # ── Finnhub filenames ────────────────────────────────────────────────────

    def test_finnhub_returns_ticker_and_date_tag(self) -> None:
        p = Path("AAPL_news_2026-06-15.csv")
        ticker, tag = _parse_input_filename(p)
        assert ticker == "AAPL"
        assert tag == "2026-06-15"

    def test_finnhub_five_char_ticker(self) -> None:
        p = Path("GOOGL_news_2025-12-01.csv")
        ticker, tag = _parse_input_filename(p)
        assert ticker == "GOOGL"
        assert tag == "2025-12-01"

    # ── Absolute path still works (only stem is inspected) ───────────────────

    def test_absolute_path_gdelt(self) -> None:
        p = Path("/data/raw/gdelt/NVDA_gdelt_2025-01-01_2025-12-31.csv")
        ticker, tag = _parse_input_filename(p)
        assert ticker == "NVDA"
        assert tag == "2025"

    def test_absolute_path_finnhub(self) -> None:
        p = Path("/data/raw/AAPL_news_2026-06-15.csv")
        ticker, tag = _parse_input_filename(p)
        assert ticker == "AAPL"
        assert tag == "2026-06-15"

    # ── Invalid filenames raise ValueError ───────────────────────────────────

    def test_unknown_pattern_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Cannot infer ticker"):
            _parse_input_filename(Path("random_file.csv"))

    def test_lowercase_ticker_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _parse_input_filename(Path("nvda_gdelt_2025-01-01_2025-12-31.csv"))

    def test_missing_date_in_gdelt_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _parse_input_filename(Path("NVDA_gdelt_2025-01-01.csv"))

    def test_summary_file_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            _parse_input_filename(Path("gdelt_summary_2025-01-01_2025-03-31.csv"))

    def test_error_message_contains_filename(self) -> None:
        bad = Path("not_a_valid_file.csv")
        with pytest.raises(ValueError, match="not_a_valid_file.csv"):
            _parse_input_filename(bad)


# ── _find_input_files ─────────────────────────────────────────────────────────


class TestFindInputFiles:
    """Tests for the existing Finnhub CSV discovery logic (unchanged)."""

    def _make_csv(self, directory: Path, name: str) -> Path:
        p = directory / name
        p.write_text("ticker,title\nAAPL,headline\n")
        return p

    def _patched_input_dir(self, tmp_path: Path):
        """Context manager that patches both INPUT_DIR and _PROJECT_ROOT."""
        return (
            patch.object(_rs, "INPUT_DIR", tmp_path),
            patch.object(_rs, "_PROJECT_ROOT", tmp_path.parent),
        )

    def test_auto_discovers_matching_csvs(self, tmp_path: Path) -> None:
        self._make_csv(tmp_path, "AAPL_news_2026-06-15.csv")
        self._make_csv(tmp_path, "TSLA_news_2026-06-15.csv")

        with patch.object(_rs, "INPUT_DIR", tmp_path), \
             patch.object(_rs, "_PROJECT_ROOT", tmp_path.parent):
            pairs = _find_input_files(None, "2026-06-15")

        tickers = [t for t, _ in pairs]
        assert "AAPL" in tickers
        assert "TSLA" in tickers

    def test_specific_ticker_found(self, tmp_path: Path) -> None:
        self._make_csv(tmp_path, "NVDA_news_2026-06-15.csv")

        with patch.object(_rs, "INPUT_DIR", tmp_path), \
             patch.object(_rs, "_PROJECT_ROOT", tmp_path.parent):
            pairs = _find_input_files(["NVDA"], "2026-06-15")

        assert len(pairs) == 1
        assert pairs[0][0] == "NVDA"

    def test_specific_ticker_missing_returns_empty(self, tmp_path: Path) -> None:
        with patch.object(_rs, "INPUT_DIR", tmp_path), \
             patch.object(_rs, "_PROJECT_ROOT", tmp_path.parent):
            pairs = _find_input_files(["AAPL"], "2026-06-15")

        assert pairs == []

    def test_no_csvs_for_date_returns_empty(self, tmp_path: Path) -> None:
        self._make_csv(tmp_path, "AAPL_news_2026-06-14.csv")

        with patch.object(_rs, "INPUT_DIR", tmp_path), \
             patch.object(_rs, "_PROJECT_ROOT", tmp_path.parent):
            pairs = _find_input_files(None, "2026-06-15")

        assert pairs == []

    def test_gdelt_files_not_discovered_by_date_mode(self, tmp_path: Path) -> None:
        """GDELT files should NOT be picked up by the date-based discovery path."""
        self._make_csv(tmp_path, "NVDA_gdelt_2025-01-01_2025-12-31.csv")

        with patch.object(_rs, "INPUT_DIR", tmp_path), \
             patch.object(_rs, "_PROJECT_ROOT", tmp_path.parent):
            pairs = _find_input_files(None, "2025-01-01")

        assert pairs == []


# ── Argument parser ───────────────────────────────────────────────────────────


class TestArgParser:
    """Verify the --input-file argument is wired up correctly."""

    def test_input_file_arg_recognised(self) -> None:
        with patch("sys.argv", ["run_sentiment.py", "--input-file", "/some/path.csv"]):
            args = _parse_args()
        assert args.input_file == "/some/path.csv"

    def test_input_file_defaults_to_none(self) -> None:
        with patch("sys.argv", ["run_sentiment.py"]):
            args = _parse_args()
        assert args.input_file is None

    def test_input_file_coexists_with_model_arg(self) -> None:
        with patch("sys.argv", [
            "run_sentiment.py",
            "--input-file", "/some/path.csv",
            "--model", "bert-base",
        ]):
            args = _parse_args()
        assert args.input_file == "/some/path.csv"
        assert args.model == "bert-base"

    def test_existing_args_still_present(self) -> None:
        with patch("sys.argv", ["run_sentiment.py", "--tickers", "AAPL", "--date", "2026-01-01"]):
            args = _parse_args()
        assert args.tickers == ["AAPL"]
        assert args.date == "2026-01-01"
        assert args.input_file is None


# ── main() with --input-file ──────────────────────────────────────────────────


class TestMainInputFileBranch:
    """Integration-level tests for main() when --input-file is supplied."""

    def _make_gdelt_csv(self, directory: Path, name: str) -> Path:
        p = directory / name
        p.write_text(
            "ticker,source_id,source_name,author,title,description,url,published_at,content,fetched_at\n"
            "NVDA,abc123,,, GPU demand surges,,https://example.com,2025-01-15 10:00:00+00:00,,2025-01-16\n"
        )
        return p

    def _mock_analyzer(self, df: pd.DataFrame) -> MagicMock:
        """Return a mock analyzer whose analyse_dataframe echoes the input."""
        enriched = df.copy()
        enriched["sentiment_label"]      = "positive"
        enriched["sentiment_score"]      = 1
        enriched["sentiment_confidence"] = 0.95
        enriched["analysed_at"]          = "2025-01-16T00:00:00"

        mock = MagicMock()
        mock.analyse_dataframe.return_value = enriched
        return mock

    def test_output_tag_is_year_for_gdelt_file(self, tmp_path: Path) -> None:
        """NVDA_gdelt_2025-01-01_2025-12-31.csv → NVDA_sentiment_2025.csv"""
        csv = self._make_gdelt_csv(tmp_path, "NVDA_gdelt_2025-01-01_2025-12-31.csv")
        output_dir = tmp_path / "processed"

        with (
            patch("sys.argv", ["run_sentiment.py", "--input-file", str(csv)]),
            patch.object(_rs, "INPUT_DIR", tmp_path),
            patch.object(_rs, "OUTPUT_DIR", output_dir),
            patch.object(_rs, "_PROJECT_ROOT", tmp_path),
            patch.object(_rs, "FinBERTSentimentAnalyzer") as MockCls,
        ):
            df_in = pd.read_csv(csv)
            MockCls.return_value = self._mock_analyzer(df_in)
            _rs.main()

        assert (output_dir / "NVDA_sentiment_2025.csv").exists()

    def test_output_tag_is_date_for_finnhub_file(self, tmp_path: Path) -> None:
        """AAPL_news_2026-06-15.csv → AAPL_sentiment_2026-06-15.csv"""
        csv = tmp_path / "AAPL_news_2026-06-15.csv"
        csv.write_text(
            "ticker,title,description,url\n"
            "AAPL,Apple beats Q2,,https://example.com\n"
        )
        output_dir = tmp_path / "processed"

        with (
            patch("sys.argv", ["run_sentiment.py", "--input-file", str(csv)]),
            patch.object(_rs, "INPUT_DIR", tmp_path),
            patch.object(_rs, "OUTPUT_DIR", output_dir),
            patch.object(_rs, "_PROJECT_ROOT", tmp_path),
            patch.object(_rs, "FinBERTSentimentAnalyzer") as MockCls,
        ):
            df_in = pd.read_csv(csv)
            MockCls.return_value = self._mock_analyzer(df_in)
            _rs.main()

        assert (output_dir / "AAPL_sentiment_2026-06-15.csv").exists()

    def test_missing_input_file_exits_with_error(self, tmp_path: Path) -> None:
        missing = tmp_path / "NVDA_gdelt_2025-01-01_2025-12-31.csv"

        with (
            patch("sys.argv", ["run_sentiment.py", "--input-file", str(missing)]),
            patch.object(_rs, "INPUT_DIR", tmp_path),
            patch.object(_rs, "OUTPUT_DIR", tmp_path / "processed"),
            patch.object(_rs, "_PROJECT_ROOT", tmp_path),
            pytest.raises(SystemExit) as exc_info,
        ):
            _rs.main()

        assert exc_info.value.code == 1

    def test_invalid_filename_exits_with_error(self, tmp_path: Path) -> None:
        bad = tmp_path / "notavalidname.csv"
        bad.write_text("ticker,title\nAAPL,foo\n")

        with (
            patch("sys.argv", ["run_sentiment.py", "--input-file", str(bad)]),
            patch.object(_rs, "INPUT_DIR", tmp_path),
            patch.object(_rs, "OUTPUT_DIR", tmp_path / "processed"),
            patch.object(_rs, "_PROJECT_ROOT", tmp_path),
            pytest.raises(SystemExit) as exc_info,
        ):
            _rs.main()

        assert exc_info.value.code == 1

    def test_discovery_logic_is_skipped_when_input_file_given(self, tmp_path: Path) -> None:
        """_find_input_files must not be called when --input-file is present."""
        csv = self._make_gdelt_csv(tmp_path, "NVDA_gdelt_2025-01-01_2025-12-31.csv")
        output_dir = tmp_path / "processed"

        with (
            patch("sys.argv", ["run_sentiment.py", "--input-file", str(csv)]),
            patch.object(_rs, "INPUT_DIR", tmp_path),
            patch.object(_rs, "OUTPUT_DIR", output_dir),
            patch.object(_rs, "_PROJECT_ROOT", tmp_path),
            patch.object(_rs, "_find_input_files", wraps=_rs._find_input_files) as mock_find,
            patch.object(_rs, "FinBERTSentimentAnalyzer") as MockCls,
        ):
            df_in = pd.read_csv(csv)
            MockCls.return_value = self._mock_analyzer(df_in)
            _rs.main()

        mock_find.assert_not_called()

    def test_dry_run_with_input_file_exits_zero(self, tmp_path: Path) -> None:
        csv = self._make_gdelt_csv(tmp_path, "NVDA_gdelt_2025-01-01_2025-12-31.csv")

        with (
            patch("sys.argv", ["run_sentiment.py", "--input-file", str(csv), "--dry-run"]),
            patch.object(_rs, "INPUT_DIR", tmp_path),
            patch.object(_rs, "OUTPUT_DIR", tmp_path / "processed"),
            patch.object(_rs, "_PROJECT_ROOT", tmp_path),
            pytest.raises(SystemExit) as exc_info,
        ):
            _rs.main()

        assert exc_info.value.code == 0


# ── main() existing discovery branch (unchanged behaviour) ────────────────────


class TestMainDiscoveryBranch:
    """Smoke-tests to confirm the original dry-run / no-files paths still work."""

    def test_dry_run_exits_zero_without_input_file(self) -> None:
        with (
            patch("sys.argv", ["run_sentiment.py", "--dry-run"]),
            pytest.raises(SystemExit) as exc_info,
        ):
            _rs.main()

        assert exc_info.value.code == 0

    def test_no_csv_files_exits_one(self, tmp_path: Path) -> None:
        with (
            patch("sys.argv", ["run_sentiment.py", "--date", "2099-01-01"]),
            patch.object(_rs, "INPUT_DIR", tmp_path),
            patch.object(_rs, "OUTPUT_DIR", tmp_path / "processed"),
            patch.object(_rs, "_PROJECT_ROOT", tmp_path),
            pytest.raises(SystemExit) as exc_info,
        ):
            _rs.main()

        assert exc_info.value.code == 1
