"""
Unit tests for src.processing.sentiment_analyzer.

All tests use a mocked transformers pipeline so that:
- No model is downloaded or loaded
- No GPU / CUDA is required
- Tests run fast and deterministically

Test coverage
-------------
- LABEL_TO_SCORE mapping correctness
- FinBERTSentimentAnalyzer.__init__ validation
- Device resolution logic (auto / cpu / cuda / mps)
- Model loading: success path, missing package, OSError
- analyse_texts: empty inputs, valid inputs, batching, unknown labels
- analyse_dataframe: empty DataFrame, full DataFrame, missing columns
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.processing.sentiment_analyzer import (
    LABEL_TO_SCORE,
    SENTIMENT_COLUMNS,
    FinBERTSentimentAnalyzer,
    InferenceError,
    ModelLoadError,
    SentimentAnalysisError,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pipeline_result(label: str, score: float) -> list[dict]:
    """Return the shape transformers pipeline emits with top_k=1."""
    return [{"label": label, "score": score}]


def _mock_pipeline(predictions: list[tuple[str, float]]) -> MagicMock:
    """
    Build a MagicMock that simulates the transformers pipeline call.

    ``predictions`` is a list of (label, score) tuples, one per input text.
    """
    mock = MagicMock()
    mock.return_value = [_make_pipeline_result(lbl, sc) for lbl, sc in predictions]
    return mock


# ── LABEL_TO_SCORE ────────────────────────────────────────────────────────────

class TestLabelToScore:
    def test_positive_maps_to_plus_one(self) -> None:
        assert LABEL_TO_SCORE["positive"] == 1

    def test_neutral_maps_to_zero(self) -> None:
        assert LABEL_TO_SCORE["neutral"] == 0

    def test_negative_maps_to_minus_one(self) -> None:
        assert LABEL_TO_SCORE["negative"] == -1

    def test_exactly_three_labels(self) -> None:
        assert set(LABEL_TO_SCORE.keys()) == {"positive", "neutral", "negative"}


# ── __init__ validation ───────────────────────────────────────────────────────

class TestFinBERTSentimentAnalyzerInit:
    def test_defaults_are_set(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        assert analyzer.model_name == "ProsusAI/finbert"
        assert analyzer.batch_size == 32
        assert analyzer.max_length == 512
        assert analyzer._device_arg == "auto"
        assert analyzer._pipeline is None

    def test_custom_params_stored(self) -> None:
        analyzer = FinBERTSentimentAnalyzer(
            model_name="bert-base-uncased",
            batch_size=8,
            device="cpu",
            max_length=128,
        )
        assert analyzer.model_name == "bert-base-uncased"
        assert analyzer.batch_size == 8
        assert analyzer._device_arg == "cpu"
        assert analyzer.max_length == 128

    def test_invalid_batch_size_raises(self) -> None:
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            FinBERTSentimentAnalyzer(batch_size=0)

    def test_max_length_too_low_raises(self) -> None:
        with pytest.raises(ValueError, match="max_length must be between 1 and 512"):
            FinBERTSentimentAnalyzer(max_length=0)

    def test_max_length_too_high_raises(self) -> None:
        with pytest.raises(ValueError, match="max_length must be between 1 and 512"):
            FinBERTSentimentAnalyzer(max_length=513)


# ── Device resolution ─────────────────────────────────────────────────────────

class TestResolveDevice:
    def _analyzer(self, device: str) -> FinBERTSentimentAnalyzer:
        return FinBERTSentimentAnalyzer(device=device)

    def test_explicit_cpu_returns_minus_one(self) -> None:
        analyzer = self._analyzer("cpu")
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = analyzer._resolve_device()
        assert result == -1

    def test_explicit_cuda_returns_zero(self) -> None:
        analyzer = self._analyzer("cuda")
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = analyzer._resolve_device()
        assert result == 0

    def test_explicit_mps_returns_mps_string(self) -> None:
        analyzer = self._analyzer("mps")
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = analyzer._resolve_device()
        assert result == "mps"

    def test_auto_picks_cuda_when_available(self) -> None:
        analyzer = self._analyzer("auto")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = analyzer._resolve_device()
        assert result == 0

    def test_auto_picks_mps_when_no_cuda(self) -> None:
        analyzer = self._analyzer("auto")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = analyzer._resolve_device()
        assert result == "mps"

    def test_auto_falls_back_to_cpu(self) -> None:
        analyzer = self._analyzer("auto")
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = analyzer._resolve_device()
        assert result == -1

    def test_unknown_device_falls_back_to_cpu(self) -> None:
        analyzer = self._analyzer("tpu")
        mock_torch = MagicMock()
        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = analyzer._resolve_device()
        assert result == -1


# ── Model loading ─────────────────────────────────────────────────────────────

class TestModelLoading:
    def test_load_sets_pipeline(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        mock_pipe = MagicMock()
        mock_transformers = MagicMock()
        mock_transformers.pipeline.return_value = mock_pipe

        with patch.dict("sys.modules", {"transformers": mock_transformers}), \
             patch("src.processing.sentiment_analyzer.FinBERTSentimentAnalyzer._resolve_device", return_value=-1):
            analyzer.load()
            mock_transformers.pipeline.assert_called_once()

        assert analyzer._pipeline is mock_pipe

    def test_load_is_idempotent(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        sentinel = MagicMock()
        analyzer._pipeline = sentinel

        analyzer.load()  # should not re-load

        assert analyzer._pipeline is sentinel

    def test_load_returns_self_for_chaining(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        result = analyzer.load()
        assert result is analyzer

    def test_missing_transformers_raises_model_load_error(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        with patch.dict("sys.modules", {"transformers": None}):
            with pytest.raises(ModelLoadError, match="transformers.*not installed"):
                analyzer.load()

    def test_os_error_raises_model_load_error(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        mock_transformers = MagicMock()
        mock_transformers.pipeline.side_effect = OSError("model not found")

        with patch.dict("sys.modules", {"transformers": mock_transformers}), \
             patch("src.processing.sentiment_analyzer.FinBERTSentimentAnalyzer._resolve_device", return_value=-1):
            with pytest.raises(ModelLoadError, match="Failed to load"):
                analyzer.load()

    def test_model_load_error_is_sentiment_analysis_error(self) -> None:
        assert issubclass(ModelLoadError, SentimentAnalysisError)

    def test_inference_error_is_sentiment_analysis_error(self) -> None:
        assert issubclass(InferenceError, SentimentAnalysisError)


# ── _run_batch ────────────────────────────────────────────────────────────────

class TestRunBatch:
    def _analyzer_with_mock_pipeline(self, predictions: list[tuple[str, float]]) -> FinBERTSentimentAnalyzer:
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = _mock_pipeline(predictions)
        return analyzer

    def test_returns_correct_labels_and_scores(self) -> None:
        analyzer = self._analyzer_with_mock_pipeline([
            ("positive", 0.95),
            ("negative", 0.88),
            ("neutral",  0.72),
        ])
        results = analyzer._run_batch(["text1", "text2", "text3"])
        assert results[0] == {"label": "positive", "score": 0.95}
        assert results[1] == {"label": "negative", "score": 0.88}
        assert results[2] == {"label": "neutral",  "score": 0.72}

    def test_label_case_normalised(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = MagicMock(
            return_value=[[{"label": "POSITIVE", "score": 0.9}]]
        )
        results = analyzer._run_batch(["Apple beats earnings"])
        assert results[0]["label"] == "positive"

    def test_unknown_label_treated_as_neutral(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = MagicMock(
            return_value=[[{"label": "unknown_label", "score": 0.6}]]
        )
        results = analyzer._run_batch(["some text"])
        assert results[0]["label"] == "neutral"

    def test_pipeline_exception_raises_inference_error(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = MagicMock(side_effect=RuntimeError("CUDA OOM"))
        with pytest.raises(InferenceError, match="Model inference failed"):
            analyzer._run_batch(["some text"])


# ── analyse_texts ─────────────────────────────────────────────────────────────

class TestAnalyseTexts:
    def _analyzer(self, predictions: list[tuple[str, float]]) -> FinBERTSentimentAnalyzer:
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = _mock_pipeline(predictions)
        return analyzer

    def test_empty_list_returns_empty_list(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        results = analyzer.analyse_texts([])
        assert results == []
        analyzer._pipeline.assert_not_called()

    def test_all_none_inputs_return_neutral_defaults(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        results = analyzer.analyse_texts([None, None, ""])
        assert all(r["sentiment_label"] == "neutral" for r in results)
        assert all(r["sentiment_score"] == 0 for r in results)
        assert all(r["sentiment_confidence"] == 0.0 for r in results)
        analyzer._pipeline.assert_not_called()

    def test_valid_text_returns_correct_sentiment(self) -> None:
        analyzer = self._analyzer([("positive", 0.97)])
        results = analyzer.analyse_texts(["Apple stock surges on strong earnings"])
        assert len(results) == 1
        assert results[0]["sentiment_label"] == "positive"
        assert results[0]["sentiment_score"] == 1
        assert results[0]["sentiment_confidence"] == pytest.approx(0.97, abs=1e-5)

    def test_mixed_valid_and_null_inputs(self) -> None:
        analyzer = self._analyzer([("negative", 0.88)])
        results = analyzer.analyse_texts([None, "Market sell-off continues"])
        assert results[0]["sentiment_label"] == "neutral"   # null input
        assert results[0]["sentiment_score"] == 0
        assert results[1]["sentiment_label"] == "negative"
        assert results[1]["sentiment_score"] == -1

    def test_all_three_labels_mapped_correctly(self) -> None:
        analyzer = self._analyzer([
            ("positive", 0.95),
            ("neutral",  0.80),
            ("negative", 0.91),
        ])
        results = analyzer.analyse_texts(["text1", "text2", "text3"])
        assert results[0]["sentiment_score"] ==  1
        assert results[1]["sentiment_score"] ==  0
        assert results[2]["sentiment_score"] == -1

    def test_confidence_is_rounded_to_six_places(self) -> None:
        analyzer = self._analyzer([("positive", 0.9876543210)])
        results = analyzer.analyse_texts(["test"])
        assert len(str(results[0]["sentiment_confidence"]).split(".")[-1]) <= 6

    def test_analysed_at_is_iso_format(self) -> None:
        from datetime import datetime
        analyzer = self._analyzer([("neutral", 0.7)])
        results = analyzer.analyse_texts(["test"])
        # Should not raise
        datetime.fromisoformat(results[0]["analysed_at"])

    def test_batching_processes_all_texts(self) -> None:
        """With batch_size=2 and 5 texts, the pipeline is called 3 times."""
        call_results: list[list] = []

        def fake_pipeline(texts: list[str], **_kwargs: object) -> list:
            batch_result = [_make_pipeline_result("neutral", 0.8) for _ in texts]
            call_results.append(texts)
            return batch_result

        analyzer = FinBERTSentimentAnalyzer(batch_size=2)
        analyzer._pipeline = fake_pipeline

        texts = ["t1", "t2", "t3", "t4", "t5"]
        results = analyzer.analyse_texts(texts)

        assert len(results) == 5
        assert sum(len(b) for b in call_results) == 5
        assert len(call_results) == 3  # ceil(5/2)

    def test_output_keys_match_sentiment_columns(self) -> None:
        analyzer = self._analyzer([("positive", 0.9)])
        results = analyzer.analyse_texts(["Apple beats Q2 estimates"])
        assert set(results[0].keys()) == set(SENTIMENT_COLUMNS)


# ── analyse_dataframe ─────────────────────────────────────────────────────────

class TestAnalyseDataframe:
    def _make_news_df(self, rows: list[dict]) -> pd.DataFrame:
        defaults = {
            "ticker": "AAPL",
            "title": "",
            "description": None,
            "url": "https://example.com",
        }
        return pd.DataFrame([{**defaults, **r} for r in rows])

    def _analyzer(self, predictions: list[tuple[str, float]]) -> FinBERTSentimentAnalyzer:
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = _mock_pipeline(predictions)
        return analyzer

    def test_empty_dataframe_returns_with_null_sentiment_columns(self) -> None:
        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = MagicMock()
        empty_df = pd.DataFrame(columns=["ticker", "title", "description"])
        result = analyzer.analyse_dataframe(empty_df)
        for col in SENTIMENT_COLUMNS:
            assert col in result.columns
        analyzer._pipeline.assert_not_called()

    def test_sentiment_columns_added_to_dataframe(self) -> None:
        analyzer = self._analyzer([("positive", 0.9), ("negative", 0.85)])
        df = self._make_news_df([
            {"title": "Apple surges", "description": "Strong Q2 results"},
            {"title": "Tesla falls",  "description": "Missed revenue target"},
        ])
        result = analyzer.analyse_dataframe(df)
        for col in SENTIMENT_COLUMNS:
            assert col in result.columns

    def test_row_count_preserved(self) -> None:
        analyzer = self._analyzer([("neutral", 0.7), ("positive", 0.9)])
        df = self._make_news_df([
            {"title": "Markets steady"},
            {"title": "Tech rally"},
        ])
        result = analyzer.analyse_dataframe(df)
        assert len(result) == 2

    def test_original_columns_preserved(self) -> None:
        analyzer = self._analyzer([("neutral", 0.7)])
        df = self._make_news_df([{"title": "News", "url": "https://example.com/1"}])
        result = analyzer.analyse_dataframe(df)
        assert "ticker" in result.columns
        assert "url" in result.columns

    def test_title_only_used_when_description_missing(self) -> None:
        """When description is None, only title should be sent to the pipeline."""
        captured: list[list[str]] = []

        def fake_pipeline(texts: list[str], **_kwargs: object) -> list:
            captured.extend(texts)
            return [_make_pipeline_result("positive", 0.9) for _ in texts]

        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = fake_pipeline
        df = self._make_news_df([{"title": "Apple beats estimates", "description": None}])
        analyzer.analyse_dataframe(df)
        assert captured[0] == "Apple beats estimates"

    def test_title_and_description_concatenated(self) -> None:
        captured: list[str] = []

        def fake_pipeline(texts: list[str], **_kwargs: object) -> list:
            captured.extend(texts)
            return [_make_pipeline_result("positive", 0.9) for _ in texts]

        analyzer = FinBERTSentimentAnalyzer()
        analyzer._pipeline = fake_pipeline
        df = self._make_news_df([{
            "title": "Apple beats estimates",
            "description": "Record iPhone sales drive revenue",
        }])
        analyzer.analyse_dataframe(df)
        assert captured[0] == "Apple beats estimates. Record iPhone sales drive revenue"

    def test_missing_title_col_falls_back_to_empty(self) -> None:
        analyzer = self._analyzer([("neutral", 0.7)])
        df = pd.DataFrame([{"description": "Some news"}])
        result = analyzer.analyse_dataframe(df, title_col="title", description_col="description")
        assert result["sentiment_label"].iloc[0] == "neutral"

    def test_sentiment_scores_correct_sign(self) -> None:
        analyzer = self._analyzer([
            ("positive", 0.95),
            ("neutral",  0.80),
            ("negative", 0.91),
        ])
        df = self._make_news_df([
            {"title": "Good news"},
            {"title": "Flat market"},
            {"title": "Bad news"},
        ])
        result = analyzer.analyse_dataframe(df)
        assert result["sentiment_score"].tolist() == [1, 0, -1]

    def test_dataframe_is_not_mutated_in_place(self) -> None:
        analyzer = self._analyzer([("positive", 0.9)])
        df = self._make_news_df([{"title": "Apple rallies"}])
        original_cols = list(df.columns)
        analyzer.analyse_dataframe(df)
        assert list(df.columns) == original_cols
