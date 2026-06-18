"""
FinBERT Sentiment Analyzer for financial news articles.

Uses ``ProsusAI/finbert`` from Hugging Face Transformers to classify each
article's sentiment and maps labels to a numeric score:

    positive  →  +1
    neutral   →   0
    negative  →  -1

Architecture
------------
``FinBERTSentimentAnalyzer`` wraps the Hugging Face ``transformers.pipeline``
with the following production-grade features:

- **Lazy loading** — the model is only downloaded / loaded on the first call
  to ``load()`` or any ``analyse_*`` method.
- **Batch inference** — texts are forwarded to the model in configurable
  batches, maximising hardware utilisation while keeping peak memory bounded.
- **Null-safe input** — ``None`` or empty texts are skipped; they receive a
  ``neutral`` label with zero confidence and are never sent to the model.
- **Auto device selection** — CUDA (first GPU) → Apple Silicon MPS → CPU.
- **Consistent output schema** — four new columns are appended to any
  DataFrame processed by ``analyse_dataframe()``.

Output columns
--------------
- ``sentiment_label``      : ``"positive"`` | ``"neutral"`` | ``"negative"``
- ``sentiment_score``      : ``+1`` | ``0`` | ``-1``
- ``sentiment_confidence`` : softmax probability of the winning label, in [0, 1]
- ``analysed_at``          : ISO-8601 UTC timestamp of inference

Usage
-----
    from src.processing.sentiment_analyzer import FinBERTSentimentAnalyzer

    analyzer = FinBERTSentimentAnalyzer(batch_size=32)
    enriched_df = analyzer.analyse_dataframe(news_df)
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.utils.logger import get_logger

UTC = timezone.utc
logger = get_logger(__name__)

# ── Label mapping ─────────────────────────────────────────────────────────────

LABEL_TO_SCORE: dict[str, int] = {
    "positive": 1,
    "neutral":  0,
    "negative": -1,
}

SENTIMENT_COLUMNS: list[str] = [
    "sentiment_label",
    "sentiment_score",
    "sentiment_confidence",
    "analysed_at",
]


# ── Exceptions ────────────────────────────────────────────────────────────────

class SentimentAnalysisError(Exception):
    """Base exception for all sentiment analysis errors."""


class ModelLoadError(SentimentAnalysisError):
    """Failed to load the FinBERT model from Hugging Face."""


class InferenceError(SentimentAnalysisError):
    """Error during model forward pass / pipeline inference."""


# ── Analyzer ─────────────────────────────────────────────────────────────────

class FinBERTSentimentAnalyzer:
    """
    Wraps the ProsusAI/finbert pipeline for financial news sentiment analysis.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier. Default: ``"ProsusAI/finbert"``.
    batch_size : int
        Number of texts per inference batch. Increase for GPU, decrease for
        low-memory CPU. Default: ``32``.
    device : str
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, ``"mps"``.
        ``"auto"`` selects CUDA → MPS → CPU in that priority order.
    max_length : int
        Maximum token sequence length passed to the tokeniser.
        Texts are truncated to this value. FinBERT's hard limit is 512.
        Default: ``512``.
    """

    _SUPPORTED_LABELS: frozenset[str] = frozenset(LABEL_TO_SCORE)

    def __init__(
        self,
        model_name: str = "ProsusAI/finbert",
        batch_size: int = 32,
        device: str = "auto",
        max_length: int = 512,
    ) -> None:
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        if not (1 <= max_length <= 512):
            raise ValueError(f"max_length must be between 1 and 512, got {max_length}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self._device_arg = device
        self._pipeline: Any = None  # lazy-loaded

        logger.debug(
            f"FinBERTSentimentAnalyzer configured | model={model_name} | "
            f"batch_size={batch_size} | device={device} | max_length={max_length}"
        )

    # ── Device resolution ─────────────────────────────────────────────────────

    def _resolve_device(self) -> int | str:
        """
        Return a device specifier understood by the transformers pipeline.

        - ``int >= 0``  → CUDA device index
        - ``-1``        → CPU
        - ``"mps"``     → Apple Silicon GPU
        """
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            logger.warning("torch not found — defaulting to CPU")
            return -1

        arg = self._device_arg.lower()

        if arg == "auto":
            if torch.cuda.is_available():
                logger.info("Device auto-selected: CUDA GPU")
                return 0
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Device auto-selected: Apple Silicon MPS")
                return "mps"
            logger.info("Device auto-selected: CPU (no GPU available)")
            return -1

        if arg == "cpu":
            return -1
        if arg == "cuda":
            return 0
        if arg == "mps":
            return "mps"

        logger.warning(f"Unknown device '{self._device_arg}' — falling back to CPU")
        return -1

    # ── Model loading ─────────────────────────────────────────────────────────

    def load(self) -> "FinBERTSentimentAnalyzer":
        """
        Eagerly load and warm up the FinBERT model.

        Called automatically on the first ``analyse_*`` call. Can also be
        invoked explicitly at startup to amortise load time.

        Returns
        -------
        self
            Fluent interface for chaining: ``analyzer = FinBERTSentimentAnalyzer().load()``.

        Raises
        ------
        ModelLoadError
            If ``transformers`` is not installed or the model cannot be fetched.
        """
        if self._pipeline is not None:
            return self

        try:
            from transformers import pipeline  # noqa: PLC0415
        except ImportError as exc:
            raise ModelLoadError(
                "The 'transformers' package is not installed. "
                "Run: pip install transformers torch"
            ) from exc

        device = self._resolve_device()

        logger.info(f"Loading FinBERT model '{self.model_name}' (this may take a moment on first run) …")

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._pipeline = pipeline(
                    task="text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=device,
                    truncation=True,
                    max_length=self.max_length,
                    top_k=1,
                )
        except OSError as exc:
            raise ModelLoadError(
                f"Failed to load '{self.model_name}' from Hugging Face: {exc}. "
                "Check your internet connection or set HF_HOME to point to a local cache."
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(
                f"Unexpected error while loading '{self.model_name}': {exc}"
            ) from exc

        logger.info(f"FinBERT model loaded successfully ({self.model_name})")
        return self

    # ── Core inference ────────────────────────────────────────────────────────

    def _ensure_loaded(self) -> None:
        if self._pipeline is None:
            self.load()

    def _run_batch(self, texts: list[str]) -> list[dict[str, Any]]:
        """
        Forward a single pre-chunked batch through the pipeline.

        Parameters
        ----------
        texts : list[str]
            Non-empty, non-None text strings (already validated by caller).

        Returns
        -------
        list of dicts
            Each dict has keys ``"label"`` (str) and ``"score"`` (float).

        Raises
        ------
        InferenceError
            If the model raises during the forward pass.
        """
        try:
            # pipeline returns List[List[Dict]] when top_k=1 and input is a list
            raw: list[list[dict[str, Any]]] = self._pipeline(
                texts,
                truncation=True,
                max_length=self.max_length,
            )
        except Exception as exc:  # noqa: BLE001
            raise InferenceError(f"Model inference failed: {exc}") from exc

        results: list[dict[str, Any]] = []
        for item in raw:
            # top_k=1 wraps each prediction in a single-element list
            best: dict[str, Any] = item[0] if isinstance(item, list) else item
            label: str = best["label"].lower()
            if label not in self._SUPPORTED_LABELS:
                logger.warning(f"Unexpected FinBERT label '{label}' — treating as neutral")
                label = "neutral"
            results.append({"label": label, "score": best["score"]})

        return results

    def analyse_texts(
        self,
        texts: list[str | None],
    ) -> list[dict[str, Any]]:
        """
        Run FinBERT sentiment analysis on a list of texts.

        ``None`` and empty-string inputs are treated as missing data: they
        receive ``"neutral"`` / ``0`` / ``0.0`` without hitting the model.

        Parameters
        ----------
        texts : list[str | None]
            Input texts. May contain ``None`` or empty strings.

        Returns
        -------
        list[dict]
            One dict per input with keys:
            ``sentiment_label``, ``sentiment_score``,
            ``sentiment_confidence``, ``analysed_at``.
        """
        self._ensure_loaded()

        analysed_at = datetime.now(UTC).isoformat()

        # Pre-fill all positions with neutral defaults for null/empty inputs
        results: list[dict[str, Any]] = [
            {
                "sentiment_label":      "neutral",
                "sentiment_score":      0,
                "sentiment_confidence": 0.0,
                "analysed_at":          analysed_at,
            }
            for _ in texts
        ]

        # Collect indices and texts that actually need inference
        valid_indices: list[int] = []
        valid_texts: list[str] = []
        for i, text in enumerate(texts):
            if text and text.strip():
                valid_indices.append(i)
                valid_texts.append(text.strip())

        if not valid_texts:
            logger.debug("analyse_texts: all inputs are empty — returning neutral defaults")
            return results

        n_batches = (len(valid_texts) - 1) // self.batch_size + 1
        logger.debug(
            f"Running inference on {len(valid_texts)} texts "
            f"({len(texts) - len(valid_texts)} empty skipped) "
            f"across {n_batches} batch(es) of ≤{self.batch_size}"
        )

        # Batch inference
        batch_preds: list[dict[str, Any]] = []
        for batch_idx, start in enumerate(range(0, len(valid_texts), self.batch_size)):
            batch = valid_texts[start : start + self.batch_size]
            preds = self._run_batch(batch)
            batch_preds.extend(preds)
            logger.debug(f"Batch {batch_idx + 1}/{n_batches} complete ({len(batch)} texts)")

        # Write predictions back into the correct positions
        for original_idx, pred in zip(valid_indices, batch_preds):
            results[original_idx] = {
                "sentiment_label":      pred["label"],
                "sentiment_score":      LABEL_TO_SCORE[pred["label"]],
                "sentiment_confidence": round(pred["score"], 6),
                "analysed_at":          analysed_at,
            }

        return results

    def analyse_dataframe(
        self,
        df: pd.DataFrame,
        *,
        title_col: str = "title",
        description_col: str = "description",
    ) -> pd.DataFrame:
        """
        Enrich a news DataFrame with FinBERT sentiment columns.

        The input text for each article is formed by concatenating ``title``
        and ``description`` (``"<title>. <description>"``). If either field
        is absent or null, only the available field is used.

        Parameters
        ----------
        df : pd.DataFrame
            Input news DataFrame produced by Phase 1 ingestion.
        title_col : str
            Column containing article headlines. Default: ``"title"``.
        description_col : str
            Column containing article summaries. Default: ``"description"``.

        Returns
        -------
        pd.DataFrame
            A copy of ``df`` with four new columns appended:
            ``sentiment_label``, ``sentiment_score``,
            ``sentiment_confidence``, ``analysed_at``.
        """
        if df.empty:
            logger.warning("analyse_dataframe received an empty DataFrame — returning as-is with null sentiment columns")
            result = df.copy()
            for col in SENTIMENT_COLUMNS:
                result[col] = pd.NA
            return result

        logger.info(f"Running sentiment analysis on {len(df)} articles …")

        def _build_text(row: pd.Series) -> str | None:
            title = str(row.get(title_col) or "").strip()
            desc  = str(row.get(description_col) or "").strip()
            if title and desc:
                return f"{title}. {desc}"
            return title or desc or None

        texts: list[str | None] = [_build_text(row) for _, row in df.iterrows()]
        sentiments = self.analyse_texts(texts)

        result = df.copy().reset_index(drop=True)
        sentiment_df = pd.DataFrame(sentiments, index=result.index)
        for col in SENTIMENT_COLUMNS:
            result[col] = sentiment_df[col]

        pos = int((result["sentiment_label"] == "positive").sum())
        neu = int((result["sentiment_label"] == "neutral").sum())
        neg = int((result["sentiment_label"] == "negative").sum())

        logger.info(
            f"Sentiment analysis complete — "
            f"positive: {pos} | neutral: {neu} | negative: {neg} "
            f"| mean score: {result['sentiment_score'].mean():+.3f}"
        )

        return result
