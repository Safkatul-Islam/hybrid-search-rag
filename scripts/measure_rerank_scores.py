"""Measure the live Cohere rerank score distribution over the indexed corpus.

A read-only tuning tool, not part of the serving path. It runs the real
retrieve -> rerank prefix of the pipeline against the live providers and prints
the relevance scores that ``POST /query`` never exposes, so
``RERANK_SCORE_THRESHOLD`` can be chosen from data instead of guessed.

Deliberate constraints:

- **No LLM call.** Generation is irrelevant to the threshold decision.
- **No writes.** SQLite and Pinecone are read only.
- **No document text is printed** — only ``chunk_id``, source title, and score.

Usage:
    .venv/Scripts/python.exe scripts/measure_rerank_scores.py
"""

from __future__ import annotations

import sys
import time
from collections import Counter
from dataclasses import dataclass

from src.api.app import build_services
from src.config import get_settings
from src.models.chunk import Chunk
from src.reranking.reranker import CohereReranker, RerankedChunk, RerankError

# Measurement is not latency-sensitive, so it tolerates a slower rerank call
# than the serving path does. Production keeps ``settings.cohere_timeout``.
MEASURE_TIMEOUT = 120.0
MEASURE_ATTEMPTS = 3
# A Cohere Trial key allows 10 API calls/minute and each probe spends two (one
# embed for dense retrieval, one rerank). Pace well under that ceiling rather
# than racing it: the whole run is a few minutes and correctness beats speed.
PROBE_DELAY_SECONDS = 14.0
RATE_LIMIT_BACKOFF_SECONDS = 65.0

# --- Query set -------------------------------------------------------------
# ``expected`` is the source title that should supply the answer, or None when
# the corpus genuinely cannot answer the question. Every "unanswerable" query
# below was checked against all six sample documents by hand.

ANSWERABLE = "answerable"
ADJACENT = "unanswerable-adjacent"
OFF_TOPIC = "unanswerable-off-topic"


@dataclass(frozen=True)
class Probe:
    """One measurement query and its ground truth."""

    text: str
    category: str
    expected: str | None


PROBES: tuple[Probe, ...] = (
    # A. Answerable, phrased close to the source wording (one per document).
    Probe(
        "How many days of paid annual leave do full-time employees get?",
        ANSWERABLE,
        "handbook.md",
    ),
    Probe(
        "How often must employee account passwords be rotated?",
        ANSWERABLE,
        "security_policy.md",
    ),
    Probe(
        "What percentage of base salary does the company match in the 401(k)?",
        ANSWERABLE,
        "benefits.md",
    ),
    Probe(
        "What is the per-diem allowance for international travel?",
        ANSWERABLE,
        "travel_policy.md",
    ),
    Probe(
        "What hours is the IT help desk staffed?",
        ANSWERABLE,
        "it_support.md",
    ),
    Probe(
        "What is the maximum value gift I may accept from a vendor?",
        ANSWERABLE,
        "code_of_conduct.md",
    ),
    # B. Answerable, but phrased in the user's words rather than the source's.
    Probe(
        "If my laptop gets stolen, how quickly do I need to tell someone?",
        ANSWERABLE,
        "handbook.md",
    ),
    Probe(
        "Can I fly business class to a conference in Asia?",
        ANSWERABLE,
        "travel_policy.md",
    ),
    Probe(
        "How long before my stock grant starts paying out?",
        ANSWERABLE,
        "benefits.md",
    ),
    # C. Unanswerable, but topically adjacent -- the decisive set. Retrieval
    # will still return confident-looking neighbours for all of these.
    Probe("What is the bereavement leave policy?", ADJACENT, None),
    Probe("How much notice must I give when I resign?", ADJACENT, None),
    Probe("Is a sabbatical available after five years of service?", ADJACENT, None),
    Probe(
        "What is the mileage reimbursement rate for using my personal car?",
        ADJACENT,
        None,
    ),
    Probe("Which antivirus software is installed on company laptops?", ADJACENT, None),
    # D. Unanswerable and unrelated -- the sanity floor.
    Probe("What is the best way to bake sourdough bread?", OFF_TOPIC, None),
    Probe("Who won the 2018 FIFA World Cup?", OFF_TOPIC, None),
    Probe("How do I change the oil in a diesel generator?", OFF_TOPIC, None),
)


@dataclass(frozen=True)
class Measurement:
    """Scored results for one probe."""

    probe: Probe
    scores: list[tuple[float, str, str]]  # (score, chunk_id, source_title)
    capped_at: int

    @property
    def top_score(self) -> float:
        return self.scores[0][0] if self.scores else 0.0

    @property
    def expected_min_in_cap(self) -> float | None:
        """Lowest score among correct-document chunks surviving the top-N cap."""
        if self.probe.expected is None:
            return None
        hits = [
            score
            for score, _, title in self.scores[: self.capped_at]
            if title == self.probe.expected
        ]
        return min(hits) if hits else None


def _rerank_with_retry(
    reranker: CohereReranker, query: str, candidates: list[Chunk]
) -> list[RerankedChunk]:
    """Rerank, retrying transient failures; raise if every attempt fails."""
    for attempt in range(1, MEASURE_ATTEMPTS + 1):
        try:
            return reranker.rerank(query, candidates, len(candidates))
        except RerankError:
            if attempt == MEASURE_ATTEMPTS:
                raise
            print(
                f"    (rerank attempt {attempt} failed; backing off "
                f"{RATE_LIMIT_BACKOFF_SECONDS:.0f}s)"
            )
            time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
    raise AssertionError("unreachable")


def main() -> int:
    settings = get_settings()
    services = build_services(settings)
    hybrid = services.hybrid
    query_service = services.query_service
    # Reach past the service boundary deliberately: this tool measures the
    # stages the public API intentionally hides.
    store = query_service._store  # noqa: SLF001
    cap = settings.rerank_top_n
    # Own reranker rather than the service's, purely to widen the timeout.
    reranker = CohereReranker(
        api_key=settings.cohere_api_key,
        model=settings.cohere_rerank_model,
        max_tokens_per_doc=settings.rerank_max_tokens_per_doc,
        timeout=MEASURE_TIMEOUT,
    )

    corpus = store.all_chunks()
    if not corpus:
        print("ABORT: the chunk store is empty -- nothing to measure.")
        return 1

    titles = Counter(chunk.source_title for chunk in corpus)
    print(f"Corpus: {len(corpus)} chunks across {len(titles)} documents")
    for title, count in sorted(titles.items()):
        print(f"  {title:<24} {count} chunks")
    print(f"Rerank cap (rerank_top_n): {cap}\n")

    expected_titles = {p.expected for p in PROBES if p.expected}
    missing = expected_titles - set(titles)
    if missing:
        print(f"ABORT: expected source titles not in corpus: {sorted(missing)}")
        print("The ground truth would be wrong, so no threshold is inferred.")
        return 1

    measurements: list[Measurement] = []
    total = len(PROBES)
    for index, probe in enumerate(PROBES, start=1):
        if index > 1:
            time.sleep(PROBE_DELAY_SECONDS)
        print(f"  [{index}/{total}] {probe.text}", flush=True)
        fused = hybrid.retrieve(probe.text)
        candidates = store.get_chunks([chunk_id for chunk_id, _ in fused])
        if not candidates:
            print(f"ABORT: no candidates retrieved for {probe.text!r}")
            return 1
        # Request every candidate rather than the production top-N: scores are
        # per-document and unaffected by top_n, so this costs the same call and
        # shows the full distribution, including what the cap would discard.
        ranked = _rerank_with_retry(reranker, probe.text, candidates)
        scores = [
            (item.relevance_score, item.chunk.chunk_id, item.chunk.source_title)
            for item in ranked
        ]
        measurements.append(Measurement(probe=probe, scores=scores, capped_at=cap))

    _report(measurements, cap)
    return 0


def _report(measurements: list[Measurement], cap: int) -> None:
    for category in (ANSWERABLE, ADJACENT, OFF_TOPIC):
        rows = [m for m in measurements if m.probe.category == category]
        if not rows:
            continue
        print("=" * 72)
        print(f"{category.upper()}  ({len(rows)} queries)")
        print("=" * 72)
        for m in rows:
            print(f"\n  {m.probe.text}")
            if m.probe.expected:
                print(f"  expected source: {m.probe.expected}")
            for rank, (score, chunk_id, title) in enumerate(m.scores, start=1):
                marker = " " if rank <= cap else "x"  # x = dropped by the cap
                star = "*" if title == m.probe.expected else " "
                print(f"    {marker}{star} {rank:>2}. {score:.4f}  {chunk_id}  {title}")
        print()

    answerable = [m for m in measurements if m.probe.category == ANSWERABLE]
    unanswerable = [m for m in measurements if m.probe.category != ANSWERABLE]

    a_top = min(m.top_score for m in answerable)
    u_max = max(m.top_score for m in unanswerable)
    u_adjacent = max(
        m.top_score for m in unanswerable if m.probe.category == ADJACENT
    )
    correct_mins = [
        m.expected_min_in_cap
        for m in answerable
        if m.expected_min_in_cap is not None
    ]
    a_correct_min = min(correct_mins) if correct_mins else None

    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    print(f"  A_top          {a_top:.4f}  lowest best-hit score on an answerable query")
    if a_correct_min is not None:
        print(
            f"  A_correct_min  {a_correct_min:.4f}  lowest correct-doc score within "
            f"the top-{cap} cap"
        )
    print(f"  U_adjacent     {u_adjacent:.4f}  highest score on an adjacent miss")
    print(f"  U_max          {u_max:.4f}  highest score on any unanswerable query")
    print()

    if u_max < a_top:
        gap = a_top - u_max
        suggested = u_max + 0.25 * gap
        print(f"  SEPARATION FOUND: gap of {gap:.4f} between {u_max:.4f} and {a_top:.4f}")
        print(f"  Suggested RERANK_SCORE_THRESHOLD = {suggested:.2f}")
        print("  (biased to the low end of the gap: prefer answering over declining)")
    else:
        print("  NO CLEAN SEPARATION: an unanswerable query scores at or above the")
        print("  weakest answerable query. Any threshold that blocks the miss would")
        print("  also break a real question.")
        print("  Recommendation: leave RERANK_SCORE_THRESHOLD at 0.0")


if __name__ == "__main__":
    sys.exit(main())
