"""Claude transport — the single place Anthropic's SDK is used.

A thin wrapper over the Messages API. It transports a system prompt and a user
message and returns the model's text; it holds no prompt-building or
citation logic (those live in ``generation/``). Two contracts, matching the
reranker:

- **Failure is visible and leaks nothing.** A provider error becomes a typed
  :class:`LLMError` with a generic, client-safe message; the original is chained
  (``from exc``) for server-side logs only.
- **Response text is extracted defensively.** The Messages API returns a list of
  content blocks; we concatenate the text of every text block rather than
  assuming ``content[0].text``, so multi-block or empty responses are handled.
"""

from __future__ import annotations

from anthropic import Anthropic


class LLMError(RuntimeError):
    """Raised when answer generation fails. Carries a safe message; the
    underlying provider error is available via ``__cause__`` for server logs."""


class ClaudeClient:
    """Generates text with Claude via the Messages API.

    The ``client`` argument allows injecting a fake in tests; in normal use a
    real ``anthropic.Anthropic`` is created from the API key.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
        timeout: float = 60.0,
        client: object | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or Anthropic(api_key=api_key, timeout=timeout)

    def generate(self, *, system: str, user: str) -> str:
        """Return Claude's text answer for a system prompt and user message.

        Raises:
            LLMError: the generation call failed (message is client-safe).
        """
        try:
            message = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except Exception as exc:  # noqa: BLE001 - re-raised as a safe, typed error
            raise LLMError("Answer generation failed") from exc

        return _extract_text(message)


def _extract_text(message: object) -> str:
    """Concatenate the text of every text content block in a Messages response."""
    blocks = getattr(message, "content", None) or []
    parts = [
        block.text
        for block in blocks
        if getattr(block, "type", None) == "text" and getattr(block, "text", None)
    ]
    return "".join(parts)
