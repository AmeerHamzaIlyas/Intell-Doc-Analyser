"""Google GenAI client wrapper with retry logic and error resilience."""
from typing import List, Optional
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from app.config import settings
from app.logging_config import get_logger

logger = get_logger(__name__)


class GeminiClient:
    """Wrapper for Google Gemini API models and embeddings."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                logger.warning("Could not initialize google-genai client: %s", e)

    def is_available(self) -> bool:
        """Check whether Gemini API is configured and accessible."""
        return self._client is not None and bool(self.api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.2,
    ) -> str:
        """Generate text using Gemini models with retry on transient errors."""
        if not self.is_available():
            raise RuntimeError("GeminiClient is not available. Please configure GEMINI_API_KEY.")

        target_model = model or settings.GEMINI_MODEL
        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
        )

        response = self._client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=config,
        )
        return response.text.strip() if response.text else ""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True
    )
    def embed_content(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Compute vector embeddings for a list of texts."""
        if not self.is_available():
            raise RuntimeError("GeminiClient is not available for embeddings.")

        target_model = model or settings.GEMINI_EMBEDDING_MODEL
        embeddings: List[List[float]] = []

        # Process in batches if necessary
        for text in texts:
            response = self._client.models.embed_content(
                model=target_model,
                contents=text,
            )
            # Response contains list of embeddings
            if response.embeddings:
                embeddings.append(list(response.embeddings[0].values))
            else:
                embeddings.append([0.0] * settings.EMBEDDING_DIM)

        return embeddings


# Global Gemini client
gemini_client = GeminiClient()
