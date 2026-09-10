import time
import random
from functools import wraps
from typing import Callable, Any


class RateLimiter:
    """
    Rate limiter tuned for Gemini's free tier (15 requests/minute).
    Spaces out calls and retries on transient failures / 429s.
    """

    def __init__(self, requests_per_minute: int = 12):
        # Kept below Gemini's actual 15/min limit to leave headroom
        self.requests_per_minute = requests_per_minute
        self.last_request_time = 0

    def wait_if_needed(self):
        """Enforce spacing between requests."""
        elapsed = time.time() - self.last_request_time
        required_delay = 60 / self.requests_per_minute

        if elapsed < required_delay:
            sleep_time = required_delay - elapsed
            print(f"⏳ Rate limiting: waiting {sleep_time:.1f}s before next API call...")
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def retry_with_backoff(self, max_retries: int = 3, base_delay: float = 5):
        """
        Decorator: retries on failure with exponential backoff.
        base_delay=5 means retries wait ~5s, 10s, 20s — reasonable for Gemini,
        not the 30-60s we needed for Mistral.
        """
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs) -> Any:
                last_exception = None

                for attempt in range(max_retries + 1):
                    try:
                        self.wait_if_needed()
                        return func(*args, **kwargs)

                    except Exception as e:
                        last_exception = e
                        error_msg = str(e)
                        is_rate_limit = (
                            "429" in error_msg
                            or "quota" in error_msg.lower()
                            or "rate" in error_msg.lower()
                            or "ResourceExhausted" in error_msg
                        )

                        if attempt < max_retries:
                            delay = base_delay * (2 ** attempt)
                            jitter = random.uniform(0, delay * 0.1)
                            total_delay = delay + jitter

                            print(f"❌ Attempt {attempt + 1} failed: {type(e).__name__}")
                            if is_rate_limit:
                                print("⚠️ Rate limit detected!")
                            print(f"🔄 Retrying in {total_delay:.1f}s (attempt {attempt + 2}/{max_retries + 1})...\n")
                            time.sleep(total_delay)
                        else:
                            print(f"❌ All {max_retries + 1} attempts failed!")

                raise last_exception

            return wrapper
        return decorator


# Global instance — 14 req/min leaves headroom under Gemini's 15/min free-tier limit
rate_limiter = RateLimiter(requests_per_minute=14)