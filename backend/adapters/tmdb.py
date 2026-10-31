"""Trailer lookup on TMDB, used by the show_trailer tool.

The v3 API takes the key as a query parameter, so request URLs carry a secret: errors are
logged by type only, never with the URL. Spec: docs/desc/03-llm-and-tool-calling.md.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger(__name__)

VIDEO_TYPES = ("Trailer", "Teaser")  # in order of preference


@dataclass(frozen=True)
class TmdbTrailers:
    api_key: str
    base_url: str = "https://api.themoviedb.org/3"
    timeout_s: float = 10.0
    transport: httpx.BaseTransport | None = None  # tests inject httpx.MockTransport

    def find(self, movie_id: int) -> str | None:
        """YouTube key of the movie's trailer (official first, else a teaser), or None."""
        try:
            with httpx.Client(transport=self.transport, timeout=self.timeout_s) as http:
                response = http.get(
                    f"{self.base_url}/movie/{movie_id}/videos", params={"api_key": self.api_key}
                )
                response.raise_for_status()
        except httpx.HTTPError as exc:
            log.warning("TMDB videos lookup failed for movie %s: %s", movie_id, type(exc).__name__)
            return None
        candidates = [
            video
            for video in response.json().get("results", [])
            if video.get("site") == "YouTube"
            and video.get("type") in VIDEO_TYPES
            and video.get("key")
        ]
        candidates.sort(key=lambda v: (VIDEO_TYPES.index(v["type"]), not v.get("official", False)))
        return candidates[0]["key"] if candidates else None
