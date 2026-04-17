"""Gamma API client for Polymarket market discovery."""

from __future__ import annotations

import json
import aiohttp
import structlog
from typing import Any

logger = structlog.get_logger()


class GammaClient:
    """Async client for Polymarket Gamma API."""

    def __init__(self, base_url: str, session: aiohttp.ClientSession) -> None:
        self.base_url = base_url.rstrip("/")
        self.session = session

    def _match_tags(self, text: str, tags: list[str]) -> bool:
        text_lower = text.lower()
        return any(t.lower() in text_lower for t in tags)

    def _parse_market(self, event: dict, market: dict, tags: list[str]) -> dict | None:
        """Parse a single market from an event, return dict if it matches tags."""
        title = (event.get("title") or "").lower()
        question = (market.get("question") or "").lower()
        m_tags_str = market.get("tags") or "[]"
        try:
            m_tags = json.loads(m_tags_str) if isinstance(m_tags_str, str) else m_tags_str
        except json.JSONDecodeError:
            m_tags = []

        all_text = f"{title} {question} {market.get('category', '') or ''} {event.get('category', '') or ''}"
        all_text += " " + " ".join(str(t) for t in m_tags + (event.get("tags") or []))

        if not self._match_tags(all_text, tags):
            return None

        token_ids_str = market.get("clobTokenIds") or "[]"
        outcomes_str = market.get("outcomes") or "[]"
        try:
            token_ids = json.loads(token_ids_str) if isinstance(token_ids_str, str) else token_ids_str
        except json.JSONDecodeError:
            token_ids = []
        try:
            outcomes = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
        except json.JSONDecodeError:
            outcomes = []

        if not token_ids:
            return None

        return {
            "event_id": event.get("id", ""),
            "market_id": market.get("id", ""),
            "condition_id": market.get("conditionId", ""),
            "question": market.get("question", ""),
            "slug": market.get("slug", ""),
            "outcomes": outcomes,
            "token_ids": token_ids,
            "active": market.get("active", False),
            "closed": market.get("closed", False),
            "tick_size": market.get("minimumTickSize", "0.01"),
            "min_order_size": market.get("minimumOrderSize", "1"),
            "category": market.get("category") or event.get("category", ""),
            "tags": m_tags,
            "end_date": market.get("endDate"),
        }

    async def fetch_btc_markets(self, tags: list[str]) -> list[dict]:
        """Fetch BTC-related active markets with order books enabled.

        Uses a two-pass approach:
        1. Search events by tag slug (fast, targeted)
        2. If not enough results, do a keyword-filtered scan with early termination
        """
        all_markets: list[dict] = []
        seen_ids: set[str] = set()
        page_size = 100

        # Fetch events in large pages and filter client-side
        offset = 0
        max_pages = 10
        timeout = aiohttp.ClientTimeout(total=15)

        for _ in range(max_pages):
            params = {
                "active": "true",
                "closed": "false",
                "limit": str(page_size),
                "offset": str(offset),
            }
            try:
                async with self.session.get(
                    f"{self.base_url}/events", params=params, timeout=timeout
                ) as resp:
                    if resp.status != 200:
                        logger.error("gamma_api_error", status=resp.status)
                        break
                    events = await resp.json()
            except Exception as e:
                logger.warning("gamma_api_timeout", error=str(e))
                break

            if not events:
                break

            for event in events:
                for market in event.get("markets") or []:
                    mid = market.get("id", "")
                    if mid in seen_ids:
                        continue
                    parsed = self._parse_market(event, market, tags)
                    if parsed:
                        seen_ids.add(mid)
                        all_markets.append(parsed)

            offset += page_size
            logger.info("gamma_page_fetched", offset=offset, found_so_far=len(all_markets))

            # Early exit if we got a partial page (no more data)
            if len(events) < page_size:
                break

        logger.info("gamma_markets_found", total=len(all_markets))
        return all_markets

    async def fetch_market_book(self, token_id: str) -> dict[str, Any] | None:
        """Fetch current order book snapshot from CLOB REST."""
        try:
            async with self.session.get(
                f"https://clob.polymarket.com/book",
                params={"token_id": token_id},
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
        except Exception as e:
            logger.warning("fetch_book_error", token_id=token_id, error=str(e))
        return None

    async def fetch_tick_size(self, token_id: str) -> str | None:
        try:
            async with self.session.get(
                f"https://clob.polymarket.com/tick-size",
                params={"token_id": token_id},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("minimum_tick_size")
        except Exception:
            pass
        return None

    async def fetch_fee_rate(self, token_id: str) -> float | None:
        try:
            async with self.session.get(
                f"https://clob.polymarket.com/fee-rate",
                params={"token_id": token_id},
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("base_fee")
        except Exception:
            pass
        return None
