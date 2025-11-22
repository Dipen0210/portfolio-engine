import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import pandas as pd
import requests

BASE_URL = "https://api.coingecko.com/api/v3"
API_KEY = os.environ.get("COINGECKO_API_KEY", "CG-61RX5kqiaAtRHWCQH2tbu2dt")
API_HEADER = os.environ.get("COINGECKO_API_HEADER", "x-cg-demo-api-key")
HEADERS = {"accept": "application/json"}
if API_KEY:
    HEADERS[API_HEADER] = API_KEY

DETAIL_PARAMS = {
    "localization": "false",
    "tickers": "false",
    "market_data": "true",
    "community_data": "false",
    "developer_data": "false",
    "sparkline": "false",
}
VS_CURRENCY = "usd"
MARKET_PAGES = 2
MARKET_PAGE_SIZE = 50
RATE_LIMIT_DELAY = 0.5


def fetch_json(
    session: requests.Session,
    endpoint: str,
    params: Optional[Dict[str, Any]] = None,
    max_retries: int = 5,
) -> Any:
    """GET helper that retries on HTTP 429 with exponential backoff."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(max_retries):
        resp = session.get(url, params=params, timeout=30)
        if resp.status_code == 429:
            time.sleep(2**attempt)
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return resp.json()


def fetch_global_snapshot(session: requests.Session) -> Dict[str, Any]:
    return fetch_json(session, "/global").get("data", {})


def fetch_trending_ids(session: requests.Session) -> List[str]:
    payload = fetch_json(session, "/search/trending")
    coins = payload.get("coins", [])
    return [
        item["item"]["id"]
        for item in coins
        if item.get("item") and item["item"].get("id")
    ]


def fetch_top_mover_ids(session: requests.Session) -> List[str]:
    try:
        payload = fetch_json(
            session,
            "/coins/top_gainers_losers",
            params={"vs_currency": VS_CURRENCY},
        )
    except requests.HTTPError as exc:
        status = getattr(exc.response, "status_code", "unknown")
        print(f"Skipping top gainers/losers (HTTP {status}).")
        return []
    ids: List[str] = []
    for key in ("top_gainers", "top_losers"):
        ids.extend(coin["id"] for coin in payload.get(key, []) if coin.get("id"))
    return ids


def fetch_market_ids(session: requests.Session, pages: int, per_page: int) -> List[str]:
    ids: List[str] = []
    for page in range(1, pages + 1):
        params = {
            "vs_currency": VS_CURRENCY,
            "order": "market_cap_desc",
            "per_page": per_page,
            "page": page,
            "price_change_percentage": "24h",
        }
        markets = fetch_json(session, "/coins/markets", params=params)
        ids.extend(coin["id"] for coin in markets if coin.get("id"))
    return ids


def fetch_known_categories(session: requests.Session) -> Set[str]:
    payload = fetch_json(session, "/coins/categories")
    return {item["name"] for item in payload if item.get("name")}


def fetch_coin_record(
    session: requests.Session,
    coin_id: str,
    valid_categories: Optional[Set[str]],
) -> Optional[Dict[str, Any]]:
    try:
        payload = fetch_json(session, f"/coins/{coin_id}", params=DETAIL_PARAMS)
    except requests.HTTPError:
        return None

    market_cap = payload.get("market_data", {}).get("market_cap", {}).get(VS_CURRENCY)
    if market_cap is None:
        return None

    coin_categories = payload.get("categories") or []
    if valid_categories:
        coin_categories = [cat for cat in coin_categories if cat in valid_categories]

    return {
        "name": payload.get("name"),
        "symbol": payload.get("symbol"),
        "market_cap": market_cap,
        "category": " | ".join(coin_categories),
    }


def dedupe_preserve_order(iterable: Sequence[str]) -> List[str]:
    seen: Set[str] = set()
    ordered: List[str] = []
    for item in iterable:
        if item and item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def build_candidate_list(session: requests.Session) -> List[str]:
    trending = fetch_trending_ids(session)
    top_movers = fetch_top_mover_ids(session)
    top_markets = fetch_market_ids(session, MARKET_PAGES, MARKET_PAGE_SIZE)
    combined: List[str] = []
    combined.extend(trending)
    combined.extend(top_movers)
    combined.extend(top_markets)
    return dedupe_preserve_order(combined)


def main() -> None:
    session = requests.Session()
    session.headers.update(HEADERS)

    global_snapshot = fetch_global_snapshot(session)
    print(
        "Global snapshot:",
        f"{global_snapshot.get('active_cryptocurrencies', 0)} active coins,",
        f"{global_snapshot.get('markets', 0)} markets.",
    )

    candidate_ids = build_candidate_list(session)
    print(f"Collecting details for {len(candidate_ids)} coins...")
    known_categories = fetch_known_categories(session)

    rows: List[Dict[str, Any]] = []
    for idx, coin_id in enumerate(candidate_ids, start=1):
        record = fetch_coin_record(session, coin_id, known_categories)
        if record:
            rows.append(record)
        time.sleep(RATE_LIMIT_DELAY)
        if idx % 25 == 0:
            print(f"Processed {idx} coins...")

    if not rows:
        raise RuntimeError("No rows were fetched from CoinGecko.")

    df = pd.DataFrame(rows, columns=["name", "symbol", "market_cap", "category"])
    df = df.dropna(subset=["market_cap"])
    df = df.sort_values("market_cap", ascending=False)

    output_path = Path("data/processed/crypto.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} records to {output_path}")
    print(df.head(10))


if __name__ == "__main__":
    main()
