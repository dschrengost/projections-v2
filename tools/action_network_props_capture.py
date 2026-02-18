#!/usr/bin/env python3
"""
Capture Action Network API requests for player props.

This script uses Playwright to load a game page and capture all network
requests to identify how props data is fetched.

Usage:
    uv run python tools/action_network_props_capture.py

Requirements:
    uv add playwright
    uv run playwright install chromium
"""

import asyncio
import json
from datetime import datetime
from playwright.async_api import async_playwright


# Example game URLs from different dates
GAME_URLS = [
    # Oct 2025 (current season)
    "https://www.actionnetwork.com/nba-game/bucks-raptors-score-odds-october-24-2025/261653#props",
    # Jan 2025
    "https://www.actionnetwork.com/nba-game/76ers-kings-score-odds-january-1-2025/233089#props",
]


async def capture_requests(url: str) -> dict:
    """Load a game page and capture all API requests."""

    captured_requests = []
    captured_responses = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = await context.new_page()

        # Capture all network requests
        async def handle_request(request):
            url = request.url
            # Filter for API calls
            if any(domain in url for domain in [
                "api.actionnetwork.com",
                "bam.actionnetwork.com",
                "actionnetwork.com/api",
            ]):
                captured_requests.append({
                    "url": url,
                    "method": request.method,
                    "headers": dict(request.headers),
                    "post_data": request.post_data,
                })

        async def handle_response(response):
            url = response.url
            # Filter for API calls
            if any(domain in url for domain in [
                "api.actionnetwork.com",
                "bam.actionnetwork.com",
                "actionnetwork.com/api",
            ]):
                try:
                    body = await response.text()
                    # Try to parse as JSON
                    try:
                        body = json.loads(body)
                    except json.JSONDecodeError:
                        pass
                except Exception as e:
                    body = f"<error reading body: {e}>"

                captured_responses.append({
                    "url": url,
                    "status": response.status,
                    "headers": dict(response.headers),
                    "body": body,
                })

        page.on("request", handle_request)
        page.on("response", handle_response)

        print(f"\n{'='*80}")
        print(f"Loading: {url}")
        print('='*80)

        # Navigate and wait for network idle
        await page.goto(url, wait_until="networkidle", timeout=30000)

        # Give extra time for dynamic content
        await asyncio.sleep(3)

        # Try clicking on props tab if it exists
        try:
            props_tab = page.locator("text=Props").first
            if await props_tab.is_visible():
                await props_tab.click()
                await asyncio.sleep(2)
        except Exception:
            pass

        # Also try expanding prop categories
        try:
            points_tab = page.locator("text=Points").first
            if await points_tab.is_visible():
                await points_tab.click()
                await asyncio.sleep(2)
        except Exception:
            pass

        await browser.close()

    return {
        "url": url,
        "requests": captured_requests,
        "responses": captured_responses,
    }


async def main():
    print("Action Network Props API Capture")
    print("=" * 80)

    all_results = []

    for url in GAME_URLS:
        result = await capture_requests(url)
        all_results.append(result)

        print(f"\nCaptured {len(result['requests'])} API requests:")
        for req in result['requests']:
            print(f"  [{req['method']}] {req['url'][:100]}...")

        print(f"\nCaptured {len(result['responses'])} API responses:")
        for resp in result['responses']:
            body_preview = str(resp['body'])[:200] if resp['body'] else "<empty>"
            print(f"  [{resp['status']}] {resp['url'][:80]}...")
            print(f"       Body preview: {body_preview}...")

    # Save full results to file
    output_file = f"action_network_capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nFull results saved to: {output_file}")

    # Print summary of unique endpoints found
    print("\n" + "=" * 80)
    print("UNIQUE API ENDPOINTS FOUND:")
    print("=" * 80)

    unique_endpoints = set()
    for result in all_results:
        for req in result['requests']:
            # Extract base endpoint (remove query params)
            endpoint = req['url'].split('?')[0]
            unique_endpoints.add(endpoint)

    for endpoint in sorted(unique_endpoints):
        print(f"  {endpoint}")

    # Print any responses that look like prop data
    print("\n" + "=" * 80)
    print("RESPONSES CONTAINING 'prop' OR 'odds':")
    print("=" * 80)

    for result in all_results:
        for resp in result['responses']:
            url_lower = resp['url'].lower()
            body_str = str(resp['body']).lower()
            if 'prop' in url_lower or 'prop' in body_str or 'odds' in body_str:
                print(f"\n[{resp['status']}] {resp['url']}")
                print(f"Headers: {json.dumps(dict(resp['headers']), indent=2)}")
                if isinstance(resp['body'], dict):
                    print(f"Body (formatted):\n{json.dumps(resp['body'], indent=2)[:2000]}")
                else:
                    print(f"Body: {str(resp['body'])[:2000]}")


if __name__ == "__main__":
    asyncio.run(main())
