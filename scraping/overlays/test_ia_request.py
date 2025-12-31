#!/usr/bin/env python3
"""
Test script to verify we can make requests to Internet Archive.
"""

import requests

def main():
    # Test 1: Basic request to Internet Archive
    print("Test 1: Basic request to archive.org...")
    resp = requests.get("https://archive.org", timeout=10)
    print(f"   Status: {resp.status_code}")
    print(f"   Content length: {len(resp.text)} bytes")

    # Test 2: CDX API - query for openrouter.ai snapshots
    print("\nTest 2: CDX API query for OpenRouter snapshots...")
    cdx_url = "https://web.archive.org/cdx/search/cdx"
    params = {
        "url": "openrouter.ai/api/v1/models",
        "output": "json",
        "limit": 10,
    }
    resp = requests.get(cdx_url, params=params, timeout=30)
    print(f"   Status: {resp.status_code}")

    if resp.status_code == 200:
        data = resp.json()
        print(f"   Rows returned: {len(data)}")
        if len(data) > 1:
            # First row is header
            print(f"   Header: {data[0]}")
            print(f"   First snapshot: {data[1]}")
            # Parse timestamp
            if len(data) > 1:
                timestamp = data[1][1]  # timestamp is second column
                print(f"   Timestamp: {timestamp}")
                print(f"   Wayback URL: https://web.archive.org/web/{timestamp}/https://openrouter.ai/api/v1/models")
    else:
        print(f"   Error: {resp.text[:500]}")


if __name__ == "__main__":
    main()
