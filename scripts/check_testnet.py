#!/usr/bin/env python3
"""
Check that the public Bittensor testnet (test.finney.opentensor.ai) is reachable.
Run from repo root: poetry run python scripts/check_testnet.py
"""
import socket
import sys

ENDPOINT_HOST = "test.finney.opentensor.ai"
ENDPOINT_PORT = 443
WSS_URL = "wss://test.finney.opentensor.ai:443"
NETUID = 323


def check_tcp():
    """Verify TCP connection to testnet host:port."""
    print(f"1. TCP connection to {ENDPOINT_HOST}:{ENDPOINT_PORT} ...")
    try:
        s = socket.create_connection((ENDPOINT_HOST, ENDPOINT_PORT), timeout=10)
        s.close()
        print("   OK - Port is reachable.")
        return True
    except OSError as e:
        print(f"   FAILED - {e}")
        return False


def check_websocket():
    """Verify WebSocket endpoint accepts a connection."""
    print(f"2. WebSocket to {WSS_URL} ...")
    try:
        import websockets
    except ImportError:
        print("   Skipped (install websockets for WSS check: pip install websockets)")
        return True

    async def connect():
        async with websockets.connect(WSS_URL, close_timeout=5) as ws:
            pass

    try:
        import asyncio
        asyncio.run(connect())
        print("   OK - WebSocket connected.")
        return True
    except Exception as e:
        print(f"   FAILED - {e}")
        return False


def main():
    print("Checking public testnet: test.finney.opentensor.ai")
    print("(Natix testnet netuid:", NETUID, ")")
    print()

    ok = check_tcp()
    if not ok:
        print("\nPublic testnet is not reachable from this machine.")
        sys.exit(1)

    check_websocket()
    print()
    print("Public testnet at test.finney.opentensor.ai is reachable.")


if __name__ == "__main__":
    main()
