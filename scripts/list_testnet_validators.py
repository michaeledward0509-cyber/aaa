#!/usr/bin/env python3
"""
List validators on the Natix testnet (netuid 323) from the public chain.
Uses wss://test.finney.opentensor.ai:443. This shows on-chain validator info only;
you cannot see other validators' process logs from here.

Run from repo root:
  poetry run python scripts/list_testnet_validators.py

Or source miner.env and use its endpoint:
  source miner.env && poetry run python scripts/list_testnet_validators.py
"""
import sys
import os

# Add repo root so we can import natix
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    try:
        import bittensor as bt
    except ImportError:
        print("ERROR: bittensor not installed. Run: poetry install", file=sys.stderr)
        sys.exit(1)

    # Use project config so Subtensor is created correctly
    from natix.base.miner import BaseMinerNeuron
    from natix.utils.config import add_all_args

    parser = add_all_args(BaseMinerNeuron)
    # Defaults for testnet; can override via env (e.g. from miner.env)
    netuid = int(os.environ.get("NETUID", "323"))
    endpoint = os.environ.get("SUBTENSOR_CHAIN_ENDPOINT", "wss://test.finney.opentensor.ai:443")
    network = os.environ.get("SUBTENSOR_NETWORK", "test")
    wallet_name = os.environ.get("WALLET_NAME", "default")
    wallet_hotkey = os.environ.get("WALLET_HOTKEY", "default")

    argv = [
        "list_testnet_validators.py",
        "--netuid", str(netuid),
        "--subtensor.network", network,
        "--subtensor.chain_endpoint", endpoint,
        "--wallet.name", wallet_name,
        "--wallet.hotkey", wallet_hotkey,
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        config = bt.Config(parser)
    finally:
        sys.argv = old_argv

    try:
        subtensor = bt.Subtensor(config=config)
    except Exception as e:
        print("Could not create Subtensor:", e, file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {endpoint} netuid={netuid} ...")
    try:
        metagraph = subtensor.metagraph(netuid)
    except Exception as e:
        print("Failed to load metagraph:", e, file=sys.stderr)
        print("(Chain may be up but metagraph query failed, e.g. storage compatibility.)", file=sys.stderr)
        sys.exit(1)

    n = metagraph.n
    if hasattr(n, "item"):
        n = n.item()
    print(f"Metagraph loaded: {n} neurons\n")

    # Validators: UIDs with validator_permit
    validators = []
    for uid in range(n):
        try:
            if metagraph.validator_permit[uid]:
                ax = metagraph.axons[uid]
                stake = float(metagraph.S[uid])
                emission = float(metagraph.E[uid]) if hasattr(metagraph, "E") else 0.0
                last_update = int(metagraph.last_update[uid]) if hasattr(metagraph, "last_update") else 0
                hotkey = str(metagraph.hotkeys[uid])
                ip = getattr(ax, "ip", "") or ""
                port = getattr(ax, "port", 0) or 0
                validators.append({
                    "uid": uid,
                    "hotkey": hotkey[:16] + "..." if len(hotkey) > 16 else hotkey,
                    "stake": stake,
                    "emission": emission,
                    "last_update": last_update,
                    "ip": ip,
                    "port": port,
                })
        except Exception as e:
            continue

    if not validators:
        print("No validators with validator_permit found on this subnet.")
        return

    print("Validators (on-chain; netuid={}):".format(netuid))
    print("-" * 80)
    for v in validators:
        print(f"  UID {v['uid']:3d}  stake={v['stake']:.4f}  emission={v['emission']:.6f}  "
              f"last_update={v['last_update']}  {v['ip']}:{v['port']}  {v['hotkey']}")
    print("-" * 80)
    print(f"Total: {len(validators)} validator(s).")
    print("\nNote: This is on-chain data only. To see a validator's process logs, run that validator locally and use pm2 logs or the terminal.")


if __name__ == "__main__":
    main()
