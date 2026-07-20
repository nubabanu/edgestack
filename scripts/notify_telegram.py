"""Telegram push notifications for the tranche watcher.

One-time setup (5 minutes, free):
  1. In Telegram, message @BotFather -> /newbot -> pick a name; copy the bot token.
  2. Send any message to your new bot (this opens the chat).
  3. Open https://api.telegram.org/bot<TOKEN>/getUpdates in a browser and copy
     "chat":{"id": <number>} from the response.
  4. Write both into artifacts/telegram_config.json (gitignored):
       {"bot_token": "123456:ABC...", "chat_id": 123456789}
  5. Verify: .venv/Scripts/python scripts/notify_telegram.py --test "hello"

send() is a graceful no-op (with a printed WARN) when the config is missing or
the API call fails — notifications must never break the nightly chain.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "artifacts" / "telegram_config.json"


def send(text: str) -> bool:
    """Send a message to the configured chat; True on success."""
    if not CONFIG_PATH.exists():
        print(f"WARN telegram: no config at {CONFIG_PATH} — skipping push")
        return False
    try:
        cfg = json.loads(CONFIG_PATH.read_text())
        resp = requests.post(
            f"https://api.telegram.org/bot{cfg['bot_token']}/sendMessage",
            json={"chat_id": cfg["chat_id"], "text": text[:4000]},
            timeout=15,
        )
        resp.raise_for_status()
        return bool(resp.json().get("ok"))
    except Exception as exc:
        print(f"WARN telegram send failed: {exc}")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test", default="EdgeStack Telegram test message")
    args = parser.parse_args()
    ok = send(args.test)
    print("sent" if ok else "NOT sent (see warnings above)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
