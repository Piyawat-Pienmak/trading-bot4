#!/usr/bin/env python3
"""Simple public IP watcher that emails when your address changes."""

from __future__ import annotations

import argparse
import json
import os
import smtplib
import ssl
import sys
import time
from email.message import EmailMessage
from pathlib import Path
from urllib.request import Request, urlopen

from dotenv import load_dotenv

load_dotenv()


def fetch_public_ip(endpoint: str) -> str:
    req = Request(endpoint, headers={"User-Agent": "ip-watch/1.0"})
    with urlopen(req, timeout=10) as resp:  # nosec B310
        return resp.read().decode("utf-8").strip()


def load_last_ip(state_file: Path) -> str | None:
    if not state_file.exists():
        return None
    try:
        payload = json.loads(state_file.read_text())
        return payload.get("ip")
    except (json.JSONDecodeError, OSError):
        return None


def save_ip(state_file: Path, ip: str) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps({"ip": ip, "ts": time.time()}))


def send_email(
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    to_email: str,
    new_ip: str,
    old_ip: str | None,
) -> None:
    subject = f"[IP Watch] New public IP: {new_ip}"
    body_lines = [
        f"Detected a new public IP address: {new_ip}",
        f"Previous IP: {old_ip or 'unknown'}",
        "",
        "Update your Binance Futures API whitelist with the new address.",
    ]
    message = EmailMessage()
    message["Subject"] = subject
    message["From"] = smtp_user
    message["To"] = to_email
    message.set_content("\n".join(body_lines))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context) as server:
        server.login(smtp_user, smtp_password)
        server.send_message(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notify when your public IP changes.")
    parser.add_argument(
        "--interval",
        type=int,
        default=0,
        help="Seconds between checks; 0 runs once and exits.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Send an email even if the IP has not changed.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    state_file = Path(os.getenv("IP_WATCH_STATE_FILE", "data/state/ip_watch.json"))
    endpoint = os.getenv("IP_WATCH_ENDPOINT", "https://api.ipify.org")

    alert_email = os.getenv("IP_ALERT_EMAIL")
    smtp_user = os.getenv("IP_SMTP_USER", alert_email)
    smtp_password = os.getenv("IP_SMTP_PASSWORD")
    smtp_host = os.getenv("IP_SMTP_HOST", "smtp.gmail.com")
    smtp_port = int(os.getenv("IP_SMTP_PORT", "465"))

    if not all([alert_email, smtp_user, smtp_password]):
        print("Missing one of IP_ALERT_EMAIL/IP_SMTP_USER/IP_SMTP_PASSWORD env vars.", file=sys.stderr)
        return 1

    def run_check() -> None:
        current_ip = fetch_public_ip(endpoint)
        last_ip = load_last_ip(state_file)
        if args.force or current_ip != last_ip:
            send_email(
                smtp_host,
                smtp_port,
                smtp_user,
                smtp_password,
                alert_email,
                current_ip,
                last_ip,
            )
            save_ip(state_file, current_ip)
            print(f"IP changed to {current_ip}. Email sent.")
        else:
            print(f"IP unchanged ({current_ip}). No email.")

    if args.interval > 0:
        while True:
            run_check()
            time.sleep(args.interval)
    else:
        run_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
