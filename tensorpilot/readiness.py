"""Container healthcheck: exit 0 only when the worker reports ready on /readyz."""
import os
import sys
import urllib.error
import urllib.request


def main():
    port = os.environ.get("TENSORPILOT_PORT", "8080")
    url = f"http://127.0.0.1:{port}/readyz"
    try:
        with urllib.request.urlopen(url, timeout=3) as response:
            sys.exit(0 if response.status == 200 else 1)
    except (urllib.error.URLError, OSError):
        sys.exit(1)


if __name__ == "__main__":
    main()
