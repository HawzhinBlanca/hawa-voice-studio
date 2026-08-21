#!/usr/bin/env python3
"""
Hawa Sorani Voice Studio - Unified Developer CLI.
Single entrypoint to launch FastAPI Control Plane, run test suites, or start Next.js Studio.
"""

import argparse
import os
import subprocess
import sys


def run_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = True):
    print("=" * 65)
    print(" Starting Hawa Sorani Voice Studio - FastAPI Control Plane ")
    print("=" * 65)
    os.environ["PYTHONPATH"] = "."
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "services.api.main:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    if reload:
        cmd.append("--reload")
    subprocess.run(cmd)


def run_tests():
    print("=" * 65)
    print(" Running Hawa Sorani Voice Studio Complete Test Suite ")
    print("=" * 65)
    os.environ["PYTHONPATH"] = "."
    cmd = [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"]
    res = subprocess.run(cmd)
    sys.exit(res.returncode)


def run_frontend():
    print("=" * 65)
    print(" Starting Next.js Studio Frontend on http://localhost:3000 ")
    print("=" * 65)
    web_dir = os.path.join(os.getcwd(), "apps", "web")
    cmd = ["npm", "run", "dev"]
    subprocess.run(cmd, cwd=web_dir, shell=True)


def main():
    parser = argparse.ArgumentParser(description="Hawa Sorani Voice Studio CLI")
    parser.add_argument("command", choices=["api", "test", "web", "all"], help="Command to execute")
    parser.add_argument("--port", type=int, default=8000, help="API Port")
    args = parser.parse_args()

    if args.command == "api":
        run_api(port=args.port)
    elif args.command == "test":
        run_tests()
    elif args.command == "web":
        run_frontend()
    elif args.command == "all":
        run_tests()


if __name__ == "__main__":
    main()
