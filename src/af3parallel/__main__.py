"""Allow ``python -m af3parallel`` invocation."""

from af3parallel.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
