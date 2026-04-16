"""Entry point for the CodeKnowledge MCP server."""

from __future__ import annotations

import argparse
import logging

from .app import create_mcp_server


def main() -> None:
    parser = argparse.ArgumentParser(description="CodeKnowledge MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http"],
        default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8767, help="HTTP port (default: 8767)")
    parser.add_argument(
        "--allow",
        action="append",
        default=[],
        metavar="PATTERN",
        help="Glob pattern for allowed project paths (repeatable). "
        "Example: --allow '/home/james/dev/**'",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    logging.getLogger("codeknowledge").setLevel(
        logging.DEBUG if args.verbose else logging.INFO
    )

    mcp = create_mcp_server(allow=args.allow)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        mcp.run(transport="streamable-http", host=args.host, port=args.port)


if __name__ == "__main__":
    main()
