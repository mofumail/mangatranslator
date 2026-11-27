"""The studio: a local server that agents drive over MCP to translate manga, with a web UI to
watch them work.

    uv run python studio.py                      # http://127.0.0.1:7870 (UI), /mcp (MCP)
    claude mcp add --transport http mangatl-studio http://127.0.0.1:7870/mcp

Projects (a folder of pages each) are kept in the workspace (default studio_data/): the project
file, the rendered pages (out/), chapter notes, worker reports and the activity log.
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=7870)
    p.add_argument("--workspace", default=str(ROOT / "studio_data"), help="where projects are kept")
    p.add_argument(
        "--set", action="append", default=[], metavar="PATH=VALUE", help="settings override for drafts"
    )
    p.add_argument("--offline", action="store_true", help="never download models")
    args = p.parse_args()

    os.environ.update(
        HF_HUB_DISABLE_TELEMETRY="1",
        YOLO_OFFLINE="1",
        DO_NOT_TRACK="1",
        PYTHONUNBUFFERED="1",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",  # before CUDA starts
    )
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "24")
    if args.offline:
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    sys.path.insert(0, str(ROOT))

    import uvicorn

    from mangatl.common.log import init_file_logging
    from mangatl.engine import Engine
    from mangatl.settings import Settings
    from mangatl.studio.web import build_app
    from mangatl.studio.workspace import Studio
    from mangatl.typeset.fontconfig import start_fontconfig_quietly

    init_file_logging()
    start_fontconfig_quietly()
    engine = Engine.from_settings(Settings().resolve_secrets(), offline=args.offline)
    studio = Studio(engine, Path(args.workspace), args.set)
    print(f"[studio] UI  http://{args.host}:{args.port}/", flush=True)
    print(f"[studio] MCP http://{args.host}:{args.port}/mcp", flush=True)
    print(f"[studio] workspace {studio.workspace}", flush=True)
    try:
        uvicorn.run(build_app(studio), host=args.host, port=args.port, log_level="warning")
    finally:
        engine.close()


if __name__ == "__main__":
    main()
