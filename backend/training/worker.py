"""Process entry point for a training job.

Run as ``python -m backend.training.worker --experiment-id <id>``. The API
spawns this module; it does not train inside the request process.
"""

from __future__ import annotations

import argparse
import logging
import traceback

from backend.storage.db import init_db
from backend.storage.repository import update_experiment
from backend.training.loop import train_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-id", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    init_db()
    try:
        train_experiment(args.experiment_id)
    except Exception:
        logging.getLogger("lunarmind.worker").exception("training failed")
        update_experiment(args.experiment_id, status="failed", error=traceback.format_exc()[-4000:])
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
