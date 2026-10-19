# API

All routes are under `/api/v1`.

- `GET /health`, `GET /meta`, `GET /overview`
- `POST /experiments` and `GET /experiments/{id}`
- `POST /experiments/{id}/start|pause|resume|stop`
- `GET /experiments/{id}/checkpoints` and `GET /experiments/{id}/analytics`
- `POST /sessions` then `WS /ws/sessions/{id}`
- `WS /ws/experiments/{id}` for training progress
- `GET /trajectories` and `GET /trajectories/{id}` plus `export.csv` and `export.json`
- `POST /comparisons` and `GET /comparisons/{group_id}`

Training `start` spawns `python -m backend.training.worker`. Pause and stop are flags the worker reads between PPO chunks. A pause request while no process is alive returns HTTP 409 rather than pretending to pause.

Live flights stream telemetry, the reward breakdown and the render geometry. Human input is a websocket message listing the keys that are currently down.

SQLite lives at `data/lunarmind.sqlite`. Checkpoints live under `models/checkpoints/<experiment>/`. Trajectory JSON lives under `data/trajectories/`.
