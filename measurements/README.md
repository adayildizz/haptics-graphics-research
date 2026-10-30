# Lab measurements shared through git

Sessions written here with `--data-dir measurements/<date>_<name>` are meant to be
committed and pushed, traces included: `.gitignore` excludes `*.sqlite3`
everywhere else, but not under this folder. Everything else the experiment
writes still goes to `experiment/data/`, which git ignores.

Commit the whole session folder, including any `-wal` / `-shm` file next to a
trace: if the session did not close cleanly, part of the trace is still in them.

| Folder | What | Config |
|---|---|---|
| `2026-10-30_kalman/` | Render-mode check: predictive, position and kalman mixed within one session, felt breaks reported with the Down arrow | `experiment/configs/kalman_check.yaml` |
