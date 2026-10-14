#!/usr/bin/env bash
# Run or resume a reproduce.py run without supervision.
#
#   [PY=.venv/bin/python] bash scripts/run_unattended.sh RUN_DIR [--jobs 5 --procs 16 --threads 12]
#
# - At most one supervisor per run: the lock RUN/supervisor.lock is held until every child has exited.
# - It runs the frozen copy RUN/code/reproduce.py, so editing the package never changes a run in progress.
# - --retry-interrupted: stages cut short by a stopped container are retried (partial work set aside, not deleted).
# - RUN/supervisor.json says "running", "finished" or "stopped" (stages failed after their retries: see
#   RUN/last_invocation.json and RUN/logs/<stage>.log). A run left "running" by a killed container is resumed by
#   the next call.
set -uo pipefail
RUN=$(cd "$1" && pwd)
shift
PY=${PY:-python3}  # the interpreter with requirements.txt installed
exec 9>"$RUN/supervisor.lock"
flock -n 9 || { echo "A supervisor is already running for $RUN"; exit 0; }
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
printf '{"state": "running", "pid": %d, "since": "%s"}\n' $$ "$(now)" > "$RUN/supervisor.json"
echo "=== $(now) supervisor $$: reproduce.py run $*" >> "$RUN/logs/runner.log"
"$PY" "$RUN/code/reproduce.py" run --run-dir "$RUN" --retry-interrupted "$@" >> "$RUN/logs/runner.log" 2>&1
code=$?
state=$([ "$code" -eq 0 ] && echo finished || echo stopped)
printf '{"state": "%s", "exit_code": %d, "at": "%s"}\n' "$state" "$code" "$(now)" > "$RUN/supervisor.json"
echo "=== $(now) supervisor $$: $state (exit $code)" >> "$RUN/logs/runner.log"
