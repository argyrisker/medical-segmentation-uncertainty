#!/usr/bin/env bash
# Shows how far scripts/run_all.sh has got. Safe to run at any time; it only reads files.
#   bash scripts/status.sh
cd "$(dirname "$0")/.."
TOTAL="${TOTAL:-21}"   # training runs with the default settings (PRIORITY=1)

if pgrep -f "scripts/run_all.sh" > /dev/null; then running=1; else running=0; fi
done_flag=0
[[ -f run_all.log ]] && grep -q "All stages finished" run_all.log && done_flag=1

if (( done_flag )); then echo "Job: FINISHED"
elif (( running )); then echo "Job: RUNNING"
else echo "Job: NOT RUNNING"
fi

if [[ -f run_all.log ]]; then
  echo
  echo "Stages started:"
  grep "########" run_all.log | sed 's/^######## /  /'
fi

n=$(ls outputs/checkpoints/*/summary.json 2>/dev/null | wc -l)
echo
echo "Finished training runs: $n of $TOTAL"

latest=$(ls -t outputs/logs/*.log 2>/dev/null | head -1)
if (( running )) && [[ -n "$latest" ]]; then
  echo "Current run: $(basename "$latest" .log)"
  last=$(tr '\r' '\n' < "$latest" | grep -E "^Epoch [0-9]+ \|" | tail -1)
  [[ -n "$last" ]] && echo "  ${last:0:110}"
fi

# Rough time left: average minutes per finished run (smoke test excluded) x runs left.
python3 - "$TOTAL" <<'PY' 2>/dev/null
import glob, json, sys
total = int(sys.argv[1])
done = [json.load(open(p)) for p in glob.glob("outputs/checkpoints/*/summary.json")]
mins = [d["elapsed_min"] for d in done if not d["run_name"].startswith("smoke")]
left = total - len(done)
if mins and left > 0:
    avg = sum(mins) / len(mins)
    print(f"Average {avg:.0f} min per run, about {avg * left / 60:.1f} hours of training left")
PY

if (( !running && !done_flag )) && [[ -f run_all.log ]]; then
  echo
  echo "It stopped before finishing. Last lines of run_all.log:"
  tail -n 15 run_all.log
  echo
  echo "Fix the problem, then start the same command again; finished runs are skipped."
fi
if (( done_flag )); then
  echo
  echo "Results: outputs/tables/ (tables), reports/figures/ (plots)"
fi
