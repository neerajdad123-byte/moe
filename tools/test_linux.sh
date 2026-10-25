#!/bin/bash
# The Windows gates, on Linux (tested in WSL2 Ubuntu 24.04). Prints PASS/FAIL per check and a summary; exit 1 on any FAIL.
#   tools/test_linux.sh cpu            ./build.sh, then the checks that need no GPU and no model (what CI runs)
#   tools/test_linux.sh gpu [MODEL]    + test_kernels, --check (default and --load --mtp), serve + its tests, /reset,
#                                      bench, and the low-RAM path (MOEX_LOW_RAM=1); MODEL defaults to $MOEX_MODEL
# CHECK_SETS: the --check sets (default ref/eval1). PY: a python with tokenizers, jinja2, openai, anthropic.
# NOBUILD=1 skips ./build.sh. Every model run has a time limit, so a hang (e.g. on exit) is a FAIL, not a stuck gate.
set -u
cd "$(dirname "$0")/.."
WHAT=${1:-cpu}
MODEL=${2:-${MOEX_MODEL:-$HOME/models/qwen36}}
PY=${PY:-python3}
SETS=${CHECK_SETS:-ref/eval1}
fails=0; summary=""
result() { if [ "$2" = 0 ]; then summary+="PASS  $1"$'\n'; else summary+="FAIL  $1"$'\n'; fails=$((fails + 1)); fi; echo "-- $1: $([ "$2" = 0 ] && echo PASS || echo FAIL)"; }
step() { local name=$1; shift; echo "== $name"; "$@"; result "$name" $?; }
with_env() { local kv=$1; shift; export "$kv"; "$@"; local rc=$?; unset "${kv%%=*}"; return $rc; }  # VAR=1 for a function
# rc, then the output must contain a string
expect() { local want_rc=$1 want=$2; shift 2; local out rc; out=$("$@" 2>&1); rc=$?; [ $rc -eq "$want_rc" ] && [[ "$out" == *"$want"* ]] || { echo "rc $rc: $out" | tail -5; return 1; }; }
TMP=$(mktemp -d); trap 'rm -rf "$TMP"; [ -n "${SRV:-}" ] && kill $SRV 2>/dev/null' EXIT

[ -n "${NOBUILD:-}" ] || step "build (./build.sh)" ./build.sh
step "moex --version" expect 0 "moex " build/moex --version
step "help" expect 0 "moex download" build/moex help
step "no arguments: rc 2" expect 2 "" build/moex
step "unknown command: rc 2" expect 2 "" build/moex nosuchcommand
step "missing model: rc 1" expect 1 "No model at" build/moex chat --model "$TMP/no_model_here"
step "GGUF as --model: rc 1" expect 1 "moex import" build/moex chat --model some.gguf
step "import without a file: rc 2" expect 2 "" build/moex import
step "import of a missing file: rc 1" expect 1 "cannot open" build/moex import "$TMP/no_such.gguf"
step "download and model-path helpers (test_download)" build/test_download
# Linux's `moex install` (Windows: test_userenv): a link in ~/.local/bin to the real binary, idempotent, no PATH edits
install_check() {
    mkdir -p "$TMP/home" && HOME="$TMP/home" build/moex install && HOME="$TMP/home" build/moex install &&
    [ "$(readlink -f "$TMP/home/.local/bin/moex")" = "$(readlink -f build/moex)" ] && "$TMP/home/.local/bin/moex" --version
}
step "moex install (link in ~/.local/bin, twice)" install_check
step "GGUF reader (test_gguf)" build/test_gguf
step "speculative sampling (test_spec)" build/test_spec
if $PY -c "import torch, numpy" 2> /dev/null; then  # tools/quant_dense.py writes the reference with torch
    step "dense int4/int6 decoder (test_dense)" bash -c "$PY tools/quant_dense.py --selftest > /dev/null && build/test_dense"
else summary+="SKIP  dense int4/int6 decoder (test_dense: no torch in $PY)"$'\n'; fi
TOK=$MODEL; [ -f "$TOK/tokenizer.json" ] || TOK=tok
if [ ! -f "$TOK/tokenizer.json" ]; then mkdir -p tok && curl -sSfL -o tok/tokenizer.json https://huggingface.co/Kreuzzelg/qwen36-35b-a3b-colibri-i4-gs64/resolve/main/tokenizer.json; fi
step "tokenizer vs Hugging Face (check_tokenizer.py)" env QWEN36_MODEL="$TOK" $PY tools/check_tokenizer.py
step "chat template, offline (check_template.py --offline)" env QWEN36_MODEL="$TOK" $PY tools/check_template.py --offline "$TOK"

if [ "$WHAT" = gpu ]; then
    step "kernels (test_kernels)" with_env QWEN36_MODEL="$MODEL" build/test_kernels
    check_set() {  # set, options...
        local out; out=$(timeout 1800 build/moex --check "$@" --model "$MODEL" 2>&1); local rc=$?
        echo "$out" | grep -a "\[check\]\|\[mtp\]\|\[ram\]\|\[load\]\|error\|failed"
        [ $rc -eq 0 ] && [[ "$out" == *"-> PASS"* ]] || { echo "rc $rc, last lines:"; echo "$out" | tail -8; return 1; }
    }
    for s in $SETS; do
        step "--check $s" check_set "$s"
        step "--check $s --load --mtp" check_set "$s" --load --mtp
    done
    step "--check ref/eval1, low-RAM path (MOEX_LOW_RAM=1)" with_env MOEX_LOW_RAM=1 check_set ref/eval1

    serve_tests() {
        MOEX_DUMP_PROMPT=build/prompt_ids.txt build/moex serve --model "$MODEL" > "$TMP/serve.log" 2>&1 & SRV=$!
        local t=0; until curl -sf http://127.0.0.1:8765/health > /dev/null; do
            sleep 2; t=$((t + 2)); kill -0 $SRV 2>/dev/null && [ $t -lt 1200 ] || { tail -20 "$TMP/serve.log"; return 1; }
        done
        grep -a "load\|slots\|\[ram\]" "$TMP/serve.log" | head -20
        local rc=0
        $PY tools/test_server.py > "$TMP/ts.log" 2>&1 || rc=1; cat "$TMP/ts.log"; grep -q FAIL "$TMP/ts.log" && rc=1
        for t in test_tools test_security test_robust; do echo "-- $t"; $PY tools/$t.py || rc=1; done
        echo "-- check_template (vs the running server)"; QWEN36_MODEL="$MODEL" $PY tools/check_template.py || rc=1
        kill $SRV; wait $SRV 2>/dev/null; SRV=
        return $rc
    }
    step "moex serve: test_server, test_tools, test_security, test_robust, check_template" serve_tests
    step "/reset in chat (test_reset.py)" timeout 3600 $PY tools/test_reset.py build/moex "$MODEL"
    bench() { timeout 1800 build/moex bench --model "$MODEL" "$@"; }  # exits by itself: a hang on exit is a FAIL
    step "moex bench" bench
    step "moex bench, low-RAM path (MOEX_LOW_RAM=1)" with_env MOEX_LOW_RAM=1 bench
fi

echo; echo "== summary ($WHAT)"; printf "%s" "$summary"
[ $fails -eq 0 ] && echo "ALL PASS" || echo "$fails FAILED"
exit $((fails > 0))
