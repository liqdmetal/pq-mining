# Chitin-PoW loopback testnet PoC

**Status:** isolated research PoC. This is not DERO, does not modify DERO
consensus, and does not claim post-quantum security.

## What it exercises

- deterministic block/header encoding;
- nonce-bound `pqmine.verifier` consensus rule;
- bounded CPU reference partial-syndrome mining;
- genesis and sequential parent/height validation;
- duplicate-support, wrong-nonce, wrong-parent, and future-timestamp rejection;
- JSON block transport;
- two loopback HTTP nodes and block broadcast;
- profile IDs bound into genesis and every header;
- ancestry-bound timestamp-window difficulty control;
- median-time-past timestamp rule with an odd, profile-bound window;
- branch-aware chain selection and deterministic reorganization;
- persistence of all known branches with restart validation;
- `/sync` import for recovering a missed chain;
- malformed and oversized HTTP payload rejection.

Both nodes bind to `127.0.0.1`. There is no peer discovery, wallet, payout,
external network, or GPU allocation.

## Run the in-process smoke harness

From `C:/Users/officeone/pq-mining`:

```text
python tools/run_loopback_testnet.py --blocks 3 --max-nonces 128 --matched-bits 8
```

## Run the separate-process testnet

This starts two independent Python processes with separate state files. Alice
mines the toy profile; Bob reconciles over loopback HTTP. The parent process
owns cleanup and prints one machine-readable final report.

```text
python tools/run_loopback_network.py \
  --state-dir C:/temp/chitin-network \
  --run-for 10 \
  --mine-interval 0.2 \
  --max-nonces 128
```

The runner is deliberately bounded and loopback-only. It refuses non-loopback
peers. It is a process/lifecycle testnet, not a public or LAN network.

## Restart recovery

Run the demonstrated stop/restart/reconcile scenario:

```text
python tools/run_loopback_testnet.py --blocks 3 --max-nonces 128 --matched-bits 8 --restart-recovery
```

The command persists both nodes, stops Bob while Alice mines, restarts Bob from
its state file, imports Alice's chain through `/chain`, and verifies equal tip
and height.

## Persistent serve mode

Run one node for manual loopback inspection:

```text
python tools/run_loopback_testnet.py --serve --state-dir C:/temp/chitin-alice --port 49613
```

Inspect:

```text
http://127.0.0.1:49613/status
http://127.0.0.1:49613/chain
http://127.0.0.1:49613/graph
```

`/chain` returns the selected best branch. `/graph` returns every known valid
branch plus the deterministic selected tip; it is intended for local fork and
reorganization inspection.

`--peer` is restricted to `http://127.0.0.1:<port>` by the launcher. No
external bind or peer URL is accepted.

## Automatic peer reconciliation

The node API supports a bounded background sync loop:

```python
node.start(port=0, auto_sync_interval=0.5)
node.peers = [peer_url]
```

The interval cannot be set below 50 ms. Nodes accept at most eight normalized
loopback peers. Chain imports and `/chain` responses are capped at 64 KiB by
default; peer requests use a 100 ms timeout. Failed peers use exponential
backoff (0.5 seconds up to 30 seconds) rather than a hot retry loop.
`/status` reports `running`, `attempts`, `successes`, `backoff_skips`, and
`last_error` under `sync`. Shutdown stops and joins the sync worker before
closing the HTTP server.

The launcher uses toy parameters:

```text
n_columns=32
syndrome_bits=32
w_max=4
matched_bits=8
```

Those parameters are for a quick local chain only. They are not comparable to
the candidate profile (`n=512`, `w_max=60`, `t=256`) and must not be used as a
security result.

The candidate profile can be inspected without mining:

```text
python tools/run_loopback_testnet.py --profile candidate-128 --sync-only
```

Attempting to mine the candidate profile with the bounded reference solver is
refused deliberately. Its current screens are research estimates, not proof of
PQ security or evidence of practical mining feasibility.

## Verification

```text
python -m pytest tests/test_testnet.py -q
python -m pytest tests/test_process_testnet.py -q
python -m pytest tests/ -q
```

Passing tests demonstrate the exercised implementation paths. They do not
prove consensus safety, network safety, PQ security, ASIC resistance, or
production readiness.
