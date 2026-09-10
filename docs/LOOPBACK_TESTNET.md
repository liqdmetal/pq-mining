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
- two loopback HTTP nodes and block broadcast.

Both nodes bind to `127.0.0.1`. There is no peer discovery, wallet, payout,
external network, production parameter profile, or GPU allocation.

## Run

From `C:/Users/officeone/pq-mining`:

```text
python tools/run_loopback_testnet.py --blocks 3 --max-nonces 128 --matched-bits 8
```

The launcher uses toy parameters:

```text
n_columns=32
syndrome_bits=32
w_max=4
matched_bits=8
```

Those parameters are for a quick local chain only. They are not comparable to
the research profile (`n=512`, `w_max=32`, `t=164`) and must not be used as a
security result.

## Verification

```text
python -m pytest tests/test_testnet.py -q
python -m pytest tests/ -q
```

Passing tests demonstrate the exercised implementation paths. They do not
prove consensus safety, network safety, PQ security, ASIC resistance, or
production readiness.
