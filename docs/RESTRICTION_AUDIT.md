# Restricted-support audit — decoupled selector

**Status:** toy-scale structural screen; not a cryptanalysis or security
certificate.

## Construction audited

The audit now uses the same pure-Python `decoupled_trace.py` transition as the
reference `search_share.py`: mutable arena, `K` steps, selection threshold for
`target_weight=32`, duplicate-column rehashing, and the verifier's column
count. It no longer audits the obsolete all-steps walk.

## Reproduction

```text
cd C:/Users/officeone/pq-mining
python -m pytest tests/test_restriction_trace.py -q
```

Current toy run:

```text
header                 actual-decoupled
n,k,w                  512,256,32
arena_bits,steps      12,400
audit failures         []
selected weight       26
parity-effective      26
weight ratio          1.0
column chi2/df        0.9185
syndrome samples      256
syndrome distinct     256
expected distinct     256.0
shared path suffix    0
```

## Interpretation

- The selector's deliberate sparse weight is no longer mislabeled as weight
  collapse: effective weight is compared with **selected weight**, not `K`.
- No gross column bias, syndrome concentration, or nonce path convergence was
  found at this sample size.
- This does **not** establish that reachable-support decoding is as hard as
  generic bounded-weight SDP. The walk remains a structured subset, and a
  better structured attack may exist.
- The current simple bounded MITM attack remains the only published work
  figure: **83.06 bits upper-bounded security** at `t=164`; no certified level
  is claimed.
