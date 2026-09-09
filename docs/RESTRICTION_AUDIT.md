# Restricted-support audit — decoupled selector

**Status:** toy-scale structural screen on the current v2 construction; not a
cryptanalysis or security certificate.

## What the audit actually tests

The audit now feeds the decoupled trace the **same nonce-bound parity-check
columns the v2 verifier derives for each (header, nonce) pair**. It does not
audit a stale header-only matrix. Each nonce therefore has its own committed
instance, and the screens below estimate whether the walk's reachable-support
structure is obviously biased or degenerate across nonces.

## Construction audited

The audit uses the same pure-Python `decoupled_trace.py` transition as the
reference `search_share.py`: mutable arena, `K` steps, selection threshold for
`target_weight`, duplicate-column rehashing, and the verifier's column count.
It no longer audits the obsolete all-steps walk.

## Reproduction

```text
cd C:/Users/officeone/pq-mining
python -m pytest tests/test_restriction_trace.py -q
```

Current toy run (nonce-derived columns):

```text
header                 v2-audit
n,k,w                  512,256,32
arena_bits,steps      12,400
audit failures         []
selected weight       36
parity-effective      36
weight ratio          1.0
column chi2/df        0.9147
selected samples      506
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
- **This does not establish that reachable-support decoding is as hard as
  generic bounded-weight SDP.** The walk remains a structured subset of
  supports, and a better structured attack may exist. The bounded MITM
  estimate deliberately ignores that restriction, so it may overstate
  hardness.
- The only published work figure remains the simple bounded two-list MITM
  estimate: **83.06-bit upper bound** at `t=164`; no certified security level
  is claimed.
