# Restricted-support attack accounting

**Status:** research analysis; not a cryptanalysis proof and not a certified
security level.

## Models kept separate

The verifier accepts any distinct support of weight `1..w_max`, regardless of
how it was found. Therefore two attack routes must be reported separately:

1. **Generic bounded-language MITM.** Enumerate arbitrary supports, split into
two lists, and match the `t`-bit syndrome prefix. This is the current
conservative attack upper bound.
2. **Walk-restricted nonce scan.** Run the decoupled selector once per nonce,
retain the support only when its weight is `<= w_max`, and model the syndrome
prefix as a random `t`-bit event. This is a conditional workload model for the
specific search path, not a verifier restriction.

The second route cannot raise security above the first: the verifier does not
require miners to use the walk.

## Reproduction

```text
cd C:/Users/officeone/pq-mining
python -c "from pqmine.codebased import SDPParameters; from pqmine.latency_bound import LatencyParameters; from pqmine.restricted_work import composite_attack; s=SDPParameters(n=512,k=256,weight=32); w=LatencyParameters(arena_bits=8,steps=400,write_density=1,cell_bytes=8); print(composite_attack(b'recommended-attack-audit',s,w,matched_bits=164,w_max=32,nonces=256,target_weight=32).summary())"
```

Measured run:

```text
generic bounded MITM upper bound: 83.060 bits
left/right list weights:           16 / 16
walk samples:                      256
walk admissible samples:           148
walk admissibility probability:    0.578125
conditional walk scan estimate:    173.434 bits
best modeled route:                generic-bounded-mitm
```

The walk estimate includes `log2(K)` serial step work per nonce and assumes
that, after conditioning on admissible weight, the nonce-bound syndrome prefix
is uniform. That assumption is unproved and is the subject of the structural
audit; it must not be presented as a security theorem.

## Multi-level bounded-language family

`pqmine/bdd_attack.py` evaluates a small, explicit family of one-, two-, and
four-partition shapes. It uses the same two-group batching equation as the
baseline after grouping intermediate lists, and sweeps every accepted support
weight `1..w_max`. This is a symbolic cost family, not a complete BDD/ISD
algorithm or an optimality result.

At the recommended parameters:

```text
classical two-list baseline: 83.059519 bits
best tested multi-level:    83.059519 bits
best tested weight:         32
best tested partition:      (256, 256)
improvement:                false
```

The tested multi-level family therefore does not lower the current 83.06-bit
attack upper bound. It also does not validate that bound as a security level:
other partitions, actual list-merging strategies, partial-syndrome ISD/BDD
methods, quantum attacks, and walk-structure attacks remain outside this model.

## Executable partial-ISD baseline

`pqmine/partial_isd.py` is an intentionally bounded CPU reference solver for
the same two-list relation. It uses combinations, so support entries are
always distinct; it matches only the verifier's top `matched_bits`; and it can
construct a real `Share` and pass it through `verify_share()`.

Evidence:

```text
toy exhaustive targets checked: 16
toy exact agreement with exhaustive accepted-support enumeration: true
recommended-profile candidate cap: 100,000
recommended-profile measured entries: 100,000 (16.610 bits)
recommended-profile share found before cap: false
```

The recommended-profile run is deliberately capped and therefore
**inconclusive**. It is a correctness/reference implementation, not a
feasibility result and not a security measurement. Removing the cap is not
permitted for the production-sized parameters on this workstation.

## Interpretation

- The walk's observed admissibility rate is about **57.8%** in this bounded
  sample, so the `w_max=32` cap rejects a substantial tail of its outputs.
- The simple walk scan is not the current best modeled attack at `t=164`.
- The generic MITM result remains an **83.06-bit attack upper bound on any
  security claim**, because the consensus verifier accepts arbitrary bounded
  supports.
- A better generic bounded-weight, partial-syndrome, quantum, or
  walk-structure attack may be faster. No certified security level is claimed.
