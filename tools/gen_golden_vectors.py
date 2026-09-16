"""Generate cross-language golden vectors for Chitin-PoW v2.

Writes tests/golden_vectors.json (the Python-side pin) and copies it to
../obscura/crates/obscura-chitin-pow/tests/golden_vectors.json (the Rust-side
fixture). Deterministic: every random choice is seeded from the vector name,
so regenerating on any machine produces byte-identical fixtures.

Run from the pq-mining repo root:

    python tools/gen_golden_vectors.py
"""

from __future__ import annotations

import json
import pathlib
import random
import shutil
import sys

from pqmine.verifier import (
    PoWParams,
    Share,
    derive_columns,
    derive_target,
    verify_share,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT_PY = ROOT / "tests" / "golden_vectors.json"
OUT_RS = ROOT.parent / "obscura" / "crates" / "obscura-chitin-pow" / "tests" / "golden_vectors.json"

ERROR_NAMES = {
    "support is not an iterable sequence": "MalformedSupport",
    "column index is not an integer": "MalformedSupport",
    "duplicate column (F_2 cancellation attack)": "DuplicateColumn",
    "column index out of range": "ColumnOutOfRange",
    "empty support": "EmptySupport",
    "syndrome mismatch on matched bits": "SyndromeMismatch",
}


def weight_error_name(reason: str) -> str:
    return "WeightExceeded" if reason.startswith("weight ") else ERROR_NAMES[reason]


def header_bytes(spec: dict) -> bytes:
    if spec["header_encoding"] == "hex":
        return bytes.fromhex(spec["header"])
    return spec["header"].encode("ascii")


def find_valid_support(header: bytes, nonce: int, params: PoWParams,
                       want_weight: int | None, max_tries: int = 2_500_000):
    """Random-support search for a genuine share. Seeded per-vector."""
    cols = derive_columns(header, params, nonce)
    target = derive_target(header, params, nonce)
    total = 64 * ((params.syndrome_bits + 63) // 64)
    mask = ((1 << params.matched_bits) - 1) << (total - params.matched_bits)
    rng = random.Random(f"chitin-golden:{header.hex()}:{nonce}")
    upper = min(params.w_max, params.n_columns)
    for _ in range(max_tries):
        weight = want_weight if want_weight is not None else rng.randint(1, upper)
        support = tuple(sorted(rng.sample(range(params.n_columns), weight)))
        acc = 0
        for j in support:
            acc ^= cols[j]
        if (acc & mask) == (target & mask):
            return support
    return None


def int_to_hex(value: int, params: PoWParams) -> str:
    words = (params.syndrome_bits + 63) // 64
    return value.to_bytes(8 * words, "big").hex()


def build_vector(spec: dict, supports: dict) -> dict:
    header = header_bytes(spec)
    params = PoWParams(
        n_columns=spec["n_columns"],
        syndrome_bits=spec["syndrome_bits"],
        w_max=spec["w_max"],
        matched_bits=spec["matched_bits"],
    )
    nonce = spec["nonce"]
    vector = {
        "name": spec["name"],
        "family": spec["family"],
        "header": spec["header"],
        "header_encoding": spec["header_encoding"],
        "nonce": nonce,
        "n_columns": params.n_columns,
        "syndrome_bits": params.syndrome_bits,
        "w_max": params.w_max,
        "matched_bits": params.matched_bits,
        "columns": [int_to_hex(col, params) for col in derive_columns(header, params, nonce)],
        "target": int_to_hex(derive_target(header, params, nonce), params),
    }
    if spec["family"] == "derive":
        return vector

    kind = spec["kind"]
    if kind == "mined":
        want_weight = spec.get("want_weight")
        support = find_valid_support(header, nonce, params, want_weight)
        if support is None:
            raise SystemExit(f"{spec['name']}: no valid support found within budget")
        if want_weight is not None and len(support) != want_weight:
            raise SystemExit(f"{spec['name']}: wanted weight {want_weight}, got {len(support)}")
        expect_valid, expect_error = True, None
    elif kind == "empty":
        support = ()
        expect_valid, expect_error = False, "EmptySupport"
    elif kind == "out-of-range":
        support = (params.n_columns,)
        expect_valid, expect_error = False, "ColumnOutOfRange"
    elif kind == "out-of-range-u32-max":
        support = (2**32 - 1,)
        expect_valid, expect_error = False, "ColumnOutOfRange"
    elif kind == "weight-over":
        support = tuple(range(params.w_max + 1))
        expect_valid, expect_error = False, "WeightExceeded"
    elif kind in ("duplicate-from", "mismatch-from"):
        base = supports[spec["from"]]
        if kind == "duplicate-from":
            support = (base[0], base[0])
            expect_valid, expect_error = False, "DuplicateColumn"
        else:
            j = base[0] ^ 1
            if j == base[0]:
                j = (base[0] + 1) % params.n_columns
            candidate = tuple(sorted({j, *base[1:]}))
            support = candidate
            expect_valid, expect_error = False, "SyndromeMismatch"
    else:
        raise SystemExit(f"{spec['name']}: unknown kind {kind!r}")

    verdict = verify_share(Share(header, nonce, support), params)
    if verdict.valid != expect_valid:
        raise SystemExit(
            f"{spec['name']}: expected valid={expect_valid}, verdict={verdict}"
        )
    if not expect_valid:
        actual_name = weight_error_name(verdict.reason) if verdict.reason.startswith("weight ") \
            else ERROR_NAMES[verdict.reason]
        if actual_name != expect_error:
            raise SystemExit(
                f"{spec['name']}: expected error {expect_error}, got {actual_name} ({verdict.reason})"
            )
    vector["support"] = list(support)
    vector["expect_valid"] = expect_valid
    vector["expect_error"] = expect_error
    supports[spec["name"]] = tuple(support)
    return vector


# (name, family, header, header_encoding, nonce, n, sb, w_max, mb[, kind, from])
SPECS = [
    # ---- derivation family: pin SHAKE domains, lengths, byte order ----
    dict(name="derive-baseline", family="derive", header="chitin-golden-header-v1",
         header_encoding="ascii", nonce=7, n_columns=32, syndrome_bits=32,
         w_max=4, matched_bits=8),
    dict(name="derive-full-profile-a", family="derive", header="a",
         header_encoding="ascii", nonce=0, n_columns=512, syndrome_bits=256,
         w_max=32, matched_bits=164),
    dict(name="derive-full-profile-b", family="derive", header="b",
         header_encoding="ascii", nonce=1, n_columns=512, syndrome_bits=256,
         w_max=32, matched_bits=164),
    dict(name="derive-full-profile-hi-nonce", family="derive", header="c",
         header_encoding="ascii", nonce=2**32, n_columns=512, syndrome_bits=256,
         w_max=32, matched_bits=164),
    dict(name="derive-max-syndrome", family="derive", header="d",
         header_encoding="ascii", nonce=2**63, n_columns=512, syndrome_bits=512,
         w_max=64, matched_bits=512),
    dict(name="derive-cross-word-odd", family="derive", header="e",
         header_encoding="ascii", nonce=123456789, n_columns=128,
         syndrome_bits=64, w_max=4, matched_bits=33),
    dict(name="derive-tiny-u64max-nonce", family="derive", header="f",
         header_encoding="ascii", nonce=2**64 - 1, n_columns=16,
         syndrome_bits=8, w_max=2, matched_bits=8),
    dict(name="derive-one-bit-syndrome", family="derive", header="6700",
         header_encoding="hex", nonce=5, n_columns=512, syndrome_bits=1,
         w_max=1, matched_bits=1),
    dict(name="derive-long-header", family="derive", header="h" * 40,
         header_encoding="ascii", nonce=2**40, n_columns=256,
         syndrome_bits=192, w_max=16, matched_bits=97),
    dict(name="derive-u64max-2words", family="derive", header="f",
         header_encoding="ascii", nonce=2**64 - 1, n_columns=128,
         syndrome_bits=128, w_max=8, matched_bits=100),
    # ---- verify family: valid shares across the edge cases ----
    dict(name="valid-toy", family="verify", kind="mined", header="verifier-test-header",
         header_encoding="ascii", nonce=0, n_columns=32, syndrome_bits=32,
         w_max=4, matched_bits=8),
    dict(name="valid-one-bit", family="verify", kind="mined", header="verifier-test-header",
         header_encoding="ascii", nonce=1, n_columns=32, syndrome_bits=32,
         w_max=4, matched_bits=1),
    dict(name="valid-full-profile", family="verify", kind="mined", header="fprofile",
         header_encoding="ascii", nonce=2, n_columns=512, syndrome_bits=256,
         w_max=32, matched_bits=17),
    dict(name="valid-cross-word-remainder", family="verify", kind="mined", header="cw",
         header_encoding="ascii", nonce=4, n_columns=128, syndrome_bits=128,
         w_max=8, matched_bits=20),
    dict(name="valid-giant-weight", family="verify", kind="mined", header="gw",
         header_encoding="ascii", nonce=5, n_columns=520, syndrome_bits=8,
         w_max=512, matched_bits=1),
    dict(name="valid-wmax-1", family="verify", kind="mined", header="w1",
         header_encoding="ascii", nonce=6, n_columns=16, syndrome_bits=8,
         w_max=1, matched_bits=1),
    dict(name="valid-exact-weight-cap", family="verify", kind="mined", header="cap",
         header_encoding="ascii", nonce=7, n_columns=32, syndrome_bits=32,
         w_max=4, matched_bits=8, want_weight=4),
    dict(name="valid-u64-max-nonce", family="verify", kind="mined", header="um",
         header_encoding="ascii", nonce=2**64 - 1, n_columns=64, syndrome_bits=8,
         w_max=4, matched_bits=1),
    dict(name="valid-nonce-2power63", family="verify", kind="mined", header="m",
         header_encoding="ascii", nonce=2**63, n_columns=64, syndrome_bits=8,
         w_max=4, matched_bits=2),
    dict(name="valid-binary-header", family="verify", kind="mined", header="00ff1080",
         header_encoding="hex", nonce=8, n_columns=32, syndrome_bits=32,
         w_max=4, matched_bits=2),
    dict(name="valid-sb-64", family="verify", kind="mined", header="s64",
         header_encoding="ascii", nonce=9, n_columns=128, syndrome_bits=64,
         w_max=4, matched_bits=8),
    dict(name="valid-full-profile-2", family="verify", kind="mined", header="g",
         header_encoding="ascii", nonce=2**16, n_columns=512, syndrome_bits=256,
         w_max=32, matched_bits=20),
    dict(name="valid-syndrome-eq-matched", family="verify", kind="mined", header="eq",
         header_encoding="ascii", nonce=11, n_columns=16, syndrome_bits=8,
         w_max=4, matched_bits=8),
    dict(name="valid-toy-second", family="verify", kind="mined", header="verifier-test-header",
         header_encoding="ascii", nonce=10, n_columns=32, syndrome_bits=32,
         w_max=4, matched_bits=6),
    # ---- verify family: forgeries the verifier must reject ----
    dict(name="invalid-mismatch-flip", family="verify", kind="mismatch-from",
         **{"from": "valid-toy"}, header="verifier-test-header", header_encoding="ascii",
         nonce=0, n_columns=32, syndrome_bits=32, w_max=4, matched_bits=8),
    dict(name="invalid-empty", family="verify", kind="empty",
         header="verifier-test-header", header_encoding="ascii", nonce=0,
         n_columns=32, syndrome_bits=32, w_max=4, matched_bits=8),
    dict(name="invalid-duplicate", family="verify", kind="duplicate-from",
         **{"from": "valid-toy"}, header="verifier-test-header", header_encoding="ascii",
         nonce=0, n_columns=32, syndrome_bits=32, w_max=4, matched_bits=8),
    dict(name="invalid-out-of-range", family="verify", kind="out-of-range",
         header="verifier-test-header", header_encoding="ascii", nonce=0,
         n_columns=32, syndrome_bits=32, w_max=4, matched_bits=8),
    dict(name="invalid-out-of-range-u32max", family="verify", kind="out-of-range-u32-max",
         header="verifier-test-header", header_encoding="ascii", nonce=0,
         n_columns=32, syndrome_bits=32, w_max=4, matched_bits=8),
    dict(name="invalid-weight-over-cap", family="verify", kind="weight-over",
         header="verifier-test-header", header_encoding="ascii", nonce=0,
         n_columns=32, syndrome_bits=32, w_max=4, matched_bits=8),
]


def main() -> None:
    supports: dict[str, tuple] = {}
    vectors = [build_vector(spec, supports) for spec in SPECS]
    derive_count = sum(1 for v in vectors if v["family"] == "derive")
    verify_count = len(vectors) - derive_count
    document = {
        "spec": "chitin-pow-v2-golden-vectors",
        "generator": "tools/gen_golden_vectors.py (deterministic, seeded)",
        "counts": {"total": len(vectors), "derive": derive_count, "verify": verify_count},
        "vectors": vectors,
    }
    text = json.dumps(document, indent=2, sort_keys=False) + "\n"
    OUT_PY.write_text(text, encoding="utf-8")
    OUT_PY.parent.mkdir(parents=True, exist_ok=True)
    OUT_RS.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OUT_PY, OUT_RS)
    print(f"wrote {len(vectors)} vectors ({derive_count} derive, {verify_count} verify)")
    print(f"  python pin: {OUT_PY}")
    print(f"  rust fixture: {OUT_RS}")


if __name__ == "__main__":
    sys.exit(main())
