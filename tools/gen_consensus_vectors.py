"""Regenerate the fixed 512/256/32 corpus; writes only inside this repository.

Run: python tools/gen_consensus_vectors.py
No mining, network, or random search is required. Pin changes require review.
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pqmine.verifier import (  # noqa: E402
    PoWParams, Share, ShareError, decode_pq_witness, derive_columns,
    derive_target, encode_pq_witness, verify_share,
)


def build_corpus() -> dict:
    params = PoWParams()
    derivation = []
    for name, header, nonce in [
        ("zero-nonce", b"a", 0), ("binary", bytes.fromhex("00ff1080"), 1),
        ("nonce-u32", b"c", 2**32), ("nonce-u63", b"d", 2**63),
        ("nonce-u64max", b"e", 2**64 - 1),
        ("long-header", bytes(range(256)), 123456789),
    ]:
        columns = derive_columns(header, params, nonce)
        derivation.append({
            "id": name, "header_hex": header.hex(), "nonce": nonce,
            "target_hex": derive_target(header, params, nonce).to_bytes(32, "big").hex(),
            "columns": [{"index": j, "value_hex": columns[j].to_bytes(32, "big").hex()}
                        for j in (0, 1, 63, 64, 255, 256, 511)],
        })

    verify = []

    def add(name, header, nonce, support, bits):
        try:
            math_expected = verify_share(Share(header, nonce, support),
                                         PoWParams(matched_bits=bits)).valid
        except ShareError:
            math_expected = False
        try:
            witness = encode_pq_witness(support).hex()
            canonical = True
        except ShareError:
            # Preserve intentionally invalid order/range/count in raw wire form.
            witness = (b"CHITINPQ" + bytes([len(support)]) + b"".join(
                j.to_bytes(2, "little") for j in support)).hex()
            canonical = False
        verify.append({
            "id": name, "header_hex": header.hex(), "nonce": nonce,
            "matched_bits": bits, "support": list(support),
            "witness_hex": witness, "canonical": canonical,
            "math_expected": math_expected,
            "wire_expected": math_expected and canonical,
        })

    old = json.loads((ROOT / "tests" / "golden_vectors.json").read_text())
    for v in old["vectors"]:
        if (v["family"] == "verify" and v["expect_valid"]
                and v["n_columns"] == 512 and v["syndrome_bits"] == 256):
            header = (bytes.fromhex(v["header"]) if v["header_encoding"] == "hex"
                      else v["header"].encode("ascii"))
            support = tuple(v["support"])
            add(v["name"], header, v["nonce"], support, v["matched_bits"])
            add(v["name"] + "-permutation", header, v["nonce"],
                tuple(reversed(support)), v["matched_bits"])
            add(v["name"] + "-changed-nonce", header, v["nonce"] ^ 1,
                support, v["matched_bits"])
            add(v["name"] + "-changed-header", header + b"!", v["nonce"],
                support, v["matched_bits"])
    header = b"canonical-wire-v1"
    # At t=1 we can deterministically obtain the cap-weight case in <=512 tries.
    support = next(tuple(range(31)) + (j,) for j in range(31, 512)
                   if verify_share(Share(header, 0, tuple(range(31)) + (j,)),
                                   PoWParams(matched_bits=1)).valid)
    add("weight-cap-valid", header, 0, support, 1)
    for name, indices in [
        ("empty", ()), ("duplicate", (1, 1)), ("descending", (511, 0)),
        ("index-512", (512,)), ("index-u16max", (65535,)),
        ("overweight", tuple(range(33))),
    ]:
        add(name, header, 0, indices, 1)
    for bits in (0, 1, 2, 8, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
                 164, 165, 191, 192, 193, 255, 256, 257):
        add(f"prefix-boundary-{bits}", header, 2**64 - 1, (0, 63, 64, 511), bits)
    add("empty-header", b"", 0, (0,), 1)

    canonical = encode_pq_witness((0, 1, 255, 256, 511))
    wire_bytes = [
        ("valid", canonical), ("valid-weight-cap", encode_pq_witness(support)),
        ("empty", b""), ("magic-only", b"CHITINPQ"),
        ("zero-count", b"CHITINPQ\0"), ("wrong-domain", b"CHITIN01" + canonical[8:]),
        ("truncated", canonical[:-1]), ("trailing-byte", canonical + b"\0"),
        ("count-mismatch", canonical[:8] + b"\x04" + canonical[9:]),
        ("count-over-cap", b"CHITINPQ\x21" + b"\0\0" * 33),
        ("descending", b"CHITINPQ\x02\x01\0\0\0"),
        ("duplicate", b"CHITINPQ\x02\x01\0\x01\0"),
        ("index-512", b"CHITINPQ\x01\0\x02"),
        ("index-u16max", b"CHITINPQ\x01\xff\xff"),
    ]
    wire = []
    for name, encoded in wire_bytes:
        try:
            decoded = list(decode_pq_witness(encoded))
        except ShareError:
            decoded = None
        wire.append({"id": name, "witness_hex": encoded.hex(),
                     "expected_support": decoded, "canonical": decoded is not None})
    return {
        "schema_version": 1, "spec": "chitin-pow-v2-fixed-profile-corpus",
        "parameters": {"n_columns": 512, "syndrome_bits": 256, "w_max": 32},
        "instance_encoding": "u32be(header_length)||header||u64be(nonce)",
        "wire_encoding": "CHITINPQ||u8(weight)||ascending-u16le-indices",
        "notes": "Math support order is irrelevant; wire order must be strictly ascending. "
                 "Header bytes are opaque and independent of the chain header serializer. "
                 "False parameter-boundary verdicts include invalid matched_bits.",
        "derivation": derivation, "verify": verify, "wire": wire,
    }


if __name__ == "__main__":
    output = ROOT / "tests" / "consensus_vectors_v1.json"
    output.write_text(json.dumps(build_corpus(), indent=2) + "\n", encoding="utf-8")
    print(output)
