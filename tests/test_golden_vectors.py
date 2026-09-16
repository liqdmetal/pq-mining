"""Self-check: every golden vector must satisfy the CURRENT Python verifier.

This pins tests/golden_vectors.json to pqmine.verifier so the fixture cannot
silently drift from the reference implementation. The same file is consumed by
the Rust integration tests in
../obscura/crates/obscura-chitin-pow/tests/python_golden_vectors.rs; regenerate
both with `PYTHONPATH=. python tools/gen_golden_vectors.py` if the consensus
rule ever changes deliberately.
"""

import json
import pathlib
import unittest

from pqmine.verifier import PoWParams, Share, derive_columns, derive_target, verify_share

VECTORS = json.loads(
    (pathlib.Path(__file__).resolve().parent / "golden_vectors.json").read_text(
        encoding="utf-8"
    )
)


def header_bytes(vector: dict) -> bytes:
    if vector["header_encoding"] == "hex":
        return bytes.fromhex(vector["header"])
    return vector["header"].encode("ascii")


def params_of(vector: dict) -> PoWParams:
    return PoWParams(
        n_columns=vector["n_columns"],
        syndrome_bits=vector["syndrome_bits"],
        w_max=vector["w_max"],
        matched_bits=vector["matched_bits"],
    )


def words_of(vector: dict) -> int:
    return (vector["syndrome_bits"] + 63) // 64


def to_int(hex_word: str) -> int:
    return int.from_bytes(bytes.fromhex(hex_word), "big")


def mask_of(vector: dict) -> int:
    total = 64 * words_of(vector)
    return ((1 << vector["matched_bits"]) - 1) << (total - vector["matched_bits"])


class GoldenVectorSelfCheck(unittest.TestCase):
    def test_fixture_shape_and_counts(self):
        self.assertEqual(VECTORS["spec"], "chitin-pow-v2-golden-vectors")
        counts = VECTORS["counts"]
        self.assertEqual(counts["total"], len(VECTORS["vectors"]))
        self.assertGreaterEqual(counts["total"], 20)
        for vector in VECTORS["vectors"]:
            self.assertIn(vector["family"], ("derive", "verify"))
            self.assertTrue(vector["name"])
            self.assertTrue(header_bytes(vector), "header must be non-empty")

    def test_derive_vectors_match_current_derivation(self):
        checked = 0
        for vector in VECTORS["vectors"]:
            if vector["family"] != "derive":
                continue
            header = header_bytes(vector)
            params = params_of(vector)
            nonce = vector["nonce"]
            columns = derive_columns(header, params, nonce)
            target = derive_target(header, params, nonce)
            words = words_of(vector)
            self.assertEqual(
                [c.to_bytes(8 * words, "big").hex() for c in columns],
                vector["columns"],
                vector["name"],
            )
            self.assertEqual(
                target.to_bytes(8 * words, "big").hex(), vector["target"], vector["name"]
            )
            checked += 1
        self.assertEqual(checked, VECTORS["counts"]["derive"])

    def test_verify_vectors_match_current_verifier(self):
        valid = 0
        for vector in VECTORS["vectors"]:
            if vector["family"] != "verify":
                continue
            share = Share(header_bytes(vector), vector["nonce"], tuple(vector["support"]))
            verdict = verify_share(share, params_of(vector))
            self.assertEqual(
                verdict.valid,
                vector["expect_valid"],
                f"{vector['name']}: {verdict.reason}",
            )
            if vector["expect_valid"]:
                valid += 1
                self.assertEqual(verdict.reason, "valid")
                self.assertEqual(verdict.matched, vector["matched_bits"])
            else:
                self.assertNotEqual(verdict.reason, "valid", vector["name"])
        self.assertEqual(valid, sum(
            1 for v in VECTORS["vectors"]
            if v["family"] == "verify" and v["expect_valid"]
        ))

    def test_syndrome_relation_holds_on_valid_vectors(self):
        """Independent recheck: XOR of the pinned columns equals the pinned
        target on the matched-bit prefix for every expect_valid vector."""
        for vector in VECTORS["vectors"]:
            if vector["family"] != "verify" or not vector["expect_valid"]:
                continue
            acc = 0
            for index, column_hex in enumerate(vector["columns"]):
                if index in vector["support"]:
                    acc ^= to_int(column_hex)
            mask = mask_of(vector)
            self.assertEqual(
                acc & mask,
                to_int(vector["target"]) & mask,
                vector["name"],
            )


if __name__ == "__main__":
    unittest.main()
