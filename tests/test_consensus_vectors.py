"""Fixed wire corpus, old-verifier differential oracle, and resource boundaries."""
import hashlib
import json
import pathlib
import random
import unittest
from unittest.mock import patch

from pqmine.verifier import (
    PoWParams, Share, ShareError, decode_pq_witness, derive_columns,
    derive_selected_columns, derive_target, encode_pq_witness, syndrome_mask,
    verify_pq_witness, verify_share,
)
from tools.gen_consensus_vectors import build_corpus

ROOT = pathlib.Path(__file__).resolve().parent
CORPUS = json.loads((ROOT / "consensus_vectors_v1.json").read_text())


def old_full_table_verdict(header, nonce, support, params):
    """Independent pre-optimization full-table acceptance rule."""
    if (not isinstance(header, bytes) or not header or type(nonce) is not int
            or not 0 <= nonce < 2**64 or not isinstance(support, (list, tuple))
            or not 1 <= len(support) <= params.w_max
            or any(type(j) is not int or not 0 <= j < params.n_columns for j in support)
            or len(set(support)) != len(support)):
        return False
    prefix = len(header).to_bytes(4, "big") + header + nonce.to_bytes(8, "big")
    size = 8 * ((params.syndrome_bits + 63) // 64)
    columns = [int.from_bytes(hashlib.shake_256(
        b"CHITIN-POW-H-V2" + prefix + j.to_bytes(4, "big")).digest(size), "big")
        for j in range(params.n_columns)]
    target = int.from_bytes(hashlib.shake_256(b"CHITIN-POW-S-V2" + prefix).digest(size), "big")
    acc = 0
    for j in support:
        acc ^= columns[j]
    mask = ((1 << params.matched_bits) - 1) << (size * 8 - params.matched_bits)
    return (acc & mask) == (target & mask)


class ConsensusCorpusTests(unittest.TestCase):
    def test_corpus_is_reproducible_and_fixed_profile(self):
        self.assertEqual(CORPUS, build_corpus())
        self.assertEqual(CORPUS["schema_version"], 1)
        self.assertEqual(CORPUS["parameters"], {"n_columns": 512, "syndrome_bits": 256, "w_max": 32})

    def test_derivation(self):
        params = PoWParams()
        for v in CORPUS["derivation"]:
            with self.subTest(v["id"]):
                header = bytes.fromhex(v["header_hex"])
                self.assertEqual(derive_target(header, params, v["nonce"]), int(v["target_hex"], 16))
                indices = tuple(c["index"] for c in v["columns"])
                actual = derive_selected_columns(header, params, indices, v["nonce"])
                self.assertEqual(actual, tuple(int(c["value_hex"], 16) for c in v["columns"]))

    def test_math_wire_and_old_verifier_are_distinct_and_consistent(self):
        for v in CORPUS["verify"]:
            with self.subTest(v["id"]):
                header = bytes.fromhex(v["header_hex"])
                wire = bytes.fromhex(v["witness_hex"])
                self.assertEqual(verify_pq_witness(header, v["nonce"], wire, v["matched_bits"]).valid,
                                 v["wire_expected"])
                try:
                    params = PoWParams(matched_bits=v["matched_bits"])
                except ShareError:
                    self.assertFalse(v["math_expected"])
                    continue
                share = Share(header, v["nonce"], tuple(v["support"]))
                self.assertEqual(verify_share(share, params).valid, v["math_expected"])
                self.assertEqual(old_full_table_verdict(header, v["nonce"], share.support, params),
                                 v["math_expected"])

    def test_codec(self):
        for v in CORPUS["wire"]:
            with self.subTest(v["id"]):
                wire = bytes.fromhex(v["witness_hex"])
                if v["canonical"]:
                    self.assertEqual(list(decode_pq_witness(wire)), v["expected_support"])
                    self.assertEqual(encode_pq_witness(v["expected_support"]), wire)
                else:
                    with self.assertRaises(ShareError):
                        decode_pq_witness(wire)

    def test_legacy_corpus_parity(self):
        legacy = json.loads((ROOT / "golden_vectors.json").read_text())
        for v in legacy["vectors"]:
            if v["family"] != "verify":
                continue
            params = PoWParams(**{key: v[key] for key in
                                  ("n_columns", "syndrome_bits", "w_max", "matched_bits")})
            header = bytes.fromhex(v["header"]) if v["header_encoding"] == "hex" else v["header"].encode()
            self.assertEqual(old_full_table_verdict(header, v["nonce"], v["support"], params),
                             verify_share(Share(header, v["nonce"], v["support"]), params).valid)

    def test_randomized_parity_including_non_word_aligned_syndromes(self):
        rng = random.Random(0xC4171)
        for _ in range(300):
            bits = rng.choice((1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 256))
            params = PoWParams(512, bits, 32, rng.randint(1, bits))
            header, nonce = rng.randbytes(rng.randint(1, 80)), rng.getrandbits(64)
            support = tuple(rng.sample(range(512), rng.randint(1, 32)))
            self.assertEqual(verify_share(Share(header, nonce, support), params).valid,
                             old_full_table_verdict(header, nonce, support, params))

    def test_verifier_hashes_only_support_and_target(self):
        with patch("pqmine.verifier.hashlib.shake_256", wraps=hashlib.shake_256) as shake:
            verify_share(Share(b"bounded", 0, (1, 17, 511)), PoWParams())
            self.assertEqual(shake.call_count, 4)
        with patch("pqmine.verifier.hashlib.shake_256") as shake:
            self.assertFalse(verify_pq_witness(b"x", 0, b"CHITINPQ\x21" + b"\0" * 66, 1).valid)
            shake.assert_not_called()

    def test_malformed_codec_and_parameters(self):
        for value in (None, "CHITINPQ", bytearray(b"CHITINPQ"), b"x" * 100000):
            with self.assertRaises(ShareError):
                decode_pq_witness(value)
        for support in ((True,), (1.0,), (-1,), (512,), (2, 1), (1, 1)):
            with self.assertRaises(ShareError):
                encode_pq_witness(support)
        for key in ("n_columns", "syndrome_bits", "w_max", "matched_bits"):
            for value in (True, 1.5, "1", None):
                with self.assertRaises(ShareError):
                    PoWParams(**{key: value})


if __name__ == "__main__":
    unittest.main()
