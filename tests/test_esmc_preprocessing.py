"""Verify that long-sequence inference preserves every residue in order."""
import unittest
import torch
from model.esmc_preprocessing import chunk_sequence, reassemble_chunk_states


class ESMCPreprocessingTests(unittest.TestCase):
    def test_nonoverlapping_chunks_cover_full_sequence(self):
        sequence = 'ACDEFGHIKLMNPQRSTVWY' * 120
        chunks = chunk_sequence(sequence, 2046)
        self.assertEqual([offset for offset, _ in chunks], [0, 2046])
        self.assertEqual(''.join(chunk for _, chunk in chunks), sequence)
        self.assertEqual([len(chunk) for _, chunk in chunks], [2046, 354])

    def test_reassembly_keeps_residue_order_and_outer_special_tokens(self):
        first = torch.tensor([[-1., -1.], [1., 1.], [2., 2.], [-2., -2.]])
        second = torch.tensor([[-3., -3.], [3., 3.], [-4., -4.]])
        result = reassemble_chunk_states([first, second], [2, 1])
        expected = torch.tensor([[-1., -1.], [1., 1.], [2., 2.], [3., 3.], [-4., -4.]])
        torch.testing.assert_close(result, expected)

    def test_reassembly_rejects_missing_residue_and_nonfinite_states(self):
        with self.assertRaises(ValueError):
            reassemble_chunk_states([torch.ones(3, 2)], [2])
        with self.assertRaises(ValueError):
            reassemble_chunk_states([torch.full((3, 2), float('nan'))], [1])


if __name__ == '__main__':
    unittest.main()
