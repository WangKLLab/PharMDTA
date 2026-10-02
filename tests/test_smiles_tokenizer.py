"""Regression coverage for directional bonds and atom-token alignment."""
import unittest

from model.smiles_tokenizer import (
    encode_smiles_regex,
    load_smiles_vocabulary,
    regex_smiles_token_atom_indices,
    regex_smiles_tokens,
)


class SMILESTokenizerTests(unittest.TestCase):
    def test_directional_bonds_preserve_smiles(self):
        for smiles in (r'F\C=C\F', r'F/C=C\F', 'F/C=C/F'):
            with self.subTest(smiles=smiles):
                tokens = regex_smiles_tokens(smiles)
                self.assertEqual(''.join(tokens), smiles)
                self.assertEqual(tokens[1], smiles[1])
                self.assertEqual(tokens[5], smiles[5])

    def test_encoded_stereoisomers_remain_distinct(self):
        vocabulary = load_smiles_vocabulary()
        sequences = []
        for smiles in (r'F/C=C\F', 'F/C=C/F'):
            ids, mask, count, unknown = encode_smiles_regex(
                smiles, vocabulary=vocabulary, max_length=16,
            )
            self.assertEqual(unknown, 0)
            self.assertEqual(count, 7)
            self.assertEqual(int(mask.sum()), 9)
            reverse = {index: token for token, index in vocabulary.items()}
            decoded = ''.join(reverse[int(index)] for index in ids[1:8])
            self.assertEqual(decoded, smiles)
            sequences.append(ids.tolist())
        self.assertNotEqual(*sequences)

    def test_directional_bonds_do_not_introduce_atoms(self):
        tokens, atoms = regex_smiles_token_atom_indices(r'F\C=C\F')
        self.assertEqual(tokens, ['F', '\\', 'C', '=', 'C', '\\', 'F'])
        self.assertEqual(atoms, [0, None, 1, None, 2, None, 3])


if __name__ == '__main__':
    unittest.main()
