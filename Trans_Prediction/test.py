# test.py

import torch
import unittest
from transformer_model import TransformerModel

class TestTransformerModel(unittest.TestCase):
    def setUp(self):
        self.seq_length = 10
        self.input_dim = 4
        self.output_dim = 4
        self.hidden_dim = 64
        self.nhead = 4
        self.num_layers = 2
        self.model = TransformerModel(self.input_dim, self.hidden_dim, self.output_dim, self.nhead, self.num_layers)

    def test_forward_pass_with_valid_input(self):
        src = torch.randn((1, self.seq_length, self.input_dim))
        tgt = src.clone()
        output = self.model(src, tgt)
        self.assertEqual(output.shape, (1, self.seq_length, self.output_dim))

    def test_forward_pass_with_mismatched_src_tgt_shapes(self):
        src = torch.randn((1, self.seq_length, self.input_dim))
        tgt = torch.randn((1, self.seq_length + 1, self.input_dim))
        with self.assertRaises(RuntimeError):
            self.model(src, tgt)

    def test_forward_pass_with_invalid_input_dim(self):
        src = torch.randn((1, self.seq_length, self.input_dim + 1))
        tgt = src.clone()
        with self.assertRaises(RuntimeError):
            self.model(src, tgt)

if __name__ == '__main__':
    unittest.main()