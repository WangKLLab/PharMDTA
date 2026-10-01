"""Regression coverage for padded attention under mixed precision."""
import unittest
from contextlib import nullcontext

import torch

from model.model import AlignedBidirectionalCoAttention


class AttentionTests(unittest.TestCase):
    def test_masked_softmax_precision_and_backward(self):
        valid = torch.tensor([[[True, True, False], [False, False, False]]])
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for dim in (-1, -2):
                with self.subTest(dtype=dtype, dim=dim):
                    scores = torch.tensor(
                        [[[[2., -1., 4.], [3., 5., -2.]]]],
                        dtype=dtype, requires_grad=True,
                    )
                    weights = AlignedBidirectionalCoAttention._masked_softmax(
                        scores, valid, dim=dim,
                    )
                    self.assertEqual(weights.dtype, torch.float32)
                    self.assertTrue(torch.isfinite(weights).all())
                    self.assertTrue((weights.masked_select(~valid.unsqueeze(1)) == 0).all())
                    expected_sums = valid.any(dim=dim).unsqueeze(1).float()
                    torch.testing.assert_close(weights.sum(dim=dim), expected_sums)
                    reference = torch.softmax(
                        scores.detach().float().masked_fill(~valid.unsqueeze(1), -1e30),
                        dim=dim,
                    ).masked_fill(~valid.unsqueeze(1), 0.)
                    torch.testing.assert_close(weights, reference)
                    (weights * torch.arange(6).reshape_as(weights)).sum().backward()
                    self.assertTrue(torch.isfinite(scores.grad).all())
                    self.assertTrue((scores.grad.masked_select(~valid.unsqueeze(1)) == 0).all())

    def check_layer(self, device, amp):
        torch.manual_seed(42)
        layer = AlignedBidirectionalCoAttention(dim=16, heads=4, dropout=0.).to(device)
        atoms = torch.randn(2, 3, 16, device=device, requires_grad=True)
        residues = torch.randn(2, 5, 16, device=device, requires_grad=True)
        atom_valid = torch.tensor([[True, True, False], [True, True, True]], device=device)
        residue_valid = torch.tensor(
            [[True, True, True, False, False], [True, True, True, True, True]], device=device,
        )
        context = torch.autocast(device.type, dtype=torch.float16) if amp else nullcontext()
        with context:
            outputs = layer(atoms, atom_valid, residues, residue_valid)
            loss = sum(output.square().sum() for output in outputs)
        for output in outputs:
            self.assertTrue(torch.isfinite(output).all())
        pair = outputs[2]
        self.assertEqual(pair.dtype, torch.float32)
        torch.testing.assert_close(pair.sum(dim=(1, 2)), torch.ones(2, device=device))
        pair_valid = atom_valid.unsqueeze(2) & residue_valid.unsqueeze(1)
        self.assertTrue((pair.masked_select(~pair_valid) == 0).all())
        loss.backward()
        for tensor in (atoms, residues):
            self.assertTrue(torch.isfinite(tensor.grad).all())
        for parameter in layer.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_padded_layer_forward_backward(self):
        self.check_layer(torch.device('cpu'), amp=False)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for FP16 autocast')
    def test_cuda_fp16_padded_layer_forward_backward(self):
        self.check_layer(torch.device('cuda'), amp=True)


if __name__ == '__main__':
    unittest.main()
