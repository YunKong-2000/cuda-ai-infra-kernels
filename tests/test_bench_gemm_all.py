"""CPU regressions for the full optimization ladder benchmark."""

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from unittest import TestCase, mock

import torch

from cuda_ai_kernels.gemm_stages import GEMM_STAGE_IMPLS

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestGemmAllBenchmark(TestCase):
    def setUp(self):
        self.bench = load_module("bench_all_test", ROOT / "benchmarks/bench_gemm_all.py")

    def test_full_ladder_uses_same_inputs_and_validates_before_timing(self):
        a, b = self.bench.make_inputs(128, 256, 16, device="cpu")
        self.assertEqual(a.stride(), (1, 128))
        self.assertTrue(b.is_contiguous())
        calls, timed = [], []
        expected = (a.double() @ b.double()).float()

        def fake_gemm(actual_a, actual_b, *, impl):
            self.assertIs(actual_a, a)
            self.assertIs(actual_b, b)
            calls.append(impl)
            return expected

        def timer(run, *, warmup, repeat):
            self.assertEqual(calls[:len(GEMM_STAGE_IMPLS)], list(GEMM_STAGE_IMPLS))
            self.assertEqual((warmup, repeat), (2, 3))
            run()
            timed.append(calls[-1])
            return {"mean_ms": 2.0 if calls[-1] == "cublas" else 1.0}

        with mock.patch.object(self.bench, "gemm", fake_gemm), \
             mock.patch.object(self.bench, "cuda_graph_benchmark", timer):
            rows = self.bench.benchmark_shape(a, b, warmup=2, repeat=3)
        self.assertEqual(timed, list(GEMM_STAGE_IMPLS))
        self.assertEqual(len(rows), 13)
        self.assertEqual(rows[0]["speedup_over_cublas"], 2)
        self.assertEqual(rows[-1]["speedup_over_cublas"], 1)
        self.assertAlmostEqual(rows[0]["tflops"], 2 * 128 * 256 * 16 / 1e9)

    def test_bad_result_prevents_any_timing(self):
        a, b = torch.ones(128, 16), torch.ones(16, 128)
        with mock.patch.object(self.bench, "gemm", return_value=torch.zeros(128, 128)), \
             mock.patch.object(self.bench, "cuda_graph_benchmark") as timer:
            with self.assertRaises(AssertionError):
                self.bench.benchmark_shape(a, b)
            timer.assert_not_called()

    def test_shape_validation(self):
        self.assertEqual(self.bench.parse_shape("128x256x16"), (128, 256, 16))
        for value in ("127,128,16", "128,129,16", "128,128,8", "0,128,16", "1,2", "a,b,c"):
            with self.subTest(shape=value), self.assertRaises(argparse.ArgumentTypeError):
                self.bench.parse_shape(value)

    def test_dispatch_preserves_layout_and_identity_for_every_stage(self):
        import cuda_ai_kernels
        extension = types.ModuleType("cuda_ai_kernels._C")
        seen = []
        def fake_gemm(a, b):
            seen.append((a, b))
            return a
        for name in (*GEMM_STAGE_IMPLS, "thread_tile", "tensor_core", "cutlass"):
            setattr(extension, "gemm_" + name, fake_gemm)
        for name in ("rmsnorm_naive", "rmsnorm_cache_x", "softmax_naive"):
            setattr(extension, name, fake_gemm)
        with mock.patch.dict(sys.modules, {"cuda_ai_kernels._C": extension}), \
             mock.patch.object(cuda_ai_kernels, "_C", extension, create=True):
            ops = load_module("all_dispatch_test", ROOT / "python/cuda_ai_kernels/ops.py")
            a, b = self.bench.make_inputs(128, 256, 16, device="cpu")
            for impl in (*GEMM_STAGE_IMPLS, "thread_tile"):
                ops.gemm(a, b, impl=impl)
                self.assertIs(seen[-1][0], a)
                self.assertIs(seen[-1][1], b)
            # Invalid row-major A must reach C++ validation without silent packing.
            bad_a = a.contiguous()
            ops.gemm(bad_a, b, impl="thread_tile_8x8")
            self.assertIs(seen[-1][0], bad_a)
