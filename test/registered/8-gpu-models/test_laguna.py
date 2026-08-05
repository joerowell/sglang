"""Nightly CI for Laguna (poolside) NVFP4 models.

Laguna has an unusual shape vs other models: heterogeneous per-layer attention
head counts (``num_attention_heads_per_layer`` -- full-attention layers 48 heads,
sliding-window layers 64 (XS-2.1) / 72 (S-2.1)) on a 3:1 sliding:full layer
pattern. That is exactly the config that motivated the per-layer-head planning fix
in #32625, so this end-to-end guards that path plus general Laguna serving.

Mirrors test/registered/8-gpu-models/test_mistral_large3.py and the NVFP4 nightly
examples (Inkling, Nemotron-Super).
"""

import unittest

from sglang.test.accuracy_test_runner import AccuracyTestParams
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.performance_test_runner import PerformanceTestParams
from sglang.test.run_combined_tests import run_combined_tests
from sglang.test.test_utils import ModelLaunchSettings, is_blackwell_system

# NVFP4 needs Blackwell FP4 kernels, so this runs on the Blackwell leg of the
# common 8-GPU suite (Hopper is skipped via is_blackwell_system below).
register_cuda_ci(est_time=2400, suite="nightly-8-gpu-common", nightly=True)

LAGUNA_XS_21_NVFP4_MODEL = "poolside/Laguna-XS-2.1-NVFP4"
LAGUNA_S_21_NVFP4_MODEL = "poolside/Laguna-S-2.1-NVFP4"

# TODO(poolside): confirm on the CI's B200 (sm100). We validated the recipe on
# consumer sm120 (RTX PRO 6000); backend selection can differ on sm100. Mirrors
# the Inkling/Nemotron NVFP4 recipe, adjusted for Laguna (glm4_moe + sliding
# window; no mamba).
NVFP4_ARGS = [
    "--tp=8",
    "--trust-remote-code",
    "--quantization=modelopt_fp4",
    "--attention-backend=fa4",  # TODO(poolside): confirm fa4 vs trtllm_mha on B200
    "--moe-runner-backend=flashinfer_trtllm",
    "--page-size=128",
    "--mem-fraction-static=0.85",
    # "--reasoning-parser=poolside_v1",  # TODO(poolside): Laguna reasoning parser
]

# TODO(poolside): measure on the CI hardware and floor with margin. These are
# only meaningful once the YaRN mscale rope fix lands upstream -- main's
# configs/laguna.py still double-applies mscale, which drags GSM8K down
# (~0.64 vs ~0.84 measured). Set high enough to catch that regression.
XS_21_GSM8K_BASELINE = 0.75
S_21_GSM8K_BASELINE = 0.80


@unittest.skipIf(not is_blackwell_system(), "NVFP4 requires Blackwell")
class TestLagunaNVFP4Nightly(unittest.TestCase):
    """Nightly accuracy (gsm8k) + perf for Laguna NVFP4, TP=8, Blackwell only."""

    def test_laguna_xs_21_nvfp4(self):
        run_combined_tests(
            models=[
                ModelLaunchSettings(
                    LAGUNA_XS_21_NVFP4_MODEL,
                    tp_size=8,
                    extra_args=NVFP4_ARGS,
                    variant="NVFP4",
                )
            ],
            test_name="Laguna-XS-2.1-NVFP4",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=XS_21_GSM8K_BASELINE
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_laguna_xs21",
            ),
        )

    def test_laguna_s_21_nvfp4(self):
        run_combined_tests(
            models=[
                ModelLaunchSettings(
                    LAGUNA_S_21_NVFP4_MODEL,
                    tp_size=8,
                    extra_args=NVFP4_ARGS,
                    variant="NVFP4",
                )
            ],
            test_name="Laguna-S-2.1-NVFP4",
            accuracy_params=AccuracyTestParams(
                dataset="gsm8k", baseline_accuracy=S_21_GSM8K_BASELINE
            ),
            performance_params=PerformanceTestParams(
                profile_dir="performance_profiles_laguna_s21",
            ),
        )


if __name__ == "__main__":
    unittest.main()
