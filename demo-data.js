window.EMBODIQ_DEMO = {
  "capabilities": {
    "models": [
      "openvla",
      "vla_adapter",
      "fastwam"
    ],
    "methods": {
      "fp16": "ready",
      "rtn": "ready",
      "awq": "pending",
      "hsa": "pending"
    },
    "benchmark": "libero",
    "profiles": [
      "formal"
    ],
    "execution": "local, sequential, exclusive GPU",
    "fastwam_mode": "first_frame",
    "suites": [
      "libero_spatial",
      "libero_object",
      "libero_goal",
      "libero_10"
    ],
    "csrf_token": "read-only-demo",
    "default_gpu": 0
  },
  "runs": [
    {
      "run_id": "formal_history_v3",
      "phase": "complete",
      "saved_phase": "complete",
      "active": false,
      "source": "archive",
      "created": "LIBERO 四类任务",
      "profile": "formal",
      "models": [
        "openvla",
        "vla_adapter",
        "fastwam"
      ],
      "methods": [
        "fp16",
        "rtn"
      ],
      "suites": [
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10"
      ],
      "n_episodes": 12000,
      "reasons": [],
      "stop_requested": false,
      "completed": 0,
      "total": 0,
      "jobs": []
    }
  ],
  "reports": {
    "formal_history_v3": {
      "schema": "embodiq/showcase-report/1",
      "run_id": "formal_history_v3",
      "profile": "formal",
      "formal": true,
      "complete": true,
      "n_episodes": 12000,
      "cells": [
        {
          "model": "fastwam",
          "method": "fp16",
          "suite": "libero_10",
          "n_episodes": 500,
          "n_success": 468,
          "success_rate": 0.936,
          "query_ms": 326.95943222691614,
          "per_action_ms": 32.69594322269161,
          "steady_nvml_gib": 24.013671875,
          "steady_alloc_gib": 23.256680965423584,
          "load_alloc_gib": 23.132872104644775
        },
        {
          "model": "fastwam",
          "method": "fp16",
          "suite": "libero_goal",
          "n_episodes": 500,
          "n_success": 487,
          "success_rate": 0.974,
          "query_ms": 314.38731418301666,
          "per_action_ms": 31.43873141830166,
          "steady_nvml_gib": 24.013671875,
          "steady_alloc_gib": 23.256680965423584,
          "load_alloc_gib": 23.132872104644775
        },
        {
          "model": "fastwam",
          "method": "fp16",
          "suite": "libero_object",
          "n_episodes": 500,
          "n_success": 496,
          "success_rate": 0.992,
          "query_ms": 300.91055020689964,
          "per_action_ms": 30.091055020689964,
          "steady_nvml_gib": 24.013671875,
          "steady_alloc_gib": 23.256680965423584,
          "load_alloc_gib": 23.132872104644775
        },
        {
          "model": "fastwam",
          "method": "fp16",
          "suite": "libero_spatial",
          "n_episodes": 500,
          "n_success": 485,
          "success_rate": 0.97,
          "query_ms": 305.10620728755987,
          "per_action_ms": 30.510620728755992,
          "steady_nvml_gib": 24.017578125,
          "steady_alloc_gib": 23.258222103118896,
          "load_alloc_gib": 23.126478672027588
        },
        {
          "model": "fastwam",
          "method": "rtn",
          "suite": "libero_10",
          "n_episodes": 500,
          "n_success": 472,
          "success_rate": 0.944,
          "query_ms": 370.7182419858873,
          "per_action_ms": 37.07182419858873,
          "steady_nvml_gib": 16.744140625,
          "steady_alloc_gib": 15.246236324310303,
          "load_alloc_gib": 30.957090854644775
        },
        {
          "model": "fastwam",
          "method": "rtn",
          "suite": "libero_goal",
          "n_episodes": 500,
          "n_success": 485,
          "success_rate": 0.97,
          "query_ms": 374.4408661437531,
          "per_action_ms": 37.444086614375315,
          "steady_nvml_gib": 16.744140625,
          "steady_alloc_gib": 15.246236324310303,
          "load_alloc_gib": 30.957090854644775
        },
        {
          "model": "fastwam",
          "method": "rtn",
          "suite": "libero_object",
          "n_episodes": 500,
          "n_success": 497,
          "success_rate": 0.994,
          "query_ms": 367.0784846382836,
          "per_action_ms": 36.70784846382836,
          "steady_nvml_gib": 16.744140625,
          "steady_alloc_gib": 15.246236324310303,
          "load_alloc_gib": 30.957090854644775
        },
        {
          "model": "fastwam",
          "method": "rtn",
          "suite": "libero_spatial",
          "n_episodes": 500,
          "n_success": 483,
          "success_rate": 0.966,
          "query_ms": 369.0317499823868,
          "per_action_ms": 36.90317499823868,
          "steady_nvml_gib": 16.720703125,
          "steady_alloc_gib": 15.244054317474365,
          "load_alloc_gib": 30.950697422027588
        },
        {
          "model": "openvla",
          "method": "fp16",
          "suite": "libero_10",
          "n_episodes": 500,
          "n_success": 261,
          "success_rate": 0.522,
          "query_ms": 203.82643000533182,
          "per_action_ms": 203.82643000533182,
          "steady_nvml_gib": 15.01171875,
          "steady_alloc_gib": 14.424176216125488,
          "load_alloc_gib": 14.098716735839844
        },
        {
          "model": "openvla",
          "method": "fp16",
          "suite": "libero_goal",
          "n_episodes": 500,
          "n_success": 391,
          "success_rate": 0.782,
          "query_ms": 204.6856214913229,
          "per_action_ms": 204.6856214913229,
          "steady_nvml_gib": 14.97265625,
          "steady_alloc_gib": 14.412747859954834,
          "load_alloc_gib": 14.098716735839844
        },
        {
          "model": "openvla",
          "method": "fp16",
          "suite": "libero_object",
          "n_episodes": 500,
          "n_success": 446,
          "success_rate": 0.892,
          "query_ms": 208.7239444690446,
          "per_action_ms": 208.7239444690446,
          "steady_nvml_gib": 15.01171875,
          "steady_alloc_gib": 14.415369510650635,
          "load_alloc_gib": 14.098716735839844
        },
        {
          "model": "openvla",
          "method": "fp16",
          "suite": "libero_spatial",
          "n_episodes": 500,
          "n_success": 441,
          "success_rate": 0.882,
          "query_ms": 212.06911441559592,
          "per_action_ms": 212.06911441559592,
          "steady_nvml_gib": 15.01171875,
          "steady_alloc_gib": 14.422394752502441,
          "load_alloc_gib": 14.090496063232422
        },
        {
          "model": "openvla",
          "method": "rtn",
          "suite": "libero_10",
          "n_episodes": 500,
          "n_success": 249,
          "success_rate": 0.498,
          "query_ms": 324.1944394508998,
          "per_action_ms": 324.1944394508998,
          "steady_nvml_gib": 6.701171875,
          "steady_alloc_gib": 5.567902565002441,
          "load_alloc_gib": 15.07809829711914
        },
        {
          "model": "openvla",
          "method": "rtn",
          "suite": "libero_goal",
          "n_episodes": 500,
          "n_success": 378,
          "success_rate": 0.756,
          "query_ms": 312.30028878276545,
          "per_action_ms": 312.30028878276545,
          "steady_nvml_gib": 6.701171875,
          "steady_alloc_gib": 5.555878639221191,
          "load_alloc_gib": 15.07809829711914
        },
        {
          "model": "openvla",
          "method": "rtn",
          "suite": "libero_object",
          "n_episodes": 500,
          "n_success": 439,
          "success_rate": 0.878,
          "query_ms": 315.218724993368,
          "per_action_ms": 315.218724993368,
          "steady_nvml_gib": 6.701171875,
          "steady_alloc_gib": 5.559164524078369,
          "load_alloc_gib": 15.07809829711914
        },
        {
          "model": "openvla",
          "method": "rtn",
          "suite": "libero_spatial",
          "n_episodes": 500,
          "n_success": 413,
          "success_rate": 0.826,
          "query_ms": 323.4862208366394,
          "per_action_ms": 323.4862208366394,
          "steady_nvml_gib": 6.689453125,
          "steady_alloc_gib": 5.566346168518066,
          "load_alloc_gib": 15.070964813232422
        },
        {
          "model": "vla_adapter",
          "method": "fp16",
          "suite": "libero_10",
          "n_episodes": 500,
          "n_success": 477,
          "success_rate": 0.954,
          "query_ms": 65.38428232073784,
          "per_action_ms": 8.17303529009223,
          "steady_nvml_gib": 4.47265625,
          "steady_alloc_gib": 3.549811840057373,
          "load_alloc_gib": 3.3819541931152344
        },
        {
          "model": "vla_adapter",
          "method": "fp16",
          "suite": "libero_goal",
          "n_episodes": 500,
          "n_success": 487,
          "success_rate": 0.974,
          "query_ms": 62.769575876494244,
          "per_action_ms": 7.8461969845617805,
          "steady_nvml_gib": 4.458984375,
          "steady_alloc_gib": 3.5419511795043945,
          "load_alloc_gib": 3.3819541931152344
        },
        {
          "model": "vla_adapter",
          "method": "fp16",
          "suite": "libero_object",
          "n_episodes": 500,
          "n_success": 497,
          "success_rate": 0.994,
          "query_ms": 66.76925268645088,
          "per_action_ms": 8.34615658580636,
          "steady_nvml_gib": 4.111328125,
          "steady_alloc_gib": 3.5442442893981934,
          "load_alloc_gib": 3.3819541931152344
        },
        {
          "model": "vla_adapter",
          "method": "fp16",
          "suite": "libero_spatial",
          "n_episodes": 500,
          "n_success": 490,
          "success_rate": 0.98,
          "query_ms": 64.59930296366413,
          "per_action_ms": 8.074912870458016,
          "steady_nvml_gib": 4.119140625,
          "steady_alloc_gib": 3.550358295440674,
          "load_alloc_gib": 3.3735694885253906
        },
        {
          "model": "vla_adapter",
          "method": "rtn",
          "suite": "libero_10",
          "n_episodes": 500,
          "n_success": 468,
          "success_rate": 0.936,
          "query_ms": 83.41344796742003,
          "per_action_ms": 10.426680995927503,
          "steady_nvml_gib": 3.4921875,
          "steady_alloc_gib": 1.8549480438232422,
          "load_alloc_gib": 3.9898681640625
        },
        {
          "model": "vla_adapter",
          "method": "rtn",
          "suite": "libero_goal",
          "n_episodes": 500,
          "n_success": 438,
          "success_rate": 0.876,
          "query_ms": 83.02270049229264,
          "per_action_ms": 10.37783756153658,
          "steady_nvml_gib": 3.48046875,
          "steady_alloc_gib": 1.8471875190734863,
          "load_alloc_gib": 3.9898681640625
        },
        {
          "model": "vla_adapter",
          "method": "rtn",
          "suite": "libero_object",
          "n_episodes": 500,
          "n_success": 476,
          "success_rate": 0.952,
          "query_ms": 91.40771140034,
          "per_action_ms": 11.4259639250425,
          "steady_nvml_gib": 3.1328125,
          "steady_alloc_gib": 1.849452018737793,
          "load_alloc_gib": 3.9898681640625
        },
        {
          "model": "vla_adapter",
          "method": "rtn",
          "suite": "libero_spatial",
          "n_episodes": 500,
          "n_success": 493,
          "success_rate": 0.986,
          "query_ms": 86.7705771389107,
          "per_action_ms": 10.846322142363837,
          "steady_nvml_gib": 3.166015625,
          "steady_alloc_gib": 1.8538293838500977,
          "load_alloc_gib": 3.9813919067382812
        }
      ],
      "aggregates": [
        {
          "model": "openvla",
          "method": "fp16",
          "selected_suites_success_rate": 0.7695000000000001,
          "four_suite_macro_success_rate": 0.7695000000000001,
          "delta_pp_vs_fp16": 0.0,
          "query_ms": 207.3262775953238,
          "per_action_ms": 207.3262775953238,
          "steady_nvml_gib": 15.01171875
        },
        {
          "model": "openvla",
          "method": "rtn",
          "selected_suites_success_rate": 0.7395,
          "four_suite_macro_success_rate": 0.7395,
          "delta_pp_vs_fp16": -3.0000000000000027,
          "query_ms": 318.7999185159182,
          "per_action_ms": 318.7999185159182,
          "steady_nvml_gib": 6.701171875
        },
        {
          "model": "vla_adapter",
          "method": "fp16",
          "selected_suites_success_rate": 0.9755,
          "four_suite_macro_success_rate": 0.9755,
          "delta_pp_vs_fp16": 0.0,
          "query_ms": 64.88060346183677,
          "per_action_ms": 8.110075432729596,
          "steady_nvml_gib": 4.47265625
        },
        {
          "model": "vla_adapter",
          "method": "rtn",
          "selected_suites_success_rate": 0.9375,
          "four_suite_macro_success_rate": 0.9375,
          "delta_pp_vs_fp16": -3.8000000000000034,
          "query_ms": 86.15360924974084,
          "per_action_ms": 10.769201156217605,
          "steady_nvml_gib": 3.4921875
        },
        {
          "model": "fastwam",
          "method": "fp16",
          "selected_suites_success_rate": 0.968,
          "four_suite_macro_success_rate": 0.968,
          "delta_pp_vs_fp16": 0.0,
          "query_ms": 311.84087597609806,
          "per_action_ms": 31.184087597609807,
          "steady_nvml_gib": 24.017578125
        },
        {
          "model": "fastwam",
          "method": "rtn",
          "selected_suites_success_rate": 0.9684999999999999,
          "four_suite_macro_success_rate": 0.9684999999999999,
          "delta_pp_vs_fp16": 0.04999999999999449,
          "query_ms": 370.3173356875777,
          "per_action_ms": 37.03173356875777,
          "steady_nvml_gib": 16.744140625
        }
      ],
      "pending_methods": [
        "awq",
        "hsa"
      ],
      "display_note": "评测结果汇总 · 24 个组合 · 12,000 回合",
      "caveats": [
        "原始精度采用 BF16；RTN 使用 4 位权重，每组 128 个权重。",
        "展示已完成评测的汇总结果，不是实时运行。",
        "本组显存数据的采样间隔为 100 毫秒，可能遗漏短时峰值。",
        "本组资源记录包含权重路径与量化配置，未保存完整权重文件指纹。",
        "查询耗时宜在同一模型内比较；不同模型的动作块长度与重规划频率不同。",
        "资源观测来自训练演示，仅用于测量耗时和显存；成功率来自独立的 12,000 个正式任务回合。"
      ],
      "provenance": {
        "source_schema": "embodiq/result_matrix/2",
        "matrix_sha256": "680c3d6297b3a37e4086e81936144ab3d18cddc29322d7f67754b812d5a89393",
        "generated": "2026-10-10T15:26:25+0800",
        "resource_sha256": {
          "fastwam/fp16": "96e5cd5be09c66ad257f4c89bef5cc45bea4ccd0e5d6dcc8b6611fd8ec2c10f9",
          "fastwam/rtn": "eb31aeb153f04c228209ff35e941623e1fede5f1780787b972591c79492453e2",
          "openvla/fp16": "839df13247dd3042ddc73a04d984f67bb81a75978200f8210d79210c4c3ed768",
          "openvla/rtn": "db252e65a953fe8a2b9f4a26a20f00513bd5234f5f01ab818e7c98639566f4ae",
          "vla_adapter/fp16": "fa34c2d6376f2afad1bac4542745c54a1cb4b11e95c8441ecf74d5886a25c6f8",
          "vla_adapter/rtn": "9aadaf53fe26c7e43111e149a477ad4f46f9f37c915f87d04481bb36562a7699"
        }
      }
    }
  },
  "read_only": true
};
