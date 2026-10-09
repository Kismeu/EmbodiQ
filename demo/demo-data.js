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
      "formal",
      "smoke"
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
      "run_id": "20261009_220944_ab8a066ab2",
      "phase": "complete",
      "saved_phase": "complete",
      "active": false,
      "created": "2026-10-09T22:09:44+0800",
      "profile": "smoke",
      "gpu": 3,
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
        "libero_spatial"
      ],
      "reasons": [],
      "stop_requested": false,
      "completed": 13,
      "total": 13,
      "jobs": [
        {
          "job_id": "rollout.openvla.fp16.libero_spatial",
          "stage": "rollout",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "rollout.openvla.rtn.libero_spatial",
          "stage": "rollout",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "rollout.vla_adapter.fp16.libero_spatial",
          "stage": "rollout",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "rollout.vla_adapter.rtn.libero_spatial",
          "stage": "rollout",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "rollout.fastwam.fp16.libero_spatial",
          "stage": "rollout",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "rollout.fastwam.rtn.libero_spatial",
          "stage": "rollout",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "resource.openvla.fp16",
          "stage": "resource",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "resource.openvla.rtn",
          "stage": "resource",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "resource.vla_adapter.fp16",
          "stage": "resource",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "resource.vla_adapter.rtn",
          "stage": "resource",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "resource.fastwam.fp16",
          "stage": "resource",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "resource.fastwam.rtn",
          "stage": "resource",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "report",
          "stage": "report",
          "status": "complete",
          "attempts": 1
        }
      ]
    },
    {
      "run_id": "20261009_225331_723f9a2d31",
      "phase": "complete",
      "saved_phase": "complete",
      "active": false,
      "created": "2026-10-09T22:53:31+0800",
      "profile": "smoke",
      "gpu": 3,
      "models": [
        "vla_adapter"
      ],
      "methods": [
        "rtn"
      ],
      "suites": [
        "libero_spatial"
      ],
      "reasons": [],
      "stop_requested": false,
      "completed": 3,
      "total": 3,
      "jobs": [
        {
          "job_id": "rollout.vla_adapter.rtn.libero_spatial",
          "stage": "rollout",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "resource.vla_adapter.rtn",
          "stage": "resource",
          "status": "complete",
          "attempts": 1
        },
        {
          "job_id": "report",
          "stage": "report",
          "status": "complete",
          "attempts": 1
        }
      ]
    }
  ],
  "reports": {
    "20261009_220944_ab8a066ab2": {
      "schema": "embodiq/pipeline-report/1",
      "run_id": "20261009_220944_ab8a066ab2",
      "profile": "smoke",
      "formal": false,
      "complete": true,
      "n_episodes": 6,
      "aggregates": [
        {
          "model": "openvla",
          "method": "fp16",
          "selected_suites_success_rate": 1.0,
          "four_suite_macro_success_rate": null,
          "steady_nvml_gib": 15.01171875,
          "query_ms": 209.28975753486156,
          "per_action_ms": 209.28975753486156,
          "delta_pp_vs_fp16": 0.0
        },
        {
          "model": "openvla",
          "method": "rtn",
          "selected_suites_success_rate": 1.0,
          "four_suite_macro_success_rate": null,
          "steady_nvml_gib": 6.689453125,
          "query_ms": 461.8577202782035,
          "per_action_ms": 461.8577202782035,
          "delta_pp_vs_fp16": 0.0
        },
        {
          "model": "vla_adapter",
          "method": "fp16",
          "selected_suites_success_rate": 1.0,
          "four_suite_macro_success_rate": null,
          "steady_nvml_gib": 4.119140625,
          "query_ms": 66.20559841394424,
          "per_action_ms": 8.27569980174303,
          "delta_pp_vs_fp16": 0.0
        },
        {
          "model": "vla_adapter",
          "method": "rtn",
          "selected_suites_success_rate": 1.0,
          "four_suite_macro_success_rate": null,
          "steady_nvml_gib": 3.166015625,
          "query_ms": 84.36894789338112,
          "per_action_ms": 10.54611848667264,
          "delta_pp_vs_fp16": 0.0
        },
        {
          "model": "fastwam",
          "method": "fp16",
          "selected_suites_success_rate": 1.0,
          "four_suite_macro_success_rate": null,
          "steady_nvml_gib": 24.017578125,
          "query_ms": 332.1246849372983,
          "per_action_ms": 33.21246849372983,
          "delta_pp_vs_fp16": 0.0
        },
        {
          "model": "fastwam",
          "method": "rtn",
          "selected_suites_success_rate": 1.0,
          "four_suite_macro_success_rate": null,
          "steady_nvml_gib": 16.720703125,
          "query_ms": 369.45471819490194,
          "per_action_ms": 36.945471819490194,
          "delta_pp_vs_fp16": 0.0
        }
      ],
      "pending_methods": [
        "awq",
        "hsa"
      ],
      "caveats": [
        "原精度配置名 fp16，实际模型使用 BF16。",
        "smoke 仅验证流程；只有正式档位四集合完整才报告四集合宏平均。",
        "资源输入取自成功训练演示的开发观测，不代表失败状态分布。",
        "动作步摊销推理耗时不是机器人端到端响应时间。",
        "权重、输入与源码在启动/恢复时校验，外部文件运行期间必须保持不变。"
      ],
      "cells": [
        {
          "model": "openvla",
          "method": "fp16",
          "suite": "libero_spatial",
          "n_episodes": 1,
          "n_success": 1,
          "success_rate": 1.0,
          "query_ms": 209.28975753486156,
          "per_action_ms": 209.28975753486156,
          "steady_nvml_gib": 15.01171875,
          "steady_alloc_gib": 14.421456336975098,
          "load_alloc_gib": 14.090496063232422
        },
        {
          "model": "openvla",
          "method": "rtn",
          "suite": "libero_spatial",
          "n_episodes": 1,
          "n_success": 1,
          "success_rate": 1.0,
          "query_ms": 461.8577202782035,
          "per_action_ms": 461.8577202782035,
          "steady_nvml_gib": 6.689453125,
          "steady_alloc_gib": 5.565064430236816,
          "load_alloc_gib": 15.070964813232422
        },
        {
          "model": "vla_adapter",
          "method": "fp16",
          "suite": "libero_spatial",
          "n_episodes": 1,
          "n_success": 1,
          "success_rate": 1.0,
          "query_ms": 66.20559841394424,
          "per_action_ms": 8.27569980174303,
          "steady_nvml_gib": 4.119140625,
          "steady_alloc_gib": 3.550358295440674,
          "load_alloc_gib": 3.3735694885253906
        },
        {
          "model": "vla_adapter",
          "method": "rtn",
          "suite": "libero_spatial",
          "n_episodes": 1,
          "n_success": 1,
          "success_rate": 1.0,
          "query_ms": 84.36894789338112,
          "per_action_ms": 10.54611848667264,
          "steady_nvml_gib": 3.166015625,
          "steady_alloc_gib": 1.8538293838500977,
          "load_alloc_gib": 3.9813919067382812
        },
        {
          "model": "fastwam",
          "method": "fp16",
          "suite": "libero_spatial",
          "n_episodes": 1,
          "n_success": 1,
          "success_rate": 1.0,
          "query_ms": 332.1246849372983,
          "per_action_ms": 33.21246849372983,
          "steady_nvml_gib": 24.017578125,
          "steady_alloc_gib": 23.258222103118896,
          "load_alloc_gib": 23.126478672027588
        },
        {
          "model": "fastwam",
          "method": "rtn",
          "suite": "libero_spatial",
          "n_episodes": 1,
          "n_success": 1,
          "success_rate": 1.0,
          "query_ms": 369.45471819490194,
          "per_action_ms": 36.945471819490194,
          "steady_nvml_gib": 16.720703125,
          "steady_alloc_gib": 15.244054317474365,
          "load_alloc_gib": 30.950697422027588
        }
      ]
    },
    "20261009_225331_723f9a2d31": {
      "schema": "embodiq/pipeline-report/1",
      "run_id": "20261009_225331_723f9a2d31",
      "profile": "smoke",
      "formal": false,
      "complete": true,
      "n_episodes": 1,
      "aggregates": [
        {
          "model": "vla_adapter",
          "method": "rtn",
          "selected_suites_success_rate": 1.0,
          "four_suite_macro_success_rate": null,
          "steady_nvml_gib": 3.166015625,
          "query_ms": 84.6506142988801,
          "per_action_ms": 10.581326787360013,
          "delta_pp_vs_fp16": null
        }
      ],
      "pending_methods": [
        "awq",
        "hsa"
      ],
      "caveats": [
        "原精度配置名 fp16，实际模型使用 BF16。",
        "smoke 仅验证流程；只有正式档位四集合完整才报告四集合宏平均。",
        "资源输入取自成功训练演示的开发观测，不代表失败状态分布。",
        "动作步摊销推理耗时不是机器人端到端响应时间。",
        "权重、输入与源码在启动/恢复时校验，外部文件运行期间必须保持不变。"
      ],
      "cells": [
        {
          "model": "vla_adapter",
          "method": "rtn",
          "suite": "libero_spatial",
          "n_episodes": 1,
          "n_success": 1,
          "success_rate": 1.0,
          "query_ms": 84.6506142988801,
          "per_action_ms": 10.581326787360013,
          "steady_nvml_gib": 3.166015625,
          "steady_alloc_gib": 1.8538293838500977,
          "load_alloc_gib": 3.9813919067382812
        }
      ]
    }
  },
  "read_only": true
};
