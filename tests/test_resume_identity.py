import copy
import dataclasses
import json

import pytest

from embodiq.core.protocol import PROTOCOLS
from embodiq.core.resume import validate_resume
from embodiq.core.types import PolicySpec, GripperConvention


@pytest.fixture
def case():
    spec = PolicySpec("openvla-7b", "openvla", "/local/checkpoint", ("agentview",),
                      256, False, 1, 1, GripperConvention.ENV_NATIVE_BINARIZED, "bfloat16")
    protocol = PROTOCOLS["official.openvla"]
    kwargs = dict(spec=spec, protocol=protocol, quant={"method": "fp16"},
                  git={"embodiq": "abc"}, versions={"torch": "2.2"}, suite="libero_object", task_id=0)
    saved = json.loads(json.dumps(dict(policy=dataclasses.asdict(spec),
                  protocol=dataclasses.asdict(protocol), quant=kwargs["quant"],
                  task_suite="libero_object", task_id=0, total_episodes=50, error_episodes=[],
                  run=dict(status="ok", git=kwargs["git"], versions=kwargs["versions"]))))
    return saved, kwargs


def test_same_run_can_resume(case):
    saved, kwargs = case
    validate_resume(saved, **kwargs)


@pytest.mark.parametrize("change", ["model", "quant", "checkpoint", "protocol", "incomplete", "error", "code"])
def test_other_or_incomplete_run_cannot_resume(case, change):
    saved, kwargs = case
    saved = copy.deepcopy(saved)
    if change == "model": saved["policy"]["family"] = "vla_adapter"
    elif change == "quant": saved["quant"]["method"] = "rtn"
    elif change == "checkpoint": saved["policy"]["checkpoint"] = "/other"
    elif change == "protocol": saved["protocol"]["num_steps_wait"] = 30
    elif change == "incomplete": saved["total_episodes"] = 49
    elif change == "error": saved["error_episodes"] = [3]
    elif change == "code": saved["run"]["git"] = {"embodiq": "changed"}
    with pytest.raises(ValueError, match="incompatible"):
        validate_resume(saved, **kwargs)
