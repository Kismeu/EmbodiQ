"""Fail-closed loading of explicitly scoped AWQ block pilot artifacts."""
import hashlib
import torch
from embodiq.quant.qlinear import QuantLinear

FORMAT="embodiq/awq-block-pilot/1"


def tensor_hash(tensor):
    if tensor is None:return None
    data=tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(data).hexdigest()


def load_and_apply(root,path):
    artifact=torch.load(path,map_location="cpu",weights_only=True)
    if artifact.get("format")!=FORMAT:raise ValueError("Unsupported AWQ artifact format")
    if artifact.get("group_size")!=128 or not artifact.get("layers"):
        raise ValueError("Unsupported/empty AWQ artifact")
    selected=[]
    for name,row in artifact["layers"].items():
        layer=root.get_submodule(name)
        if not isinstance(layer,torch.nn.Linear):raise ValueError(f"Not an original Linear: {name}")
        if tensor_hash(layer.weight)!=row["weight_sha256"] or tensor_hash(layer.bias)!=row["bias_sha256"]:
            raise ValueError(f"Checkpoint mismatch: {name}")
        scale=row["scale"]
        if scale.shape!=(layer.in_features,) or not torch.isfinite(scale).all() or not (scale>0).all():
            raise ValueError(f"Invalid scale: {name}")
        clip=row.get("clip_max")
        if clip is not None and (clip.shape!=(layer.out_features,layer.in_features//128) or
                                 not torch.isfinite(clip).all() or (clip<0).any()):
            raise ValueError(f"Invalid clipping: {name}")
        selected.append((name,layer,scale,clip))
    # Validate every entry before constructing or replacing any module.
    replacements=[(name,QuantLinear.from_linear(layer,input_scale=scale.to(layer.weight.device),
                                               clip_max=clip)) for name,layer,scale,clip in selected]
    for name,new in replacements:
        parent,_,leaf=name.rpartition(".")
        setattr(root.get_submodule(parent) if parent else root,leaf,new)
    return artifact
