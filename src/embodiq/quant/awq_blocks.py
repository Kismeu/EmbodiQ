"""Replayable attention/MLP scale search and per-group clipping primitives."""
import random
import torch
from embodiq.quant.pack import group_quantize_int4, dequantize_int4


def map_tensors(value, device):
    if isinstance(value, torch.Tensor): return value.detach().to(device).clone()
    if isinstance(value, tuple): return tuple(map_tensors(v, device) for v in value)
    if isinstance(value, list): return [map_tensors(v, device) for v in value]
    if isinstance(value, dict): return {k:map_tensors(v,device) for k,v in value.items()}
    if value is None or isinstance(value,(int,float,bool,str)): return value
    raise TypeError(f"Non-replayable block argument: {type(value).__name__}")


class BlockCapture:
    def __init__(self, module, limit=4, seed=42):
        if limit < 1: raise ValueError("Capture limit must be positive")
        self.module,self.limit,self.rng=module,limit,random.Random(seed)
        self.calls=0; self.records=[]
    def __enter__(self):
        def hook(module,args,kwargs):
            replay_kwargs=dict(kwargs)
            cache=kwargs.get("past_key_value")
            if cache is not None:
                if not hasattr(cache,"get_seq_length") or cache.get_seq_length(getattr(module,"layer_idx",0))!=0:
                    raise ValueError("Block calibration requires cache-free replay; nonempty/unknown cache")
                # Empty DynamicCache only receives current K/V during this call.
                # The first tensor output is identical without that output cache.
                replay_kwargs["past_key_value"]=None
                replay_kwargs["use_cache"]=False
            self.calls+=1
            idx=len(self.records) if len(self.records)<self.limit else self.rng.randrange(self.calls)
            if idx < self.limit:
                record=(map_tensors(args,"cpu"),map_tensors(replay_kwargs,"cpu"))
                if idx==len(self.records):self.records.append(record)
                else:self.records[idx]=record
        self.handle=self.module.register_forward_pre_hook(hook,with_kwargs=True)
        return self
    def __exit__(self,*_):self.handle.remove()


def first_output(value):
    return value[0] if isinstance(value,tuple) else value


@torch.no_grad()
def search_block_scale(block,linears,records,mean_abs,group_size=128,n_grid=20):
    if not records or not linears or n_grid<1: raise ValueError("Empty block calibration")
    if block.training: raise ValueError("Block must be in eval mode")
    device=linears[0].weight.device
    backups=[layer.weight.detach().clone() for layer in linears]
    if any(w.shape[1]!=mean_abs.numel() for w in backups):raise ValueError("Shared scale input dimensions differ")
    reference=[]
    for args,kwargs in records:
        y=first_output(block(*map_tensors(args,device),**map_tensors(kwargs,device)))
        reference.append(y.detach().float().cpu())
    best=None;history=[]
    try:
        for i in range(n_grid):
            s=mean_abs.to(device).float().pow(i/n_grid).clamp(min=1e-4)
            s=s/(s.max()*s.min()).sqrt()
            for layer,w in zip(linears,backups):
                q,sc,z=group_quantize_int4(w.float()*s[None],group_size)
                effective=dequantize_int4(q,sc.bfloat16(),z.bfloat16(),group_size)/s[None]
                layer.weight.copy_(effective.to(layer.weight.dtype))
            squared,count=0.0,0
            for (args,kwargs),ref in zip(records,reference):
                out=first_output(block(*map_tensors(args,device),**map_tensors(kwargs,device))).float()
                squared+=float((out-ref.to(device)).square().sum()); count+=out.numel()
            loss=squared/count
            if not torch.isfinite(torch.tensor(loss)):raise ValueError("Non-finite block error")
            history.append(loss)
            if best is None or loss<best["loss"]:best=dict(scale=s.cpu().clone(),alpha=i/n_grid,loss=loss)
    finally:
        for layer,w in zip(linears,backups):layer.weight.copy_(w)
    best.update(losses=history,rtn_loss=history[0],records=len(records),objective="block_output_mse")
    return best


@torch.no_grad()
def search_clip(weight,activation,scale,group_size=128,n_grid=20,max_shrink=.5,output_batch=64):
    """Symmetric per-group clipping on scaled weights, selected by group output MSE.

    Callers must skip query/key projections whose error is governed by QK^T.
    """
    if n_grid<1 or not 0<max_shrink<=1 or output_batch<1:raise ValueError("Invalid clipping budget")
    w=weight.detach().float()*scale.to(weight.device).float()[None]
    x=activation.to(weight.device).float()/scale.to(weight.device).float()[None]
    if not torch.isfinite(w).all() or not torch.isfinite(x).all():raise ValueError("Non-finite clip input")
    n,k=w.shape;groups=k//group_size
    wg=w.reshape(n,groups,group_size);xg=x.reshape(-1,groups,group_size)
    selected=[];errors=[]
    for start in range(0,n,output_batch):
        block=wg[start:start+output_batch]
        maximum=block.abs().amax(-1)
        reference=torch.einsum("tgi,ogi->otg",xg,block)
        best_loss=torch.full_like(maximum,float("inf"));best_max=maximum.clone()
        for i in range(max(1,int(n_grid*max_shrink))):
            clip=maximum*(1-i/n_grid)
            bounded=block.clamp(-clip[...,None],clip[...,None]).reshape(-1,k)
            q,s,z=group_quantize_int4(bounded,group_size)
            reconstructed=dequantize_int4(q,s.bfloat16(),z.bfloat16(),group_size).reshape_as(block)
            error=(torch.einsum("tgi,ogi->otg",xg,reconstructed)-reference).square().mean(1)
            improve=error<best_loss
            best_loss=torch.where(improve,error,best_loss)
            best_max=torch.where(improve,clip,best_max)
        selected.append(best_max.cpu());errors.append(best_loss.cpu())
    return dict(clip_max=torch.cat(selected),group_mse=torch.cat(errors),objective="per_group_output_mse")
