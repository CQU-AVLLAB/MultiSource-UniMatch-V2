"""Paper methodology equations (7), (10)--(17). Inputs use ImageNet normalization."""
import math
import torch
import torch.nn.functional as F


def js(p, q):
    p, q = p.float().clamp_min(1e-7), q.float().clamp_min(1e-7)
    m = (p + q) / 2
    return ((p * (p.log()-m.log())).sum(1) + (q * (q.log()-m.log())).sum(1)) / 2


def reliability(p, q, alpha=.5, penalty=1., gamma=1.):
    entropy = -(p.float().clamp_min(1e-7).log()*p.float()).sum(1)/math.log(p.shape[1])
    uncertainty = (alpha*entropy + (1-alpha)*js(p,q)/math.log(2)).clamp(0,1)
    return p.max(1).values*torch.exp(-penalty*uncertainty), torch.exp(-gamma*uncertainty)


def weighted_ce(logits, target, score, weight, ignore, threshold=.85):
    selected = (score > threshold) & (ignore != 255)
    return (F.cross_entropy(logits.float(),target,reduction='none')*weight*selected).sum()/(selected.sum()+1e-7)


def weather(x, kind=None):
    mean=x.new_tensor([.485,.456,.406])[None,:,None,None]
    std=x.new_tensor([.229,.224,.225])[None,:,None,None]
    image=(x*std+mean).clamp(0,1)
    results=[]
    for image_i in image:
        k=int(torch.randint(4,()).item()) if kind is None else kind
        h,w=image_i.shape[-2:]
        if k==0:
            # Smooth synthetic relative depth; no geometric displacement.
            depth=F.interpolate(torch.rand(1,1,4,4,device=x.device),size=(h,w),mode='bilinear',align_corners=False)[0]
            transmission=torch.exp(-(.3+torch.rand((),device=x.device)*.7)*depth)
            out=image_i*transmission+.9*(1-transmission)
        elif k==1:
            seeds=(torch.rand(1,1,h,w,device=x.device)>.997).float()
            streak=F.avg_pool2d(seeds,(15,3),stride=1,padding=(7,1))[0]*12
            out=image_i*.9+streak
        elif k==2:
            flakes=(torch.rand(1,1,h,w,device=x.device)>.995).float()
            flakes=F.max_pool2d(flakes,3,stride=1,padding=1)
            flakes=F.avg_pool2d(flakes,3,stride=1,padding=1)[0]
            out=F.avg_pool2d(image_i[None],3,stride=1,padding=1)[0]*(1-flakes)+flakes
        else:
            out=(.45+.3*torch.rand((),device=x.device))*image_i.pow(1.4+torch.rand((),device=x.device))
        results.append(out.clamp(0,1))
    return (torch.stack(results)-mean)/std


def supervised_ce(logits, target, ignore_index=255):
    """Mean over valid pixels; differentiable zero for all-ignore crops."""
    values = F.cross_entropy(logits.float(), target, ignore_index=ignore_index, reduction='none')
    return values.sum() / (target != ignore_index).sum().clamp_min(1)
