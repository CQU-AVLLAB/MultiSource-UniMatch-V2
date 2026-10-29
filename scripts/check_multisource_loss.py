import torch
from util.multi_source import reliability, weighted_ce, js
p=torch.full((2,19,8,8),1e-6,device='cuda'); p[:,0]=1-18e-6
score,w=reliability(p,p)
z=torch.randn_like(p,requires_grad=True)
loss=weighted_ce(z,p.argmax(1),score,w,torch.zeros_like(score))
loss.backward()
assert loss>0 and z.grad.abs().sum()>0
assert js(p,p).abs().max()<1e-6
print('selected pseudo-label loss and gradient PASS',loss.item())
