import sys
sys.path.insert(0, '.')
import torch
from util.multi_source import supervised_ce
for dtype in (torch.float32,torch.bfloat16):
    x=torch.randn(2,19,8,8,device='cuda',dtype=dtype,requires_grad=True)
    target=torch.full((2,8,8),255,device='cuda',dtype=torch.long)
    loss=supervised_ce(x,target)
    assert loss.item()==0 and torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()==0
    x.grad=None
    target[:,2:6,2:6]=3
    actual=supervised_ce(x,target)
    expected=torch.nn.functional.cross_entropy(x.float(),target,ignore_index=255)
    torch.testing.assert_close(actual,expected)
    actual.backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
print('PASS: all-ignore zero and gradient; partial-ignore CE equivalence; FP32/BF16')
