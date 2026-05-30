import torch

def shrinkage(x, kappa):
    """
    软阈值函数，使用 PyTorch 张量。
    """
    return torch.sign(x) * torch.maximum(torch.abs(x) - kappa, torch.tensor(0.0, device=x.device))

def sparse_bls(a, b, lam, itrs):
    """
    MATLAB的 sparse_bls 函数的Python等效实现，使用 PyTorch 张量。
    """
    aa = a.T @ a
    m, n = a.shape[1], b.shape[1]
    
    x = torch.zeros((m, n), device=a.device)
    wk = x
    ok = x
    uk = x
    
    l1 = torch.linalg.inv(aa + torch.eye(m, device=a.device))
    l2 = l1 @ a.T @ b
    
    for _ in range(itrs):
        tempc = ok - uk
        ck = l2 + l1 @ tempc
        ok = shrinkage(ck + uk, lam)
        uk = uk + (ck - ok)
        wk = ok
        
    return wk