import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import time
import copy

from sparse_bls import sparse_bls


# =========================================================================
#  BTSK-BLS: 严格按照论文 Eqs. (4)(6)(8)(21)(22)(24)(26)(28) 实现
#  关键区分: u_nk (FCM 隶属度, Eq.24)  ≠  μ_nk (高斯隶属度, Eq.4)
# =========================================================================


def compute_sigma(H, C, U, m):
    """
    计算高斯隶属函数的宽度参数 σ_ki (论文 Eq.4 下方定义):
        σ_ki^2 = Σ_n u_nk (h_ni - c_ki)^2 / N_k
        N_k 为第 k 个簇的样本大小 (通常为隶属度之和)

    参数:
        H: (N, L) 隐藏层矩阵
        C: (K, L) 聚类中心
        U: (N, K) FCM 隶属矩阵
        m: 模糊指数 (在此函数中按论文语义N_k直接用U计算即可，不用U**m)

    返回:
        sigma: (K, L) 每个簇每个维度的宽度参数 σ_ki (标准差)
    """
    # 按照模糊规则，簇的 soft 容量 N_k 为隶属度之和
    N_k = torch.sum(U, dim=0, keepdim=True).T             # (K, 1)
    N_k = torch.clamp(N_k, min=1e-12)

    K, L = C.shape
    sigma = torch.zeros_like(C)
    for k in range(K):
        diff = H - C[k].unsqueeze(0)                      # (N, L)
        # 计算加权方差 σ_ki^2
        variance_k = (U[:, k].unsqueeze(1) * diff ** 2).sum(dim=0) / N_k[k]
        # 根据论文必须开平方根得到宽度参数 σ_ki
        sigma[k] = torch.sqrt(variance_k)

    # 设定合理的下界，防止作为分母时引发数值不稳定
    sigma = torch.clamp(sigma, min=1e-8)
    return sigma


def compute_gaussian_membership(H, C, sigma):
    """
    计算高斯隶属函数 μ_nk 及其归一化 μ̄_nk (论文 Eq.4 + Eq.6 定义)。
    为数值稳定,在 log 域计算后用 log-sum-exp 归一化。

        log μ_nk = -Σ_i (h_ni - c_ki)^2 / σ_ki
        μ̄_nk    = μ_nk / Σ_j μ_nj  (log-sum-exp)

    参数:
        H:     (N, L) 隐藏层矩阵
        C:     (K, L) 聚类中心
        sigma: (K, L) 宽度参数

    返回:
        mu_bar: (N, K) 归一化高斯隶属度 μ̄_nk
        log_mu: (N, K) log μ_nk (用于调试)
    """
    N, L = H.shape
    K = C.shape[0]
    device = H.device

    # log μ_nk = -Σ_i (h_ni - c_ki)^2 / σ_ki
    log_mu = torch.zeros((N, K), device=device)
    for k in range(K):
        diff2 = (H - C[k].unsqueeze(0)) ** 2              # (N, L)
        log_mu[:, k] = -torch.sum(diff2 / sigma[k].unsqueeze(0), dim=1)

    # log-sum-exp 归一化 → μ̄_nk
    log_sum = torch.logsumexp(log_mu, dim=1, keepdim=True)  # (N, 1)
    log_mu_bar = log_mu - log_sum
    mu_bar = torch.exp(log_mu_bar)

    # 确保行和 = 1
    mu_bar = mu_bar / (mu_bar.sum(dim=1, keepdim=True) + 1e-12)
    return mu_bar, log_mu


def sample_laplace_centers(num_samples, h_tilde, theta, device):
    """
    论文 Eq. (22): 从 L 维 Laplace 分布采样聚类中心
        c_k ~ Π_i L(h̃_i, ϑ)
    h̃_i = H 第 i 列的均值, ϑ = 5 (论文推荐)
    """
    L = h_tilde.shape[0]
    u = torch.rand(num_samples, L, device=device) - 0.5
    samples = h_tilde.unsqueeze(0) - theta * torch.sign(u) * torch.log(
        torch.clamp(1.0 - 2.0 * torch.abs(u), min=1e-12)
    )
    return samples


def update_U(H, C, m):
    """
    论文 Eq. (24): 更新 FCM 隶属矩阵 U
        u_nk = ||h_n - c_k||^{-2/(m-1)} / Σ_j ||h_n - c_j||^{-2/(m-1)}
    """
    dist = torch.cdist(H, C, p=2)                         # (N, K)
    dist = torch.clamp(dist, min=1e-10)
    power = -2.0 / (m - 1.0)
    dist_pow = dist ** power
    U = dist_pow / (torch.sum(dist_pow, dim=1, keepdim=True) + 1e-12)
    return U


def update_C(H, H_bar, C, U, W_stacked, Y, sigma, mu_bar, K, m):
    """
    论文 Eq. (26): 严格按公式更新聚类中心 c_ki

                1/K Σ_n u^m_nk h_ni  +  2 Σ_n (y_n - h̄_n w_k)^2 · [Σ_j μ_nj - μ_nk]/(Σ_j μ_nj)^2 · μ_nk · h_ni/σ_ki
    c_ki = ───────────────────────────────────────────────────────────────────────────────────────────────────────────────
                1/K Σ_n u^m_nk       +  2 Σ_n (y_n - h̄_n w_k)^2 · [Σ_j μ_nj - μ_nk]/(Σ_j μ_nj)^2 · μ_nk · 1/σ_ki

    关键化简: [Σ_j μ_nj - μ_nk]/(Σ_j μ_nj)^2 × μ_nk = μ̄_nk × (1 - μ̄_nk)

    参数:
        H:          (N, L)     隐藏层矩阵
        H_bar:      (N, L+1)   增广隐藏层 [1, H]
        C:          (K, L)     当前聚类中心
        U:          (N, K)     FCM 隶属矩阵
        W_stacked:  (K*(L+1), out_dim)  各规则输出权重纵向堆叠
        Y:          (N, out_dim) 目标输出
        sigma:      (K, L)     宽度参数
        mu_bar:     (N, K)     归一化高斯隶属度 μ̄_nk
        K:          int        规则数
        m:          float      模糊指数

    返回:
        C_new:      (K, L)     更新后的聚类中心
    """
    N, L = H.shape
    L_bar = H_bar.shape[1]
    device = H.device
    U_m = U ** m                                           # (N, K)

    C_new = torch.zeros_like(C)
    for k in range(K):
        w_k = W_stacked[k * L_bar:(k + 1) * L_bar]        # (L+1, out_dim)
        res = Y - H_bar @ w_k                              # (N, out_dim)
        res2 = torch.sum(res ** 2, dim=1)                  # (N,)  ||y_n - h̄_n w_k||^2

        # 化简项: μ̄_nk(1 - μ̄_nk) × (y_n - h̄_n w_k)^2
        A_n = mu_bar[:, k] * (1.0 - mu_bar[:, k]) * res2  # (N,)

        # 分子分母的第一项 (FCM 部分)
        u_m_k = U_m[:, k]                                  # (N,)
        fcm_num = (1.0 / K) * (u_m_k.unsqueeze(1) * H).sum(dim=0)   # (L,)
        fcm_den = (1.0 / K) * u_m_k.sum()                           # scalar

        # 分子分母的第二项 (TSK 修正部分)
        # Σ_n A_n h_ni  和  Σ_n A_n  各除以 σ_ki
        A_H = (A_n.unsqueeze(1) * H).sum(dim=0)            # (L,)
        A_sum = A_n.sum()                                   # scalar
        inv_sigma_k = 1.0 / sigma[k]                       # (L,)

        tsk_num = 2.0 * inv_sigma_k * A_H                  # (L,)
        tsk_den = 2.0 * inv_sigma_k * A_sum                # (L,)

        # Eq. (26) 完整更新
        numerator = fcm_num + tsk_num                       # (L,)
        denominator = fcm_den + tsk_den                     # (L,)
        denominator = torch.clamp(denominator, min=1e-12)

        C_new[k] = numerator / denominator

    return C_new


def update_W(H_bar, mu_bar, Y, K, xi):
    """
    论文 Eq. (28): 更新各规则输出权重
        w_k = (H̄^T Λ_k H̄ + ξI)^{-1} H̄^T Λ_k Y
    其中 Λ_k = diag(μ̄_1k, μ̄_2k, ..., μ̄_Nk) ← 注意是归一化高斯隶属度,不是 u_nk

    返回: W_stacked (K*(L+1), out_dim)
    """
    L_bar = H_bar.shape[1]
    device = H_bar.device
    I = torch.eye(L_bar, device=device)
    W_list = []

    for k in range(K):
        lam_k = mu_bar[:, k].unsqueeze(1)                  # (N, 1)
        A = H_bar.T @ (lam_k * H_bar) + xi * I            # (L_bar, L_bar)
        b = H_bar.T @ (lam_k * Y)                         # (L_bar, out_dim)
        try:
            w_k = torch.linalg.solve(A, b)
        except RuntimeError:
            w_k = torch.linalg.pinv(A) @ b
        W_list.append(w_k)

    return torch.vstack(W_list)


def predict(H_bar, mu_bar, W_stacked, K):
    """
    论文 Eqs. (6)-(8): 计算 BTSK-BLS 预测
        s_nk = √μ̄_nk · h̄_n           (Eq. 6)
        Ŷ = Σ_k S_k w_k = Σ_k √μ̄_nk · (h̄_n w_k)   (Eq. 8)
    """
    L_bar = H_bar.shape[1]
    N = H_bar.shape[0]
    out_dim = W_stacked.shape[1]
    device = H_bar.device

    sqrt_mu = torch.sqrt(torch.clamp(mu_bar, min=1e-12))   # (N, K)
    pred = torch.zeros((N, out_dim), device=device)
    for k in range(K):
        w_k = W_stacked[k * L_bar:(k + 1) * L_bar]
        pred = pred + sqrt_mu[:, k:k + 1] * (H_bar @ w_k)

    return pred


def compute_O(H, H_bar, U, C, W_stacked, Y, mu_bar,
              K, xi, m, lam_poisson, beta_param):
    """
    论文 Eq. (21): BTSK-BLS 代价函数 (MAP 目标, 越大越好)

    J = -1/(2K) Σ_n Σ_k u^m_nk ||h_n - c_k||^2               ← 项1 (FCM)
      + 1/K Σ_n Σ_k (α_k - 1) log u_nk                       ← 项2 (α_k=1 → 0)
      - 1/(2K) Σ_n Σ_k (y_n √μ̄_nk - s_nk w_k)^2             ← 项3 (TSK)
      - ξ/(2K) Σ_k ||w_k||^2                                  ← 项4 (正则)
      + K log λ - Σ_k log k + βN/K                            ← 项5,6,7
    """
    N, L = H.shape
    L_bar = H_bar.shape[1]
    device = H.device

    # 项1: -1/(2K) Σ_n Σ_k u^m_nk ||h_n - c_k||^2
    U_m = U ** m
    dist2 = torch.cdist(H, C, p=2) ** 2                    # (N, K)
    term1 = -(1.0 / (2.0 * K)) * torch.sum(U_m * dist2)

    # 项2: α_k = 1 → 0, 省略

    # 项3: -1/(2K) Σ_n Σ_k μ̄_nk (y_n - h̄_n w_k)^2
    #   因为 (y_n √μ̄_nk - s_nk w_k)^2 = μ̄_nk (y_n - h̄_n w_k)^2
    term3_val = torch.tensor(0.0, device=device)
    for k in range(K):
        w_k = W_stacked[k * L_bar:(k + 1) * L_bar]
        res = Y - H_bar @ w_k                              # (N, out_dim)
        res2 = torch.sum(res ** 2, dim=1)                   # (N,)
        term3_val = term3_val + torch.sum(mu_bar[:, k] * res2)
    term3 = -(1.0 / (2.0 * K)) * term3_val

    # 项4: -ξ/(2K) Σ_k ||w_k||^2
    term4 = -(xi / (2.0 * K)) * torch.sum(W_stacked ** 2)

    # 项5+6: K log λ - Σ_k log k  (= K log λ - log(K!))
    K_t = torch.tensor(float(K), device=device)
    lam_t = torch.tensor(float(lam_poisson), device=device)
    term56 = K * torch.log(lam_t) - torch.lgamma(K_t + 1.0)

    # 项7: βN/K
    term7 = beta_param * N / K

    return (term1 + term3 + term4 + term56 + term7).item()


# =========================================================================
#                          BTSK-BLS 主训练函数
# =========================================================================

def btsk_bls_train(train_x, train_y, test_x, test_y,
                   s, num_fea, num_win, num_enhan,
                   lam_poisson=3.0, beta_param=1.0, xi=1e-6,
                   m=2.0, R=10, t_max=500,
                   epsilon=1e-3, miter=50, theta=5.0,
                   verbose=True):
    """
    Bayesian Takagi-Sugeno-Kang Fuzzy Broad Learning System (BTSK-BLS)
    严格按照论文 Algorithm 1 实现。

    参数 (默认值与论文一致):
        m            模糊指数 = 2
        epsilon      收敛阈值 ε = 1e-3
        miter        连续满足收敛条件的次数 = 50
        t_max        最大迭代次数 = 500
        beta_param   模型稀疏度参数 β ∈ {1, ..., 8}
        R            粒子数 P = 10
        xi           权重正则,接近 0 的小正数
        lam_poisson  Poisson 先验参数 λ
        theta        Laplace 尺度参数 ϑ (论文推荐 5)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_x_t = torch.from_numpy(train_x).float().to(device)
    train_y_t = torch.from_numpy(train_y).float().to(device)
    test_x_t  = torch.from_numpy(test_x).float().to(device)
    test_y_t  = torch.from_numpy(test_y).float().to(device)
    if train_y_t.dim() == 1:
        train_y_t = train_y_t.unsqueeze(1)
    if test_y_t.dim() == 1:
        test_y_t = test_y_t.unsqueeze(1)

    # ==================================================================
    #  Steps 2-4: 生成 BLS 的特征层 F^n、增强层 E^m、隐藏层 H
    # ==================================================================
    scaler_train = StandardScaler()
    train_x_scaled_np = scaler_train.fit_transform(train_x_t.T.cpu().numpy()).T
    train_x_scaled = torch.from_numpy(train_x_scaled_np).float().to(device)

    x1 = torch.hstack([train_x_scaled,
                       0.1 * torch.ones((train_x_scaled.shape[0], 1), device=device)])
    feature_nodes = torch.zeros((train_x_scaled.shape[0], num_win * num_fea),
                                device=device)
    we_list, ps_list = [], []

    for i in range(num_win):
        wr = 2 * torch.rand(train_x_scaled.shape[1] + 1, num_fea, device=device) - 1
        a1 = x1 @ wr
        a1_mapped = 2 * (a1 - torch.min(a1)) / (torch.max(a1) - torch.min(a1) + 1e-10) - 1
        ws = sparse_bls(a1_mapped, x1, 1e-3, 50).T
        we_list.append(ws)

        f1 = x1 @ ws
        ps1 = {'max': torch.max(f1, dim=0)[0], 'min': torch.min(f1, dim=0)[0]}
        f1_mapped = (f1 - ps1['min']) / (ps1['max'] - ps1['min'] + 1e-10)
        ps_list.append(ps1)
        feature_nodes[:, num_fea * i:num_fea * (i + 1)] = f1_mapped

    # 增强层: 正交随机权重
    x2 = torch.hstack([feature_nodes,
                       0.1 * torch.ones((feature_nodes.shape[0], 1), device=device)])
    m_dim = num_fea * num_win + 1
    rand_mat = torch.randn((m_dim, num_enhan), device=device)
    if m_dim >= num_enhan:
        q, _ = torch.linalg.qr(rand_mat, mode='reduced')
        wh = q[:, :num_enhan]
    else:
        q_t, _ = torch.linalg.qr(rand_mat.T, mode='reduced')
        wh = q_t.T

    raw_enh = x2 @ wh
    l2_scale = s / (torch.max(raw_enh) + 1e-10)
    enhancement_nodes = torch.tanh(raw_enh * l2_scale)

    # 隐藏层 H 与增广 H̄ = [1, H]
    H = torch.hstack([feature_nodes, enhancement_nodes])       # (N, L)
    N, L = H.shape
    H_bar = torch.hstack([torch.ones((N, 1), device=device), H])  # (N, L+1)
    L_bar = L + 1

    if verbose:
        print(f'[BTSK-BLS] 隐藏层生成完毕: N={N}, L={L}, L_bar={L_bar}')

    # ==================================================================
    #  Algorithm 1, Step 1: 初始化粒子
    # ==================================================================
    start_time = time.time()
    h_tilde = torch.mean(H, dim=0)  # Eq.(22) 中的位置参数 h̃

    def _init_particle():
        """创建初始粒子: K=1, c_1 由 Eq.(22), U=1, W 由 Eq.(28), O 由 Eq.(21)"""
        K0 = 1
        c0 = sample_laplace_centers(1, h_tilde, theta, device)   # (1, L)

        # K=1 时 u_nk = 1 对所有 n
        U0 = torch.ones((N, K0), device=device)

        # σ: K=1 → 全局方差
        sigma0 = compute_sigma(H, c0, U0, m)                    # (1, L)

        # μ̄: K=1 → μ̄_n1 = 1 对所有 n
        mu_bar0 = torch.ones((N, K0), device=device)

        # W 初始化: Eq.(28), Λ_1 = I (因 μ̄_n1 = 1)
        W0 = update_W(H_bar, mu_bar0, train_y_t, K0, xi)        # (L+1, out_dim)

        # O 由 Eq.(21)
        O0 = compute_O(H, H_bar, U0, c0, W0, train_y_t,
                       mu_bar0, K0, xi, m, lam_poisson, beta_param)

        return {'K': K0, 'C': c0, 'U': U0, 'W': W0,
                'sigma': sigma0, 'mu_bar': mu_bar0, 'O': O0}

    p0 = _init_particle()
    particles = [copy.deepcopy(p0) for _ in range(R)]
    CA = {1: copy.deepcopy(p0)}
    t = 0
    prev_max_O = p0['O']
    conv_count = 0

    # ==================================================================
    #  Algorithm 1, Steps 5-21: 粒子滤波主循环
    # ==================================================================
    while t < t_max and conv_count < miter:
        new_particles = []

        for r in range(R):
            # ---- Step 7: 采样新的规则数 K ----
            cur_K = particles[r]['K']
            K_new = int(torch.poisson(
                torch.tensor(float(cur_K), device=device)).item())
            K_new = max(1, K_new)

            # ---- Steps 8-9: 调整聚类中心 ----
            cur_C = particles[r]['C']
            if K_new <= cur_K:
                idx = torch.randperm(cur_K, device=device)[:K_new]
                new_C = cur_C[idx].clone()
            else:
                added = sample_laplace_centers(
                    K_new - cur_K, h_tilde, theta, device)
                new_C = torch.vstack([cur_C, added])

            # ---- Step 10: 更新 K ----
            # (K_new 已确定)

            # ---- Step 11: Update U by Eq. (24) ----
            new_U = update_U(H, new_C, m)

            # ---- 计算 σ 和 μ̄ (为 Step 12 的 Eq.26 做准备) ----
            new_sigma = compute_sigma(H, new_C, new_U, m)
            new_mu_bar, _ = compute_gaussian_membership(H, new_C, new_sigma)

            # ---- Step 12: Update C by Eq. (26) ----
            # 需要 W_old (来自上一轮粒子)
            W_old = particles[r]['W']
            # 如果 K 变了, W_old 的维度不匹配 → 需要适配
            old_K = particles[r]['K']
            if K_new != old_K:
                # 用当前 μ̄ 快速计算一个临时 W 以便 Eq.(26) 使用
                W_old = update_W(H_bar, new_mu_bar, train_y_t, K_new, xi)

            new_C = update_C(H, H_bar, new_C, new_U, W_old,
                             train_y_t, new_sigma, new_mu_bar, K_new, m)

            # ---- C 更新后, 重新计算 U, σ, μ̄ ----
            new_U = update_U(H, new_C, m)
            new_sigma = compute_sigma(H, new_C, new_U, m)
            new_mu_bar, _ = compute_gaussian_membership(H, new_C, new_sigma)

            # ---- Step 13: Update W by Eq. (28) ----
            new_W = update_W(H_bar, new_mu_bar, train_y_t, K_new, xi)

            # ---- Step 14: Compute O by Eq. (21) ----
            new_O = compute_O(H, H_bar, new_U, new_C, new_W,
                              train_y_t, new_mu_bar,
                              K_new, xi, m, lam_poisson, beta_param)

            particle = {'K': K_new, 'C': new_C, 'U': new_U,
                        'W': new_W, 'sigma': new_sigma,
                        'mu_bar': new_mu_bar, 'O': new_O}
            new_particles.append(particle)

            # ---- Step 15: 更新候选集 CA[K] by Eq. (29) ----
            if (K_new not in CA) or (new_O > CA[K_new]['O']):
                CA[K_new] = copy.deepcopy(particle)

        # ---- Step 17: PS = {R, CA} by Eq. (30) ----
        PS = new_particles + list(CA.values())

        # ---- Step 18: 重要性权重 by Eq. (31) ----
        O_vec = torch.tensor([p['O'] for p in PS], device=device)
        O_max = torch.max(O_vec)
        w = torch.exp(O_vec - O_max)
        w = w / (torch.sum(w) + 1e-12)

        # ---- Step 19: 带放回重采样 R 个粒子 ----
        idx = torch.multinomial(w, R, replacement=True)
        particles = [copy.deepcopy(PS[i.item()]) for i in idx]

        # ---- 收敛判据 ----
        cur_max_O = max(p['O'] for p in PS)
        if abs(cur_max_O - prev_max_O) < epsilon:
            conv_count += 1
        else:
            conv_count = 0
        prev_max_O = cur_max_O
        t += 1

        if verbose and (t % 20 == 0 or t == 1):
            best_K_iter = max(CA.keys(), key=lambda k: CA[k]['O'])
            print(f'  iter {t:4d}: max O = {cur_max_O:.4f}, '
                  f'best K = {best_K_iter}, |PS| = {len(PS)}, '
                  f'收敛计数 = {conv_count}')

    # ==================================================================
    #  Steps 22-23: 从 CA 选 O 最大的粒子
    # ==================================================================
    best_K = max(CA.keys(), key=lambda k: CA[k]['O'])
    best = CA[best_K]
    K_f = best['K']
    C_f = best['C']
    U_f = best['U']
    W_f = best['W']
    sigma_f = best['sigma']
    mu_bar_f = best['mu_bar']

    training_time = time.time() - start_time
    if verbose:
        print(f'[BTSK-BLS] 训练完成, 用时 {training_time:.4f} s, '
              f'最终规则数 K = {K_f}')

    # 训练 RMSE
    train_pred = predict(H_bar, mu_bar_f, W_f, K_f)
    train_err = torch.sqrt(torch.mean((train_pred - train_y_t) ** 2))
    if verbose:
        print(f'[BTSK-BLS] 训练 RMSE = {train_err.item():.4e}')

    # ==================================================================
    #  测试阶段
    # ==================================================================
    start_time_test = time.time()

    scaler_test = StandardScaler()
    test_x_scaled_np = scaler_test.fit_transform(test_x_t.T.cpu().numpy()).T
    test_x_scaled = torch.from_numpy(test_x_scaled_np).float().to(device)

    xx1 = torch.hstack([test_x_scaled,
                        0.1 * torch.ones((test_x_scaled.shape[0], 1), device=device)])
    feature_nodes_test = torch.zeros((test_x_scaled.shape[0], num_fea * num_win),
                                     device=device)
    for i in range(num_win):
        ws = we_list[i]
        ps1 = ps_list[i]
        f2 = xx1 @ ws
        f2_mapped = (f2 - ps1['min']) / (ps1['max'] - ps1['min'] + 1e-10)
        feature_nodes_test[:, num_fea * i:num_fea * (i + 1)] = f2_mapped

    xx2 = torch.hstack([feature_nodes_test,
                        0.1 * torch.ones((feature_nodes_test.shape[0], 1),
                                         device=device)])
    enhancement_nodes_test = torch.tanh(xx2 @ wh * l2_scale)

    H_test = torch.hstack([feature_nodes_test, enhancement_nodes_test])
    H_bar_test = torch.hstack([torch.ones((H_test.shape[0], 1), device=device),
                                H_test])

    # 用训练得到的 C_f, sigma_f 推断测试集 μ̄
    mu_bar_test, _ = compute_gaussian_membership(H_test, C_f, sigma_f)

    test_pred = predict(H_bar_test, mu_bar_test, W_f, K_f)
    test_err = torch.sqrt(torch.mean((test_pred - test_y_t) ** 2))
    testing_time = time.time() - start_time_test

    if verbose:
        print(f'[BTSK-BLS] 测试完成, 用时 {testing_time:.4f} s')
        print(f'[BTSK-BLS] 测试 RMSE = {test_err.item():.4e}')

    return (test_pred.cpu().numpy(),
            training_time, testing_time,
            train_err.cpu().numpy(),
            test_err.cpu().numpy(),
            K_f)