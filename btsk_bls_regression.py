import numpy as np
from scipy.io import loadmat
import pandas as pd
from btsk_bls_train import btsk_bls_train
import math


# ===================== 读取数据 =====================
try:
    data = loadmat('./traindata/Sunspots/sunspot06.mat')
    train_x = data['train_x']
    train_y = data['train_y']
    test_x  = data['test_x']
    test_y  = data['test_y']
except FileNotFoundError:
    print("错误:sunspot06.mat 未找到,请提供数据文件。")
    exit()

assert train_x.dtype == np.float64, 'train_x 必须是浮点型'
assert test_x.dtype  == np.float64, 'test_x 必须是浮点型'


Nsa = train_x.shape[0]  # 训练集样本数量
lam_poisson = math.log(Nsa)  # 自然对数 ln(Nsa)

# ===================== BTSK-BLS 超参数(论文设定) =====================
# 模糊指数 m = 2
# 收敛阈值 ε = 1e-3
# 连续收敛计数 miter = 50
# 最大迭代次数 t_max = 500
# 模型稀疏度参数 β = 3
# 粒子数 P = 10
BTSK_PARAMS = dict(
    m           = 2.0,
    epsilon     = 1e-3,
    miter       = 50,
    t_max       = 500,
    R           = 10,          # 粒子数
    xi          = 1e-6,        # 接近 0 的小正数
    lam_poisson = lam_poisson,         # Poisson 先验 λ,可按数据集微调
    theta       = 5.0,         # Laplace 尺度参数 ϑ (论文推荐)
    verbose     = True,       # 网格搜索时关掉中间打印
)

# BETA_GRID = [1, 2, 3, 4, 5, 6, 7, 8]   # 稀疏度参数搜索
BETA_GRID = [3]   # 粗略搜索范围,后续可微调
# BLS 骨架超参:由于 BTSK-BLS 的粒子滤波开销较大,
# 建议把网格范围缩小。
s = 0.8
best = np.inf
result = []


# ===================== 网格搜索 =====================
for beta_param in BETA_GRID:
    for num_fea in range(1, 21):             #  1..20
        for num_win in range(1, 21):         #  1..20
            for num_enhan in range(2, 201, 2):   #  2..200
                print(f'β = {beta_param}, 特征节点数 = {num_fea}, '
                      f'窗口数 = {num_win}, 增强节点数 = {num_enhan}')

                try:
                    (_, training_time, testing_time,
                     train_err, test_err, K_final) = btsk_bls_train(
                        train_x, train_y, test_x, test_y,
                        s, num_fea, num_win, num_enhan,
                        beta_param=beta_param,
                        **BTSK_PARAMS,
                    )
                except Exception as e:
                    print(f"训练过程中发生错误: {e}")
                    continue

                total_time = training_time + testing_time
                result.append([beta_param, num_fea, num_win, num_enhan,
                               K_final, test_err, train_err])

                # 实时保存最优结果
                if best > test_err:
                    best = test_err
                    best_dict = {
                        'Beta'         : [beta_param],
                        'Test_ERR'     : [test_err],
                        'Train_ERR'    : [train_err],
                        'K_rules'      : [K_final],
                        'NumFea'       : [num_fea],
                        'NumWin'       : [num_win],
                        'NumEnhan'     : [num_enhan],
                        'Total_Time'   : [total_time],
                        'Training_Time': [training_time],
                        'Testing_Time' : [testing_time],
                    }
                    pd.DataFrame(best_dict).to_excel(
                        'best_result_sunspot06.xlsx', index=False
                    )
                    print(f"✓ 最佳结果已保存: test_err = {test_err:.4e}, K = {K_final}")

# 全部结果一并保存,方便后续分析
df_all = pd.DataFrame(result, columns=[
    'Beta', 'NumFea', 'NumWin', 'NumEnhan',
    'K_rules', 'Test_ERR', 'Train_ERR'
])
df_all.to_excel('all_results_sunspot06.xlsx', index=False)

print(f"\n网格搜索完成。最终找到的最佳测试误差: {best}")
