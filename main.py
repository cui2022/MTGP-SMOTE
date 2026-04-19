import time
import numpy as np
import example_nparray

# hof[0]中存储了最棒的结果将其保存
for i in range(0, 30):
    print(i)
    b_time = time.time()
    result, result_f, less_class, Train = example_nparray.main(i)
    n_time = time.time()
    # 保存产生的样本与表达式
    example_nparray.write_excel(result, result_f, i, b_time, n_time)
    # 保存可用数据
    # 先添加标签列
    labels_np = np.full((result.shape[0], 1), less_class)
    result = np.hstack((result, labels_np))
    # 合并gp生成的样本与原始训练数据
    result = np.vstack((result, Train))
    # 保存
    example_nparray.writeinto_excel(result, i)