# 指定环境运行记录

TianLab2，RTX 3090 GPU 0，解释器 /data6/chensq/miniconda3/envs/insid3/bin/python。官方仓库 /data6/chensq/INSID3，main，0c165a10cf52ab91f335883d06260de86854adbe。DINOv3 6876159a11b4df116f30f667f8c9888617df0751；Large/16 权重 /data6/chensq/INSID3/pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth，SHA-256 8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035。数据根 /data6/chensq/datasets，Pascal-Part/VOCdevkit/VOC2010，fold 0 animals，31 类，1-shot，seed 0，默认 2500 对。原装 torchvision 0.29.2+cu132 与 torch 2.14.1+cu126 不匹配。本次在实验目录加载官方 torchvision 0.29.1+cu126。环境其余推理依赖沿用 insid3。论文参考环境为 PyTorch 2.7.1。

每对调用原版模型 set_reference、set_target、segment，图像 1024、SVD 500、tau 0.6、合并阈值 0.2、GPU CRF 640。CRF 源码为 13a123f7cd3ea1f975e6c483b1e7c52112d7c151。针对 Torch ABI 使用 ctypes 将 contiguous float32 张量地址交给原 CUDA computeKernelGPU 和 latticeFilterGPU（sm_86），使用官方二分类空间与双边核、Frank–Wolfe 求解和边界带细化。每对直接运行至 CRF 后预测。原版 INSID3 工作区未改。无诊断状态时可正常调用原版 segment；有诊断状态时只截取中间值。源码调用链及指标口径见 source-audit.md，那里记录的 sam2coco 数字属于早期运行。

启动目录 /data6/chensq/UFSAM2/insid3_round1。实际命令 bash run_insid3.sh insid3_run01，复跑时使用新目录名。前 8 对分别用开启及关闭诊断状态的 segment 计算并比对掩码，随后完整清单重新推理。full/*.npz 含每对全部轻量诊断数组，shape 是评测网格，labels 为原生 64×64 离散簇标号。支持与查询、对象框、类、顺序和 seed 见 full/manifest.jsonl。预测阶段只输入支持真值，查询真值仅在预测后参与评估。

首批 8 对的原生聚合掩码和 CRF 后最终预测，与先前另一套 Torch 环境下使用官方 CRF 扩展的独立核对也均为 0 像素差异。
