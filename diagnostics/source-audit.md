# INSID3 第一轮部件诊断 · 服务器 TianLab2

原仓库 /data6/chensq/INSID3，main，0c165a10cf52ab91f335883d06260de86854adbe。隔离的诊断目录位于 /data6/chensq/UFSAM2/insid3_round1。GPU 0，RTX 3090 24 GiB。sam2coco 环境 Python 3.10，torch 2.5.1+cu121，torchvision 0.20.1+cu121，scikit-learn 1.7.2。insid3 环境 torchvision 导入报 nms 算子缺失。DINOv3 骨干源码 6876159a11b4df116f30f667f8c9888617df0751。模型 pretrain/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth，SHA-256 8aa4cbddda325040fc78db2c272754af6ebe8ff2c55f6ec4f1964d8890f66035。CRF 源码 13a123f7cd3ea1f975e6c483b1e7c52112d7c151。官方默认超参 1024、SVD 500、tau 0.6、merge 0.2、CRF 640。默认命令行未开启 CRF。

数据根 /data6/chensq/datasets。datasets/pascal_part.py 的 sample_episode 从同一 object+part 的 val 池抽取不同图像，选定对象实例并合并该实例同名部件的 RLE，支持图和查询图各自按对应对象框裁剪。fold 0 animals 共 31 类，seed 0，num_workers 0，2500 对，按类轮转。manifest0/manifest.jsonl 保存采样顺序、支持与查询 ID、实例框和类别。图像与 mask 的空间尺寸逐例断言。该数据集无单独 ignore 标记，背景参与并集。

models/insid3.py 的 predict_mask（第 120 行附近）从归一化的原始查询特征聚类。utils/clustering.py 的 agglomerative_clustering 使用余弦距离和 average linkage，距离阈值 1−tau。_locate_candidates（第 272 行附近）取去偏前向匹配与支持真值反向投票的交集。_seed_and_aggregate（第 310 行附近）选择候选交叠簇里原型匹配分数最高者，再将簇内平均跨图相似度、簇到种子的图内相似度及候选占簇面积比例相乘；种子的面积权重置 1，最终仍须过阈值。候选以外的簇分数为 0。候选为空时官方跳过聚类，导出器只为离线 A 补算同一原始特征的聚类。

A/B 将 64×64 离散标签最近邻映射至 1024×1024。按有效像素前景纯度排序，扫描所有前缀，空集得分 0；对 32 种 5 像素真值、3 种候选及 2 种 ignore 共 192 个组合进行穷举核对。A/B 汇总为逐 episode 最优解的按类累计交并量。C 为 raw mask 按官方 bilinear、align_corners=False 放大至评测网格并以 >0.5 二值化，因此 B−C 除了选择和聚合也包含栅格化差异。D 经官方 _finalize_mask → utils/refinement.py 的 crf_refine（原始 mask 先放大至 640，再经带状 GPU CRF，最终放大至 1024）。GT 与 inference_segmentation.py 一样按 nearest 缩放。utils/metrics.py 的 AverageMeter 对每类累计前景交/并，分母至少 1，再等权平均。逐 episode 分数另列。

每个索引的 npz 保留原生簇标签、候选 ID 和种子匹配分数、种子 ID、实际合并 ID、原生聚合 mask、评测网格和 packbits 压缩的 GT/预测，解码使用 np.unpackbits(...).reshape(shape)。无 CRF 的 baseline0 中 final/D 字段只是同一无 CRF 输出。后处理重放时 raw 数据来自 baseline0，真正的 D 单独保存。

为恢复官方 CRF，下载了 CUDA 12.1 编译器和 CCCL 头文件，用 GCC 9 将扩展编为 RTX 3090 的 sm_86，运行库保存在 crf_runtime/。GPU 计算路径仍为官方 GPU Permutohedral 与 Frank–Wolfe 实现。完整运行先按原始配置输出 C 与原生簇，再从对应查询 RGB 裁剪和 64×64 原生 mask 逐例调用同一 _finalize_mask 产生 D；8 对另跑端到端预测核对。上界算法未向预测过程提供查询真值。

实际启动（工作目录为本诊断目录，解释器为 /data6/chensq/miniconda3/envs/sam2coco/bin/python）：
```bash
python run.py manifest0 --manifest-only
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python run.py baseline0 --no-crf
PYTHONPATH=/tmp/insid3-crf-official/src CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python refine.py baseline0 final0
PYTHONPATH="$PWD/crf_runtime" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 python diagnose.py smoke_verified --limit 8
```
本机端到端复跑命令为 `bash /data6/chensq/UFSAM2/insid3_round1/run_round.sh rerun_01`；参数必须是新的目录名。论文的 50.5 作为跨 fold 参考。
