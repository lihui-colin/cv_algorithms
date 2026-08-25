# Tiny RoMa 模板匹配

`roma_template_match.py` 使用 RoMa 仓库的 `tiny_roma_v1_outdoor`，在模板图和目标图之间计算稠密对应关系，并用 RANSAC 单应性把模板四个角投影到目标图中。

## Conda Python 3.12

如果已有名为 `py312` 的 Conda 环境，直接使用：

```bash
conda activate py312
pip install -e ./RoMa opencv-python
```

也可以用项目里的 `environment.yml` 创建一个独立环境：

```bash
conda env create -f environment.yml
conda activate roma-template-py312
```

如果不使用 Conda 的 PyTorch（例如 CPU-only 环境），可以改为：

```bash
conda create -n roma-template-py312 python=3.12 pip -y
conda activate roma-template-py312
pip install torch torchvision
pip install -e ./RoMa opencv-python
```

首次运行会通过 PyTorch Hub 下载 Tiny RoMa 和 XFeat 权重，需要网络连接；权重会缓存到 PyTorch 默认缓存目录。

## 运行

```bash
python roma_template_match.py \
  --template path/to/template.jpg \
  --target path/to/scene.jpg \
  --output results/scene_match.jpg \
  --json-output results/scene_match.json
```

输出内容：

- `scene_match.jpg`：目标图上绘制的模板四边形（顶点编号 0→3）和内点统计；
- `scene_match.json`：单应矩阵、投影角点、匹配数、内点率和重投影误差。

可用 `--device cpu|cuda|mps|auto` 选择设备。默认 `auto` 优先 CUDA，其次 MPS，最后 CPU。
