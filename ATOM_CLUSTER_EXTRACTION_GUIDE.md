# 原子团簇轮廓提取指南 (Atom Cluster Contour Extraction Guide)

## 概述

本指南说明如何使用增强的 `blur_filter_points.py` 脚本来自动识别和提取原子团簇的外轮廓，并将其坐标导出为文本文件和可视化图像。

## 主要功能

### 1. **自动轮廓检测**
- 使用高斯模糊和自适应阈值分割
- 通过形态学操作（开闭运算）清理掩码
- 使用分水岭算法分离接触的团簇
- 轮廓平滑处理以获得更好的视觉效果

### 2. **输出文件**

对于每个输入图像，脚本生成：

#### a) **坐标文件**
- `{base_name}_boundary_coords.txt` - 所有团簇的合并坐标
- `{base_name}_boundary_{idx}.txt` - 单个团簇的边界坐标（CSV格式：x,y）

#### b) **可视化图像**
- `{base_name}_cluster_{idx}.png` - 单个团簇裁剪图像（带红色轮廓）
- `{base_name}_origin_gaussian_blur.png` - 原始图像的完整轮廓叠加

#### c) **调试文件** (可选)
- `{base_name}_blur.png` - 高斯模糊后的图像
- `{base_name}_mask.png` - 二值化掩码

## 使用方法

### 方法 1：使用专用脚本（推荐）

```bash
source /home/jicwang/miniconda3/bin/activate atomsegnet

python3 extract_atom_clusters.py <输入目录> <输出目录> [选项]
```

**示例**：
```bash
python3 extract_atom_clusters.py blured_1742/ output_clusters/ \
  --sigma 3.0 \
  --max-contours 2 \
  --head 10
```

**参数说明**：
- `input_dir` - 包含要处理的图像的目录
- `output_dir` - 输出文件保存目录
- `--pattern` - 输入文件匹配模式 (默认: `*.png`)
- `--sigma` - 高斯模糊标准差 (默认: 3.0)
- `--threshold-scale` - Otsu阈值缩放因子 (默认: 0.95)
- `--max-contours` - 每个图像最多保留的轮廓数 (默认: 2)
- `--min-obj-area` - 最小连通分量面积 (默认: 1200)
- `--head` - 仅处理前N个图像

### 方法 2：使用原始 blur_filter_points.py

```bash
source /home/jicwang/miniconda3/bin/activate atomsegnet

python3 blur_filter_points.py <检测文件目录> <原始图像目录> \
  --save-blur-dir <输出目录> \
  --sigma 3.0 \
  --max-contours 2
```

## 输出文件格式

### 坐标文件格式 (x, y)

```
371.6000,706.2000
372.1000,705.5000
372.6000,704.8000
...
```

- **x** - 图像中的水平位置（像素）
- **y** - 图像中的垂直位置（像素）
- 点按轮廓顺序排列，形成闭合多边形

### 在Python中读取坐标

```python
import numpy as np

# 读取坐标文件
coords = np.loadtxt("0000_denoise_blur_boundary_0.txt", delimiter=",")
print(f"轮廓点数: {len(coords)}")
print(f"中心点: ({coords[:, 0].mean():.1f}, {coords[:, 1].mean():.1f})")

# 计算多边形面积
x = coords[:, 0]
y = coords[:, 1]
area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
print(f"团簇面积: {area:.1f}")
```

## 参数调整建议

### 检测大型团簇
```bash
python3 extract_atom_clusters.py input/ output/ \
  --sigma 4.0 \
  --min-obj-area 800 \
  --local-offset 3.0
```

### 检测小型团簇
```bash
python3 extract_atom_clusters.py input/ output/ \
  --sigma 2.0 \
  --min-obj-area 300 \
  --local-offset 8.0
```

### 分离接触的团簇
```bash
python3 extract_atom_clusters.py input/ output/ \
  --max-contours 3 \
  --no-split-touching  # 禁用此参数以启用分水岭分割
```

## 增强功能详解

### 1. 轮廓平滑 (Contour Smoothing)
- 使用中值滤波器平滑边界点
- 减少锯齿状边缘，提高可视效果
- 在 `smooth_contour()` 函数中实现

### 2. 个别团簇提取
- `save_individual_contour_images()` 为每个团簇生成裁剪图像
- 自动计算包围盒，减少背景区域
- 在每个裁剪中显示红色轮廓线

### 3. 红色轮廓显示
- 轮廓颜色：RGB(255, 0, 0) - 红色
- 线宽：2-3像素
- 自动闭合（首尾相连）

## 故障排除

### 问题：检测到的团簇过多或过少

**解决方案**：调整 `--min-obj-area` 参数
```bash
# 如果有很多小的噪声：增加此值
python3 extract_atom_clusters.py ... --min-obj-area 2000

# 如果漏掉了小的团簇：减少此值
python3 extract_atom_clusters.py ... --min-obj-area 500
```

### 问题：轮廓不准确或分割不好

**解决方案**：调整 `--sigma` 和 `--local-offset`
```bash
# 增加模糊以合并邻近像素
python3 extract_atom_clusters.py ... --sigma 4.0 --local-offset 3.0

# 减少模糊以获得更细致的边界
python3 extract_atom_clusters.py ... --sigma 2.0 --local-offset 8.0
```

### 问题：接触的团簇未分离

**解决方案**：确保未禁用 watershed 分割
```bash
# 默认启用（推荐）
python3 extract_atom_clusters.py ... # 不使用 --no-split-touching

# 或显式启用
python3 extract_atom_clusters.py ... --no-split-touching 不使用此标志
```

## 工作流示例

完整的工作流程：

```bash
source /home/jicwang/miniconda3/bin/activate atomsegnet

# 第1步：处理第一批图像（测试）
python3 extract_atom_clusters.py blured_1742/ test_output/ \
  --head 5 \
  --sigma 3.0

# 第2步：检查输出
ls -lh test_output/
cat test_output/0000_denoise_blur_boundary_0.txt | head -10

# 第3步：调整参数（如需要）
python3 extract_atom_clusters.py blured_1742/ output_clusters/ \
  --head 50 \
  --sigma 3.5 \
  --min-obj-area 1500

# 第4步：处理所有文件
python3 extract_atom_clusters.py blured_1742/ output_all/
```

## 集成到现有工作流

### 与 AtomSegNet 模型结合使用

```bash
# 1. 运行 AtomSegNet 模型得到检测点
python3 batch_run_atomsegnet.py ...

# 2. 应用轮廓提取
python3 extract_atom_clusters.py detection_output/ contour_output/

# 3. 可选：过滤点
python3 blur_filter_points.py detection_output/ original_images/ \
  --save-blur-dir filtered_output/
```

## 文件结构

```
project/
├── blur_filter_points.py          # 核心轮廓提取逻辑
├── extract_atom_clusters.py       # 便捷脚本
├── ATOM_CLUSTER_EXTRACTION_GUIDE.md  # 本文档
├── blured_1742/                   # 输入图像
│   ├── 0000_denoise_blur.png
│   ├── 0001_denoise_blur.png
│   └── ...
└── output_clusters/               # 输出目录
    ├── 0000_denoise_blur_cluster_0.png
    ├── 0000_denoise_blur_boundary_0.txt
    ├── 0000_denoise_blur_boundary_coords.txt
    └── ...
```

## API 参考

### 关键函数

#### `extract_topk_contours(mask, k=2, smooth=True)`
从二值掩码提取前k个最大轮廓。

```python
from blur_filter_points import extract_topk_contours
import numpy as np

mask = binary_mask  # np.ndarray of bool/uint8
contours = extract_topk_contours(mask, k=2, smooth=True)
# 返回: list[np.ndarray]，每个数组形状为 (N, 2)，格式为 (x, y)
```

#### `save_boundary_coords_multi(output_dir, base_name, contours_xy)`
将轮廓坐标保存到文件。

```python
from pathlib import Path
from blur_filter_points import save_boundary_coords_multi

output_dir = Path("output/")
contours_xy = [np.array([[x1, y1], ...]), ...]
save_boundary_coords_multi(output_dir, "image_001", contours_xy)
# 生成:
#   image_001_boundary_coords.txt
#   image_001_boundary_0.txt
#   image_001_boundary_1.txt
#   ...
```

## 性能优化

对于大量图像，建议：

1. **批处理** - 使用 `--head` 和 `--tail` 参数分批处理
2. **参数预优化** - 先在小数据集上调整参数
3. **并行处理** - 可修改脚本以支持多进程（使用 `multiprocessing`）

## 许可和引用

本脚本基于：
- scikit-image 的轮廓检测
- scipy 的形态学和分水岭算法
- Pillow 的图像处理

---

**最后更新**: 2025-10-28
**版本**: 1.0
