# 快速开始指南 (Quick Start Guide)

## 原子团簇轮廓提取系统

### 核心功能
✅ 自动识别原子团簇
✅ 提取精确的外轮廓（红色线标注）
✅ 导出所有轮廓点坐标
✅ 生成单个团簇的裁剪图像

---

## 3分钟快速上手

### 1️⃣ 激活环境
```bash
source /home/jicwang/miniconda3/bin/activate atomsegnet
```

### 2️⃣ 运行提取脚本
```bash
python3 extract_atom_clusters.py <输入目录> <输出目录>
```

**示例**：
```bash
python3 extract_atom_clusters.py blured_1742/ my_output/
```

### 3️⃣ 查看结果
```bash
ls -lh my_output/
cat my_output/*_boundary_0.txt | head  # 查看坐标
```

---

## 示例工作流

### 快速演示（仅处理3张图像）
```bash
source /home/jicwang/miniconda3/bin/activate atomsegnet

python3 extract_atom_clusters.py blured_1742/ test_output/ --head 3

# 查看输出
ls test_output/
cat test_output/0000_denoise_blur_boundary_0.txt | head -10
```

### 完整处理（所有图像）
```bash
source /home/jicwang/miniconda3/bin/activate atomsegnet

python3 extract_atom_clusters.py blured_1742/ final_output/ --sigma 3.0
```

### 详细演示（查看处理步骤）
```bash
source /home/jicwang/miniconda3/bin/activate atomsegnet

python3 demo_contour_extraction.py blured_1742/0000_denoise_blur.png demo_output/ --sigma 3.0
```

这会生成：
- `01_original.png` - 原始图像
- `02_blurred.png` - 模糊处理后
- `03_mask.png` - 分割掩码
- `04_contours.png` - 轮廓叠加
- `06_cluster_0_crop.png` - **红色轮廓团簇** ⭐

---

## 输出文件说明

对于每个处理的图像，你会得到：

| 文件名 | 说明 | 用途 |
|-------|------|------|
| `*_boundary_coords.txt` | 所有团簇的坐标合集 | 数据分析 |
| `*_boundary_0.txt` | 第0个团簇的边界坐标 | 单个团簇数据 |
| `*_boundary_1.txt` | 第1个团簇的边界坐标 | 单个团簇数据 |
| `*_cluster_0.png` | **红色轮廓的团簇0** | 可视化检查 ⭐ |
| `*_cluster_1.png` | **红色轮廓的团簇1** | 可视化检查 ⭐ |

---

## 坐标文件格式

每行表示轮廓上的一个点：
```
x_coordinate,y_coordinate
371.6000,706.2000
372.1000,705.5000
372.6000,704.8000
...
```

### 在 Python 中使用坐标

```python
import numpy as np

# 读取坐标
coords = np.loadtxt("output/0000_denoise_blur_boundary_0.txt", delimiter=",")

# 计算中心
center_x = coords[:, 0].mean()
center_y = coords[:, 1].mean()
print(f"团簇中心: ({center_x}, {center_y})")

# 计算面积（使用鞋带公式）
x = coords[:, 0]
y = coords[:, 1]
area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
print(f"面积: {area:.1f} px²")

# 计算周长
perimeter = np.sum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
print(f"周长: {perimeter:.1f} px")
```

---

## 常见参数调整

### 🔍 检测质量不好？

**问题：检测到太多噪声**
```bash
python3 extract_atom_clusters.py input/ output/ \
  --min-obj-area 2000  # 增加最小面积
```

**问题：漏掉了小团簇**
```bash
python3 extract_atom_clusters.py input/ output/ \
  --min-obj-area 500   # 减少最小面积
```

**问题：轮廓不清晰**
```bash
python3 extract_atom_clusters.py input/ output/ \
  --sigma 2.5          # 减少模糊（更清晰）
  --local-offset 8.0   # 调整阈值
```

**问题：接触的团簇未分离**
- 默认启用了 watershed 分割，应该能分离
- 如需禁用，使用：`--no-split-touching`

---

## 文件结构

```
AtomSegNet/
├── blur_filter_points.py           ✨ 核心处理逻辑
├── extract_atom_clusters.py        ✨ 便捷脚本（推荐使用）
├── demo_contour_extraction.py      ✨ 演示脚本
├── ATOM_CLUSTER_EXTRACTION_GUIDE.md  📖 详细文档
├── QUICKSTART.md                   📖 本文件
│
└── blured_1742/                    📁 输入数据
    ├── 0000_denoise_blur.png
    ├── 0001_denoise_blur.png
    └── ...

my_output/                           📁 输出目录
├── 0000_denoise_blur_boundary_0.txt
├── 0000_denoise_blur_cluster_0.png  ⭐ 红色轮廓
├── 0001_denoise_blur_boundary_0.txt
├── 0001_denoise_blur_cluster_0.png  ⭐ 红色轮廓
└── ...
```

---

## 关键改进

本系统相比原始版本的改进：

✅ **轮廓平滑** - 使用中值滤波器平滑边界，减少锯齿
✅ **红色标记** - 清晰的红色轮廓线（RGB 255,0,0）
✅ **单个团簇提取** - 自动为每个团簇生成裁剪图像
✅ **坐标导出** - 完整的边界点坐标（CSV格式）
✅ **自动分割** - Watershed 算法自动分离接触的团簇

---

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| `No module named 'skimage'` | 使用 `source activate atomsegnet` 激活环境 |
| 检测不到团簇 | 调整 `--sigma` 和 `--min-obj-area` 参数 |
| 轮廓不准确 | 尝试不同的 `--sigma` 值（1.0-5.0） |
| 文件权限错误 | 确保输出目录有写入权限 |
| 输出图像太大 | 这是正常的，使用 `convert` 或 PIL 压缩 |

---

## 下一步

1. **处理你的数据**
   ```bash
   python3 extract_atom_clusters.py your_images/ your_output/
   ```

2. **集成到工作流**
   - 将坐标导入分析脚本
   - 使用图像进行质量检查

3. **批量处理**
   ```bash
   for dir in dataset_*; do
     python3 extract_atom_clusters.py "$dir" "output_$dir"
   done
   ```

4. **参考完整文档**
   - 查看 `ATOM_CLUSTER_EXTRACTION_GUIDE.md` 了解所有参数

---

## 获取帮助

```bash
# 查看所有参数
python3 extract_atom_clusters.py --help

# 查看演示脚本帮助
python3 demo_contour_extraction.py --help
```

---

**现在就开始吧！** 🚀

```bash
source /home/jicwang/miniconda3/bin/activate atomsegnet
python3 extract_atom_clusters.py blured_1742/ my_output/ --head 5
```
