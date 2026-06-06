# LoGeR 新增功能开发说明

本文档总结当前项目在原始 LoGeR 功能上新增的工程能力，重点覆盖视频抽帧、分组重建、多卡并行、稳定参考系、chunk 可视化，以及基于点云的稀疏 occupancy 计算。目标是让后续开发者能快速理解数据流、运行方式和关键代码接口。

## 1. 总体设计逻辑

新增流程围绕 `/data/xby/YTB` 数据组织设计：

```text
video files
  -> vid2img.py
  -> /data/xby/YTB/<city>/img/<video_id>/<video_id>_000/*.jpg
  -> parallel_traj_recon.py
  -> /data/xby/YTB/<city>/traj/<video_id>/<video_id>_000/{points.pt,camera_poses.npz,conf.npz,depth_maps.npz,meta.yaml}
  -> vis_recon.py / vis_recon_chunk.py / collision_occupancy.py
```

核心原则：

- 一个原始视频会被整理到一个视频级目录，例如 `BV1xxx/BV1xxx_000`、`BV1xxx/BV1xxx_001`。
- 每个 `_000`、`_001` 目录是一个独立重建 chunk，避免超长视频一次性推理。
- `camera_poses.npz`、`conf.npz`、`depth_maps.npz` 保存全部原始帧输出。
- `points.pt` 只保存按 `pt_strides` 抽帧且经过置信度过滤的点云，避免磁盘体积过大。
- 可视化默认 lazy load chunk，避免一次加载全部视频导致 viser 卡死。
- occupancy 默认保存稀疏体素索引，服务后续程序计算，而不是保存 dense grid 或 CSV。

## 2. 数据目录约定

推荐目录：

```text
/data/xby/YTB/<city>/
  vid/                         # 原始视频，可按实际路径指定
  img/
    <video_id>/
      <video_id>_000/
      <video_id>_001/
  traj/
    <video_id>/
      <video_id>_000/
      <video_id>_001/
```

`img` 和 `traj` 保持同构目录结构。`parallel_traj_recon.py` 会递归寻找包含图片的 leaf folder，并把相对路径映射到输出目录。

## 3. 视频抽帧与分组

入口脚本：`vid2img.py`

主要功能：

- 从视频按 `--target_fps` 抽帧。
- 按 `--target_frames` 将图片切成多个 chunk 子目录。
- 使用 `--min_last_group_frames` 合并过短的最后一组。
- 使用 `--overwrite` 时，可以直接读取已有图片并重新整理 chunk，不重新解码视频。

常用命令：

```bash
python vid2img.py \
  --video_root /data/xby/YTB/dongguan/vid \
  --output_root /data/xby/YTB/dongguan/img \
  --target_fps 3.0 \
  --target_frames 900 \
  --min_last_group_frames 128
```

如果图片已经抽好，只想重新按 chunk 整理：

```bash
python vid2img.py \
  --video_root /data/xby/YTB/dongguan/vid \
  --output_root /data/xby/YTB/dongguan/img \
  --target_fps 3.0 \
  --target_frames 900 \
  --min_last_group_frames 128 \
  --overwrite
```

关键接口：

- `list_video_files(video_root)`：列出可处理视频。
- `list_existing_frame_files(output_dir)`：读取已有帧。
- `_chunk_frames(frame_paths, target_frames, min_last_group_frames)`：按目标长度切分图片。
- `rebuild_from_existing_frames(...)`：从已有图片重建 chunk 目录。
- `sample_video_frames(...)`：从视频解码、抽帧、保存图片。

## 4. 单任务与并行重建

单任务入口：`traj_recon.py`

多卡并行入口：`parallel_traj_recon.py`

推荐优先使用并行入口：

```bash
python parallel_traj_recon.py \
  --sample_root /data/xby/YTB/dongguan/img \
  --output_root /data/xby/YTB/dongguan/traj \
  --gpus 0,1,2,3,4,5,6,7 \
  --workers_per_gpu 1 \
  --cpu_threads 12 \
  --decode_workers 8 \
  --window_batch_size 8 \
  --stride 3
```

性能设计：

- `window_batch_size` 会把多个滑窗合成一个 Pi3X forward，提高 GPU 利用率。
- `workers_per_gpu` 会在同一张卡上启动多个进程，也会重复加载模型。
- 一般先固定 `workers_per_gpu=1`，逐步调大 `window_batch_size`，直到显存接近合理上限。
- 如果单 worker 仍不能吃满 GPU，再考虑 `workers_per_gpu=2`。
- `tensor_utils.py` 只给并行重建使用，避免改动原始 `data_utils.py` 的通用行为。

并行重建内部流程：

```text
build_tasks()
  -> 每个 worker 绑定一张 GPU
  -> 每个 worker load model once
  -> 任务队列分发 chunk
  -> tensor_utils 解码/resize 图片
  -> loger.reconstruction.run_inference()
  -> data_utils.save_result_directory()
```

关键接口：

- `parallel_traj_recon.build_tasks(sample_root, output_root, force_annotation)`：发现待重建 chunk。
- `parallel_traj_recon.process_task(...)`：处理单个 chunk。
- `parallel_traj_recon.worker_main(...)`：worker 主循环。
- `tensor_utils.load_images_from_paths(...)`：多线程解码图片，GPU resize，返回 `[N,C,H,W]` tensor。
- `loger.reconstruction.load_reconstruction_model(...)`：加载 Pi3X/LoGeR 模型。
- `loger.reconstruction.build_forward_kwargs(...)`：组装 window、overlap、Sim3/SE3 等推理参数。
- `loger.reconstruction.run_inference(...)`：运行重建模型。

## 5. 重建结果格式

每个重建目录包含：

```text
points.pt
camera_poses.npz
conf.npz
depth_maps.npz
meta.yaml
trajectory_xz.png
alignment.pt            # 仅开启 save_alignment 且存在 alignment payload 时保存
```

文件含义：

- `points.pt`：世界坐标系下的点云，shape 通常为 `[num_frames_pts,H,W,3]`，只包含按 `pt_strides` 抽样后的帧；低置信度位置写为 `NaN`。
- `camera_poses.npz`：全部原始帧的 camera-to-world pose，key 为 `camera_poses`。
- `conf.npz`：全部原始帧的置信度，uint8 存储，读取后会还原到 `[0,1]`。
- `depth_maps.npz`：全部原始帧的原始 depth，float16 存储，不再用 confidence mask 过滤。
- `meta.yaml`：记录图像帧数、点云帧数、点云抽帧间隔、模型、窗口参数、存储格式和文件名。

`meta.yaml` 的重要字段：

- `num_frames_img`：模型输出的全部原始图像帧数。
- `num_frames_pts`：`points.pt` 中保存的点云帧数。
- `pt_strides`：点云相对于原始帧的抽样间隔。
- `conf_threshold`：点云导出时使用的置信度阈值。
- `reference_frame`：保存结果时轨迹图的参考系标记，通常为 `initial_camera` 或 `result`。
- `window_size`、`overlap_size`：推理滑窗参数。
- `files`：各结果文件名。

当前 `meta.yaml` 的标准格式如下：

```yaml
num_frames_pts: 300
num_frames_img: 900
pt_strides: 3
video_name: BV14p4y1876e_000
reference_frame: initial_camera
conf_threshold: 0.3
conf_storage: npz_uint8_255
camera_pose_storage: npz_float32
depth_storage: npz_float16
save_alignment: false
target_resolution:
  - 512
  - 288
model_name: ckpts/Pi3X
model_kind: pi3x
window_size: 32
overlap_size: 3
files:
  points: points.pt
  conf: conf.npz
  camera_poses: camera_poses.npz
  depth_maps: depth_maps.npz
inference_stats:
  num_frames: 900
  num_windows: 31
  window_batch_size: 2
  effective_window_size: 32
  effective_overlap_size: 3
  inference_seconds: 208.74
```

字段格式约定：

- `num_frames_pts` 是 `int`，表示 `points.pt` 中点云帧数，通常等于 `ceil(num_frames_img / pt_strides)`。
- `num_frames_img` 是 `int`，表示 `camera_poses.npz`、`conf.npz`、`depth_maps.npz` 保存的完整模型输出帧数。
- `pt_strides` 是 `int`，表示点云相对原始输出的抽帧间隔；例如 `pt_strides: 3` 表示 `points.pt` 对应原始帧 `0,3,6,...`。
- `video_name` 是 `str | null`，通常是当前图片 chunk 目录名。
- `reference_frame` 是 `str`，目前主要是 `initial_camera` 或 `result`，描述保存轨迹图时使用的参考系。
- `conf_threshold` 是 `float`，范围 `[0,1]`，只用于导出 `points.pt` 时过滤点云；不会过滤 `conf.npz` 和 `depth_maps.npz`。
- `conf_storage` 固定描述置信度存储格式；当前为 `npz_uint8_255`，读取时需要除以 `255` 转回 `[0,1]`。
- `camera_pose_storage` 固定描述 pose 存储格式；当前为 `npz_float32`。
- `depth_storage` 固定描述 depth 存储格式；当前为 `npz_float16`。
- `save_alignment` 是 `bool`，表示是否保存 `alignment.pt`。
- `target_resolution` 是 `[width, height]`，对应保存点云和深度的图像分辨率。
- `model_name` 是 `str`，表示模型 checkpoint 或 HuggingFace/local model 路径。
- `model_kind` 是 `str`，例如 `pi3x` 或其他 LoGeR/Pi3 类型。
- `window_size` 和 `overlap_size` 是 `int`，记录滑窗推理参数。
- `files` 是 `dict[str,str]`，记录结果文件名，读取代码不要硬编码文件名，优先通过这里解析。
- `inference_stats` 是可选 `dict`，当前记录推理窗口统计和推理耗时，不记录显存峰值。
- `inference_stats.num_frames` 是模型输入帧数，通常等于 `num_frames_img`。
- `inference_stats.num_windows` 是滑窗推理窗口数量。
- `inference_stats.window_batch_size` 是一次 forward 合并的 window 数量；如果未显式 batch，可能不存在。
- `inference_stats.effective_window_size` 和 `inference_stats.effective_overlap_size` 是实际生效的滑窗大小和重叠大小。
- `inference_stats.inference_seconds` 是模型推理和窗口合并阶段耗时，不包含完整 pipeline 的抽帧时间。

兼容性要求：

- 读取端应该允许 `inference_stats` 缺失。
- 后续新增字段应保持向后兼容，不要改变已有字段语义。
- 如果新增结果文件，例如 `intrinsics.npz`，应同步在 `files` 中添加文件名，并新增对应的 `*_storage` 描述字段。
- 读取端目前兼容旧字段 `num_frames`、`raw_num_frames`、`stride`，但新生成和新迁移的数据应只使用 `num_frames_pts`、`num_frames_img`、`pt_strides`。
- 如果修改 `points.pt` 的抽帧策略，必须同步更新 `num_frames_pts`、`num_frames_img` 和 `pt_strides`。

当前没有保存相机内参 `K`。Pi3X 重建输出主要包含 pose、points、local_points、confidence/depth 等几何结果；如果后续需要内参，建议新增 `intrinsics.npz` 并同步更新 `meta.yaml`、`load_result_tensors()` 和可视化入口。

关键接口：

- `data_utils.save_result_directory(...)`：统一保存重建结果。
- `data_utils.load_result_meta(result_dir)`：读取 `meta.yaml`。
- `data_utils.load_result_tensors(result_dir)`：读取点云、pose、conf、depth。
- `data_utils.load_result_for_viser(...)`：读取并裁剪/采样给 viser 使用的数据。
- `data_utils.load_alignment_payload(result_dir)`：读取 alignment 相关中间信息。

## 6. 参考系与轨迹稳定化

入口脚本：`align_ground.py`

这里的 `align_ground` 不再理解为“拟合地面”，而是定义稳定的初始参考系：

- `up` 方向来自前若干帧相机旋转。
- `forward` 方向来自 `frame_step` 对应的初始位移方向。
- 默认 origin 使用第一帧相机位置。

关键接口：

- `estimate_trajectory_frame(camera_poses, frame_step)`：估计稳定参考系。
- `apply_transform_to_points(points, transform)`：把点云变换到新参考系。
- `apply_transform_to_poses(camera_poses, transform)`：把相机 pose 变换到新参考系。
- `camera_centers_from_poses(camera_poses)`：提取相机中心轨迹。

`vis_recon.py` 和 `vis_recon_chunk.py` 中的 `--reference_frame trajectory_plane` 会调用这套逻辑。

`--reference_frame` 选项：

- `auto`：按默认策略选择可视化参考系。
- `initial_camera`：把第一帧相机作为参考。
- `result`：直接使用保存结果中的坐标系。
- `trajectory_plane`：用稳定 up 和初始 forward 定义参考系。

## 7. 可视化

单 chunk 可视化：`vis_recon.py`

```bash
python vis_recon.py \
  --result_dir /data/xby/YTB/dongguan/traj/<video_id>/<video_id>_000 \
  --frame_dir /data/xby/YTB/dongguan/img/<video_id>/<video_id>_000 \
  --reference_frame trajectory_plane \
  --frame_step 8 \
  --port 8080
```

视频级 chunk 可视化：`vis_recon_chunk.py`

```bash
python vis_recon_chunk.py \
  --result_root /data/xby/YTB/dongguan/traj/<video_id> \
  --frame_root /data/xby/YTB/dongguan/img/<video_id> \
  --reference_frame trajectory_plane \
  --max_cached_chunks 1 \
  --port 8080
```

设计要点：

- `vis_recon_chunk.py` 以 chunk 为单位 lazy load。
- GUI 中提供 chunk slider、Next、Prev。
- 每次只显示当前 chunk 的 camera frame 和点云，避免一次性加载过多数据。
- `max_cached_chunks` 可以限制缓存 chunk 数，降低内存压力。
- chunk 之间默认不强制拼成统一全局坐标系；每个 chunk 使用自己的结果坐标系或参考系。

关键接口：

- `vis_recon.infer_frame_dir(result_dir)`：从 traj 路径推断 img 路径。
- `vis_recon_chunk.list_chunk_result_dirs(result_root)`：列出视频下的 chunk 重建目录。
- `vis_recon_chunk.transform_payload_for_reference_frame(...)`：按 reference frame 变换可视化 payload。
- `loger.utils.viser_utils.viser_wrapper(...)`：实际启动 viser viewer。

## 8. 稀疏 occupancy 计算

入口脚本：`collision_occupancy.py`

用途：基于重建点云为每一帧生成局部相机坐标系下的 occupied voxel，供后续碰撞检测或 RL 训练读取。

默认 ROI 和体素：

- `voxel_size=0.25`，单位米。
- `x_range=(-2.0, 2.0)`，左右宽度 4m。
- `y_range=(-0.8, 0.8)`。
- `z_range=(0.2, 5.0)`。
- `point_stride=1`，计算 occupancy 时默认不再空间跳采样。

运行命令：

```bash
python collision_occupancy.py \
  --result_dir /data/xby/YTB/shanghai0/traj/BV14p4y1876e/BV14p4y1876e_000 \
  --vis \
  --port 8090
```

输出目录默认是：

```text
<result_dir>/collision_occupancy/
  sparse_occupancy.npz
  metadata.json
```

`sparse_occupancy.npz` 内容：

- `frame_offsets`：shape `[F+1]`，第 `t` 帧数据范围是 `[frame_offsets[t], frame_offsets[t+1])`。
- `voxel_indices`：shape `[N,3]`，只保存有占据的 voxel index，顺序为 `x,y,z`。
- `voxel_probabilities`：shape `[N]`，对应 voxel 的 occupancy probability。

读取第 `t` 帧示例：

```python
import numpy as np

occ = np.load("sparse_occupancy.npz")
t = 0
start, end = occ["frame_offsets"][t], occ["frame_offsets"][t + 1]
indices_xyz = occ["voxel_indices"][start:end]
probs = occ["voxel_probabilities"][start:end]
```

`metadata.json` 保存：

- `grid_shape_xyz`：完整 dense grid 的 xyz 维度。
- `grid_dims_xyz_m`：ROI 实际物理尺寸。
- `grid_origin_xyz`：局部相机坐标系下 grid 最小角点。
- `config`：voxel size、ROI、confidence threshold 等参数。
- `files`：输出文件名。

当前设计不保存 collision probability 统计，也不默认保存 CSV 或 dense grid。这样更适合程序读取和 RL 训练中的快速查询。训练时建议离线预计算 occupancy，训练循环中只读取 sparse index 或转换后的 dense/bitset，避免每一步重复从点云构建 voxel。

关键接口：

- `OccupancyConfig`：occupancy 参数 dataclass。
- `compute_frame_occupancy(frame_idx, points, conf, camera_poses, config)`：计算单帧 sparse occupancy。
- `run_collision_analysis(result_dir, out_dir, config, save_frame_npz, save_sparse_voxels)`：批量计算并保存。
- `launch_vis_recon_viewer(args, result_dir)`：复用 `vis_recon` 风格 viewer，并叠加 occupancy。

## 9. 一键 pipeline

入口脚本：`run_new_ytb_pipeline.sh`

示例：

```bash
ROOT=/data/xby/YTB/new \
GPUS=0,1,2,3,4,5,6,7 \
WORKERS_PER_GPU=1 \
CPU_THREADS=12 \
DECODE_WORKERS=8 \
TARGET_FPS=3.0 \
TARGET_FRAMES=900 \
MIN_LAST_GROUP_FRAMES=128 \
WINDOW_BATCH_SIZE=8 \
bash run_new_ytb_pipeline.sh
```

常用环境变量：

- `ROOT`：YTB 根目录。
- `PYTHON_BIN`：Python 解释器。
- `GPUS`：GPU 列表。
- `WORKERS_PER_GPU`：每张 GPU 的 worker 数。
- `CPU_THREADS`：每个 worker 的 CPU 线程数。
- `DECODE_WORKERS`：每个 worker 的图片解码线程数。
- `TARGET_FPS`：抽帧 FPS。
- `TARGET_FRAMES`：每个 chunk 目标图片数。
- `MIN_LAST_GROUP_FRAMES`：过短尾段合并阈值。
- `WINDOW_BATCH_SIZE`：每次 forward 的 window batch。
- `LOG_ROOT`：日志目录。
- `MODE`：运行模式。
- `MAX_TASKS`：最多处理多少个重建任务。
- `DRY_RUN`：只打印计划，不执行。

## 10. 后续开发建议

修改结果格式时，优先从 `data_utils.save_result_directory()` 和 `data_utils.load_result_tensors()` 入手，并同步更新 `meta.yaml` 和本文档。

新增可视化层时，优先复用 `loger.utils.viser_utils.viser_wrapper()`，避免另写一套 viewer。当前 occupancy overlay 就是通过在 `pred_dict` 中添加 `collision_occupancy` 字段接入。

新增需要大规模读取图片的重建逻辑时，优先复用 `tensor_utils.load_images_from_paths()`。它已经把多线程解码和 GPU resize 与并行重建结合起来。

新增 RL 碰撞检测时，建议把 `sparse_occupancy.npz` 转成训练环境最方便的格式，例如 per-frame hash set、dense bool grid、bitset 或 GPU tensor cache。不要在训练 step 内重复从原始点云构建 occupancy。

如果后续需要统一不同 chunk 的全局坐标系，需要额外保存跨 chunk 的 alignment 关系。当前 chunk 可视化接受“每个 chunk 一个坐标系”的设计，不保证不同 chunk 的坐标连续。
