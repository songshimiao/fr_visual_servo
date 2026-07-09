# PnP多解消除功能说明

## 功能概述

在当前实现中，我们添加了PnP姿态多解消除功能，解决了cv2.solvePnP(IPPE_SQUARE)在相机正对marker时存在的两个数值接近候选解随机跳变的问题，这是视觉伺服抖动的经典根因。

## 实现方式

在`scripts/algorithm/aruco_detector.py`文件中，我们实现了`estimate_marker_poses_robust`函数，该函数：

1. 首先调用`estimate_marker_poses`获取多个候选解
2. 对于每个marker，计算与上一帧的相似度
3. 选择与前一帧姿态最相似的解作为最终结果
4. 有效避免了相机接近正对marker时的解跳变问题

## 代码实现

```python
def estimate_marker_poses_robust(corners, marker_length, camera_matrix, dist_coeffs, 
                                prev_poses: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
    """
    估计每个 marker 相对相机的位姿，采用多解消除策略以避免相机正对时的解跳变。
    """
    # ... 实现细节 ...
```

## 配置和使用

该功能已集成到主程序中，无需额外配置即可使用。在`main_pbvs.py`中，通过`estimate_marker_poses_robust`替代了原有的`estimate_marker_poses`调用。

## 性能优化

- 通过比较相邻帧的姿态相似度，避免了不必要的计算
- 保持了原有的姿态估计精度
- 提升了视觉伺服系统的稳定性