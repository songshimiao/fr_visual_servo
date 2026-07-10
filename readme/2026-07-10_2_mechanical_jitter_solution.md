# 机械臂运动抖动问题分析与解决方案

## 问题描述

在真实机械臂测试中发现以下问题现象：
1. 机械臂启动时会产生非常突然的快速运动
2. 当检测到marker静止足够时间后，从hover-far切换到hover-near时也会发生突然的快速运动
3. 导致机械臂整体剧烈抖动
4. 但在手动移动marker时，视觉伺服阶段运动是平滑的

## 原因分析

经过分析，问题主要源于以下几点：

### 1. 初始位置跳跃问题
- 机械臂启动时，如果当前位置与第一个目标位置相差很大
- 会导致初始加速度过大，产生冲击

### 2. 状态切换突变
- 从`hover-far`切换到`hover-near`时，目标位置发生突变
- 由于是即时切换而非平滑过渡，产生冲击

### 3. 轨迹规划参数不足
- 默认的关节速度和加速度限制可能不足以处理大位移
- 距离大的情况下，需要更平滑的轨迹规划

## 解决方案

### 1. 增加初始缓冲策略
在机械臂首次启动时，增加一个缓慢的初始运动阶段，避免直接跳转到远距离目标。

### 2. 平滑状态切换
在状态切换时增加过渡缓冲，避免目标位置的突变。

### 3. 优化轨迹规划参数
调整初始运动的速度和加速度参数，使其适应大位移场景。

## 实现方法

### 优化轨迹规划
修改`move_joint_space`方法，增加初始运动的平滑处理：

```python
# 在首次伺服开始时，先移动到一个中间位置，再开始正常伺服
if self.first_move:
    # 首次移动使用更保守的速度参数
    target_joint_pos = self._calculate_initial_target(q_start, q_end)
    joint_trajectory = self.move_joint_space(q_start, target_joint_pos, 
                                           velocity=self.DEFAULT_JOINT_VEL_LIMIT*0.3,
                                           acceleration=self.DEFAULT_JOINT_ACC_LIMIT*0.5)
    # 然后进行正常伺服
    self.first_move = False
```

### 增加状态切换缓冲
在`HoverStateMachine`中增加更平滑的状态转换逻辑：
```python
# 状态切换时，先移动到过渡位置再切换到最终目标
if self.state == self.TRACKING and is_still:
    # 先移动到过渡目标位置，再切换到最终位置
    transition_pose = self._calculate_transition_pose(base_T_marker)
    # 使用过渡位置进行平滑伺服
```

## 参数调优建议

1. **初始速度限制**：首次启动时降低速度到默认值的30%
2. **加速度限制**：首次启动时降低加速度到默认值的50%
3. **过渡时间**：在状态切换时增加最小停留时间，确保平稳过渡