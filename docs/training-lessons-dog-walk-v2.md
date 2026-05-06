# Dog Walk V2 训练经验总结

**日期**: 2026-05-01  
**分支**: feat/dog-walk-v2  
**训练总时长**: ~30小时（含多次失败重训）

---

## 1. 整体结论

- Tier 1 通用运动策略可以通过多阶段课程训练达到竞赛地形 level 4（5cm台阶、7°坡度）
- 受限于奖励函数偏离规范设计，未能突破 level 4 天花板
- **核心教训**：规范设计的奖励权重是经过权衡的，随意偏离会产生隐性冲突，且多项同时修改后无法归因

---

## 2. 训练稳定性问题

### 2.1 action_rate_l2 数值爆炸（复现3次）

**现象**：训练在 1000-3000 轮时 reward 突然跳至 -6e13，value_function_loss 爆炸  
**权重尝试**：-0.05 → -0.02 → 均爆炸  
**根本原因**：joint acceleration 惩罚（dof_acc_l2）已覆盖动作平滑性，action_rate_l2 与之叠加在某些地形接触事件下产生极大梯度  
**解决方案**：**彻底删除 action_rate_l2**，dof_acc_l2 单独负责平滑性  
**注意**：恢复 adaptive LR 之后理论上可以加回极小权重（-0.005），但无强需求不建议加

### 2.2 Critic Loss 爆炸（value_function_loss > 1e33）

**现象**：训练约 3200 轮时 critic loss 突然爆炸，策略完全崩溃  
**根本原因**：`schedule="fixed"`, `lr=1e-3` 在 action_rate_l2 爆炸事件后遗留的梯度累积  
**解决方案**：
```python
self.algorithm.learning_rate = 3e-4
self.algorithm.schedule = "adaptive"
self.algorithm.desired_kl = 0.01
self.algorithm.max_grad_norm = 0.5
```
**教训**：原来改成 fixed LR 是为了防止 adaptive LR 在 action_rate_l2 爆炸时 LR→0 崩溃。删除 action_rate_l2 后，adaptive LR 就可以安全使用了。两个问题互相关联。

---

## 3. 地形课程设计

### 3.1 跨越难度鸿沟：必须设中间阶段

**失败案例**：直接从平地跳到竞赛地形  
- terrain_levels 全程维持在 0.0005，课程从未推进
- 竞赛地形包含 45% 阶梯/跳石，robot 在这些地块直接失败

**解决方案**：渐进式中间阶段

```
Phase 1 (平地) 
  → Phase 1.5 (50%平+30%轻度起伏+20%坡) 
  → Phase 1.75 (25%平+15%起伏+20%坡+30%低难度阶梯+10%跳石)
  → Phase 2 (竞赛地形)
```

**结果**：Phase 1.75 完成后 terrain_levels ~3.0，Phase 2 启动时 terrain_levels ~1.2（而非 0）

### 3.2 各阶段地形参数必须对齐

**失败案例**：Phase 2 跳石比 Phase 1.75 最高难度还难（stone_height 5cm vs 3cm，石块更窄更散）  
- 即使 Phase 1.75 训练到 level 3，Phase 2 level 0 的跳石仍然超出能力
- 导致 terrain_levels 在 Phase 2 从 1.2 缓慢下降到 0

**解决方案**：Phase 2 跳石参数对齐 Phase 1.75 最高难度（stone_height 3cm，宽度 0.35-0.50m，间距 0.02-0.10m）

**规律**：相邻阶段之间，**后阶段 level 0 的难度 ≤ 前阶段最高 level 的难度**

### 3.3 terrain_levels 指标的含义

terrain_levels 是所有环境的平均难度等级：
- 某类地形（如跳石）占 15%，机器人在这类地形上全部失败
- 即使其余 85% 地形通过，跳石的持续失败会把平均值压低
- 因此 terrain_levels=4 不代表"能通过 4 级难度的所有地形"，而是"加权平均难度是 4"

---

## 4. 奖励函数设计教训

### 4.1 偏离规范的代价

以下偏离规范的改动事后确认是导致 level 4 平台期的主要原因：

| 奖励项 | 规范值 | 实际使用值 | 影响 |
|--------|--------|------------|------|
| `lin_vel_z_l2` | -0.5 | -1.5 (3×) | 爬台阶必然产生垂直速度，过重惩罚阻止爬台阶 |
| `flat_orientation_l2` | -0.5 | -2.0 (4×) | 爬坡/台阶时躯干自然前倾，过重惩罚阻止倾斜 |
| `ang_vel_xy_l2` | -0.1 | -0.2 (2×) | 同上，过度约束姿态 |

两项惩罚合力：机器人在地形上"不敢"颠簸也"不敢"倾斜，最优策略变成在平地上徘徊。

### 4.2 补丁式奖励项的陷阱

训练过程中陆续添加了规范外的奖励项来"修补"观察到的问题：

| 补丁项 | 添加原因 | 实际效果 |
|--------|----------|----------|
| `gait_clock` (1.5Hz trot) | 步频太高 | Phase 1.75 有效，进入竞赛地形后奖励值跌 20x，与爬台阶的慢速步态冲突 |
| `hip_deviation` | 髋关节外张 | 增加了复杂性，与行走策略存在隐性冲突 |
| `prolonged_air` | 腿悬空太久 | 不在规范，叠加了不必要的约束 |

**规律**：观察到问题不应立即加新 reward，应先诊断原因。加了新 reward 等于增加一个新的优化目标，可能与现有目标冲突，且难以归因。

### 4.3 gait_clock 专项分析

gait_clock 在不同阶段的表现：
- Phase 1（平地）：~0.07-0.10（正常，约占最大值的 15-20%）
- Phase 1.75（过渡地形）：~0.09-0.10（正常）
- Phase 2（竞赛地形）：**0.003-0.005**（崩塌，约占最大值的 0.6-1%）

崩塌原因：gait_clock 要求 1.5Hz trot（快速交替步），竞赛地形上爬台阶需要慢速单腿支撑。机器人面临"trot 奖励 vs 爬台阶奖励"的冲突，最终选择放弃 trot。

**结论**：gait_clock 在平地阶段有价值（引导步态形成），在竞赛地形阶段应移除。

---

## 5. 动作幅度

| 关节 | 原始值 | 最大脚掌抬升贡献 |
|------|--------|-----------------|
| thigh scale 0.20 | | ~4.2cm |
| calf scale 0.25 | | ~5.3cm |
| **合计** | | **~9.5cm**（恰好是 10cm 台阶的极限） |

调整到 thigh=0.25, calf=0.30 后最大抬升约 11.5cm，留有余量。

**教训**：action scale 决定了物理上能做到的动作范围上限，必须在设计阶段验证是否满足比赛要求，不能临时提高。

---

## 6. 工程问题

### 6.1 磁盘管理

根文件系统（30G）的主要占用来源：

| 来源 | 大小 | 说明 |
|------|------|------|
| `/tmp/IsaacLab` | 2-3G | Isaac Sim 地形网格缓存，每次新地形配置重新生成 |
| `/tmp/*.log` (训练日志) | 50-100M | nohup 重定向的 stdout |
| `/tmp/opencode_install_*` | 30-50M×多个 | 安装包 |

**解决方案**：
```bash
# 把 IsaacLab 缓存软链到数据盘（训练结束后执行）
rm -rf /tmp/IsaacLab
mkdir -p /root/gpufree-data/legged_lab/.isaac_cache
ln -s /root/gpufree-data/legged_lab/.isaac_cache /tmp/IsaacLab

# 训练日志重定向到数据盘
nohup python scripts/rsl_rl/train.py ... > /root/gpufree-data/legged_lab/logs/phase_train.log 2>&1 &
```

**注意**：模型 checkpoint (.pt) 本来就保存在数据盘，不受影响。

### 6.2 RSL-RL 断点续训的迭代计数行为

加载 checkpoint 时，若 `checkpoint_iter > max_iterations`，RSL-RL 实际运行 `checkpoint_iter + max_iterations` 轮：
- Phase 1.5: max=2000, 加载自 iter ~5000 → 训练到 ~7000
- Phase 2: max=8000, 加载自 iter 8100 → 训练到 16100

**实践**：设置 max_iterations 时应理解为"额外轮数"而非"总轮数"。

---

## 7. 训练工作流规范（经验总结）

1. **先把平地走好**：Phase 1 必须达到 mean_reward 稳定、time_out 率 > 85% 才能进入下一阶段
2. **一次只改一个东西**：reward 权重、地形参数、action scale 不要同时改多个
3. **出问题先诊断，再修改**：不要用新 reward 项掩盖问题
4. **保持奖励权重接近规范**：偏离要有充分理由并记录
5. **地形阶段之间难度连续**：后阶段 level 0 ≤ 前阶段最高 level
6. **每个阶段有明确验收标准**：terrain_levels 达到目标值才进下一阶段
7. **不要在竞赛地形阶段使用步态约束 reward**：让策略自由选择步态

---

## 8. 下一步行动

按此优先级继续：

1. **修正 Tier 1 奖励到规范值**，删除 gait_clock/hip_deviation/prolonged_air，提高高度扫描分辨率
2. **重训 Phase 1→2→3→4**，严格执行每阶段验收标准
3. **Tier 2：爬墙专项策略**（规范第4节，独立训练，1024 envs，2000 iters）
4. **Tier 3：感知模块 + 状态机**（规范第5节）
5. **系统联调**

参考文件：
- 规范设计：`docs/superpowers/specs/2026-04-28-dog-locomotion-competition-design.md`
- 实施计划：`docs/superpowers/plans/2026-04-28-tier1-universal-locomotion.md`
