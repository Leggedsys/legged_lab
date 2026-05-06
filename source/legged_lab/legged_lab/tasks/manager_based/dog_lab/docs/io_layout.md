# Dog Lab IO Layout

This note records the observation and action vector layout for `Dog-Legged-Lab-v0` / `Dog-Legged-Lab-Walk-v0`.

## Observation Layout

Observation terms are defined in `dog_env_cfg.py` under `ObservationsCfg.PolicyCfg` and concatenated in declaration order because `self.concatenate_terms = True`.

The final policy observation is a 45-dim vector:

```text
obs[0]   = base_ang_vel.x
obs[1]   = base_ang_vel.y
obs[2]   = base_ang_vel.z

obs[3]   = projected_gravity.x
obs[4]   = projected_gravity.y
obs[5]   = projected_gravity.z

obs[6]   = velocity_commands[0]
obs[7]   = velocity_commands[1]
obs[8]   = velocity_commands[2]

obs[9]   = joint_pos[FL_hip_joint]
obs[10]  = joint_pos[FR_hip_joint]
obs[11]  = joint_pos[RL_hip_joint]
obs[12]  = joint_pos[RR_hip_joint]
obs[13]  = joint_pos[FL_thigh_joint]
obs[14]  = joint_pos[FR_thigh_joint]
obs[15]  = joint_pos[RL_thigh_joint]
obs[16]  = joint_pos[RR_thigh_joint]
obs[17]  = joint_pos[FL_calf_joint]
obs[18]  = joint_pos[FR_calf_joint]
obs[19]  = joint_pos[RL_calf_joint]
obs[20]  = joint_pos[RR_calf_joint]

obs[21]  = joint_vel[FL_hip_joint]
obs[22]  = joint_vel[FR_hip_joint]
obs[23]  = joint_vel[RL_hip_joint]
obs[24]  = joint_vel[RR_hip_joint]
obs[25]  = joint_vel[FL_thigh_joint]
obs[26]  = joint_vel[FR_thigh_joint]
obs[27]  = joint_vel[RL_thigh_joint]
obs[28]  = joint_vel[RR_thigh_joint]
obs[29]  = joint_vel[FL_calf_joint]
obs[30]  = joint_vel[FR_calf_joint]
obs[31]  = joint_vel[RL_calf_joint]
obs[32]  = joint_vel[RR_calf_joint]

obs[33]  = last_action[FL_hip_joint]
obs[34]  = last_action[FR_hip_joint]
obs[35]  = last_action[RL_hip_joint]
obs[36]  = last_action[RR_hip_joint]
obs[37]  = last_action[FL_thigh_joint]
obs[38]  = last_action[FR_thigh_joint]
obs[39]  = last_action[RL_thigh_joint]
obs[40]  = last_action[RR_thigh_joint]
obs[41]  = last_action[FL_calf_joint]
obs[42]  = last_action[FR_calf_joint]
obs[43]  = last_action[RL_calf_joint]
obs[44]  = last_action[RR_calf_joint]
```

## Action Layout

Action terms are defined in `dog_env_cfg.py` under `ActionsCfg`.

The final policy action is a 12-dim vector:

```text
act[0]   = FL_hip_joint
act[1]   = FR_hip_joint
act[2]   = FL_thigh_joint
act[3]   = FR_thigh_joint
act[4]   = FL_calf_joint
act[5]   = FR_calf_joint
act[6]   = RL_hip_joint
act[7]   = RR_hip_joint
act[8]   = RL_thigh_joint
act[9]   = RR_thigh_joint
act[10]  = RL_calf_joint
act[11]  = RR_calf_joint
```

This ordering comes from:

1. `front_legs`: `FL/FR` with `hip -> thigh -> calf`
2. `rear_legs`: `RL/RR` with `hip -> thigh -> calf`

## Default Action Offset

Both action groups use `use_default_offset=True`, so the network outputs position deltas around the robot default pose rather than absolute joint targets.

The target is approximately:

```text
target_joint_pos = default_joint_pos + action * scale
```

Default joint pose is defined in `legged_lab/assets/dog_cfg.py`:

```text
FL_hip_joint    =  0.14
FR_hip_joint    = -0.14
RL_hip_joint    =  0.10
RR_hip_joint    = -0.10
FL_thigh_joint  =  0.34
FR_thigh_joint  =  0.34
RL_thigh_joint  =  0.46
RR_thigh_joint  =  0.46
FL_calf_joint   = -0.72
FR_calf_joint   = -0.72
RL_calf_joint   = -0.82
RR_calf_joint   = -0.82
```

Action scales from `dog_env_cfg.py`:

```text
front hip    = 0.025
front thigh  = 0.035
front calf   = 0.02
rear hip     = 0.03
rear thigh   = 0.045
rear calf    = 0.025
```

## Measured Joint Order

The joint order above was checked from the loaded dog asset and is:

```text
FL_hip_joint
FR_hip_joint
RL_hip_joint
RR_hip_joint
FL_thigh_joint
FR_thigh_joint
RL_thigh_joint
RR_thigh_joint
FL_calf_joint
FR_calf_joint
RL_calf_joint
RR_calf_joint
```
