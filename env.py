import numpy as np
import mujoco
from reward import DumbbellLiftReward

class DumbbellLiftEnv:
    def __init__(self, xml_path="debug_hand.xml", max_steps=5000, control_freq=100, warmup_steps=500):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        
        self.max_steps = max_steps
        self.control_freq = control_freq
        self.step_count = 0
        self.warmup_steps = warmup_steps
        
        self.reward_fn = DumbbellLiftReward(initial_z=0.13)
        
        self.action_dim = self.model.nu
        self.action_low = -1.0
        self.action_high = 1.0
        
        self._build_spaces()
        
        self.actuator_names = [
            'Index_MCP_act', 'Index_PIP_DIP_act',
            'Middle_MCP_act', 'Middle_PIP_DIP_act',
            'Ring_MCP_act', 'Ring_PIP_DIP_act',
            'Pinky_MCP_act', 'Pinky_PIP_DIP_act',
            'Thumb_Knuckle_act', 'Thumb_MCP_act',
            'mount_z_act'
        ]
        
        self.ctrl_range_low = np.array([
            self.model.actuator(name).ctrlrange[0] for name in self.actuator_names
        ])
        self.ctrl_range_high = np.array([
            self.model.actuator(name).ctrlrange[1] for name in self.actuator_names
        ])
        
        self.sensor_name_to_idx = {}
        for i in range(self.model.nsensor):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            if name:
                self.sensor_name_to_idx[name] = (self.model.sensor_adr[i], self.model.sensor_dim[i])
        
        self.joint_name_to_idx = {}
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                self.joint_name_to_idx[name] = i
    
    def _build_spaces(self):
        # CHANGED: dropped the "+ 11" actuator_forces term (was 11+11+11+5+4+3+4+3=52).
        # actuator_forces was removed from _get_obs() below because it's a near-direct
        # algebraic function of qpos/qvel/ctrl for position actuators, which let the
        # BC policy shortcut around learning a real contact-driven behavior.
        obs_dim = 11 + 11 + 5 + 4 + 3 + 4 + 3
        self.obs_dim = obs_dim
    
    def _get_obs(self):
        obs = []
        
        for name in ['mount_z_joint', 'Index_MCP ', 'Index_PIP&DIP_Joint',
                     'MIddle MCP Joint', 'Middle PIP&DIP Joint',
                     'Ring MCP Joint', 'Ring PIP&DIP Joint',
                     'Pinky MCP', 'Pinky PIP&DIP Joint',
                     'Thumb Knuckle Joint', 'Thumb MCP Joint']:
            obs.append(self.data.qpos[self.joint_name_to_idx[name]])
        
        for name in ['mount_z_joint', 'Index_MCP ', 'Index_PIP&DIP_Joint',
                     'MIddle MCP Joint', 'Middle PIP&DIP Joint',
                     'Ring MCP Joint', 'Ring PIP&DIP Joint',
                     'Pinky MCP', 'Pinky PIP&DIP Joint',
                     'Thumb Knuckle Joint', 'Thumb MCP Joint']:
            obs.append(self.data.qvel[self.joint_name_to_idx[name]])
        
        # CHANGED: actuator_forces block removed entirely (was here, 11 dims: a loop
        # appending self.data.actuator_force[...] for each actuator in self.actuator_names)
        
        touch_names = [
            'index_dist_touch_s', 'middle_dist_touch_s', 'ring_dist_touch_s',
            'pinky_dist_touch_s', 'thumb_dist_touch_s'
        ]
        for name in touch_names:
            adr, _ = self.sensor_name_to_idx[name]
            obs.append(self.data.sensordata[adr])
        
        prox_names = ['index_prox_touch_s', 'middle_prox_touch_s', 'ring_prox_touch_s', 'pinky_prox_touch_s']
        for name in prox_names:
            adr, _ = self.sensor_name_to_idx[name]
            obs.append(self.data.sensordata[adr])
        
        adr, dim = self.sensor_name_to_idx['cylinder_pos']
        obs.extend(self.data.sensordata[adr:adr+dim])
        adr, dim = self.sensor_name_to_idx['cylinder_quat']
        obs.extend(self.data.sensordata[adr:adr+dim])
        adr, dim = self.sensor_name_to_idx['cylinder_vel']
        obs.extend(self.data.sensordata[adr:adr+dim])
        
        return np.array(obs, dtype=np.float32)
    
    def _build_row(self):
        # NOTE: left untouched on purpose. This is used for reward computation
        # (DumbbellLiftReward.compute_reward), not for the policy observation,
        # and the reward function never reads actuator_forces anyway -- so there's
        # no leakage concern here and no need to change it.
        joint_positions = {}
        joint_names = ['mount_z_joint', 'Index_MCP ', 'Index_PIP&DIP_Joint',
                       'MIddle MCP Joint', 'Middle PIP&DIP Joint',
                       'Ring MCP Joint', 'Ring PIP&DIP Joint',
                       'Pinky MCP', 'Pinky PIP&DIP Joint',
                       'Thumb Knuckle Joint', 'Thumb MCP Joint']
        for name in joint_names:
            joint_positions[name] = float(self.data.qpos[self.joint_name_to_idx[name]])
        
        joint_velocities = {}
        for name in joint_names:
            joint_velocities[name] = float(self.data.qvel[self.joint_name_to_idx[name]])
        
        actuator_forces = {}
        for name in self.actuator_names:
            actuator_forces[name] = float(self.data.actuator_force[self.model.actuator(name).id])
        
        controls = {}
        for name in self.actuator_names:
            controls[name] = float(self.data.ctrl[self.model.actuator(name).id])
        
        touch_forces = {}
        touch_names = [
            'index_dist_touch_s', 'middle_dist_touch_s', 'ring_dist_touch_s',
            'pinky_dist_touch_s', 'thumb_dist_touch_s',
            'index_prox_touch_s', 'middle_prox_touch_s', 'ring_prox_touch_s', 'pinky_prox_touch_s'
        ]
        for name in touch_names:
            adr, _ = self.sensor_name_to_idx[name]
            touch_forces[name] = float(self.data.sensordata[adr])
        
        adr, dim = self.sensor_name_to_idx['cylinder_pos']
        cylinder_pos = self.data.sensordata[adr:adr+dim].tolist()
        adr, dim = self.sensor_name_to_idx['cylinder_quat']
        cylinder_quat = self.data.sensordata[adr:adr+dim].tolist()
        adr, dim = self.sensor_name_to_idx['cylinder_vel']
        cylinder_vel = self.data.sensordata[adr:adr+dim].tolist()
        
        return {
            'joint_positions': joint_positions,
            'joint_velocities': joint_velocities,
            'actuator_forces': actuator_forces,
            'controls': controls,
            'touch_forces': touch_forces,
            'cylinder_pos': cylinder_pos,
            'cylinder_quat': cylinder_quat,
            'cylinder_vel': cylinder_vel,
        }
    
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.reward_fn.reset()
        self.step_count = 0
        
        for _ in range(50):
            mujoco.mj_step(self.model, self.data)
        
        return self._get_obs()
    
    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        scaled_action = self.ctrl_range_low + (action + 1.0) / 2.0 * (self.ctrl_range_high - self.ctrl_range_low)
        
        # Warm-up: force full curl on all fingers, keep mount_z low
        if self.step_count < self.warmup_steps:
            scaled_action[:10] = self.ctrl_range_low[:10]
            scaled_action[10] = self.ctrl_range_low[10]
        
        for i, name in enumerate(self.actuator_names):
            self.data.ctrl[self.model.actuator(name).id] = scaled_action[i]
        
        substeps = max(1, int(1.0 / (self.model.opt.timestep * self.control_freq)))
        for _ in range(substeps):
            mujoco.mj_step(self.model, self.data)
        
        self.step_count += 1
        
        obs = self._get_obs()
        row = self._build_row()
        reward, done, details = self.reward_fn.compute_reward(row)
        
        # Contact bonus
        tf = row['touch_forces']
        for f in ['index_dist_touch_s', 'middle_dist_touch_s', 'ring_dist_touch_s',
                  'pinky_dist_touch_s', 'thumb_dist_touch_s']:
            if tf.get(f, 0) > 1.0:
                reward += 0.1
        
        if self.step_count >= self.max_steps:
            done = True
        
        info = {
            'step': self.step_count,
            'lift_amount': details['lift_amount'],
            'all_curled': details['all_curled'],
            'grip_viable': details['grip_viable'],
            'thumb_rank': details['thumb_rank'],
            'max_z': details['max_z'],
        }
        
        return obs, reward, done, info
    
    @property
    def observation_space(self):
        return (self.obs_dim,)
    
    @property
    def action_space(self):
        return (self.action_dim,)