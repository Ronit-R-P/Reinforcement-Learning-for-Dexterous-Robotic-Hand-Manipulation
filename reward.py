import numpy as np

class DumbbellLiftReward:
    def __init__(self, initial_z=0.1357):
        self.initial_z = initial_z
        self.max_z_achieved = initial_z
        self.steps_at_limit = 0
        self.finger_curled = {}
        
    def get_touch_force(self, row, sensor_name):
        return row['touch_forces'].get(sensor_name, 0.0)
    
    def check_finger_curl(self, row, finger_name):
        joint_map = {
            'thumb':  'Thumb Knuckle Joint',
            'index':  'Index_MCP ',
            'middle': 'MIddle MCP Joint',
            'ring':   'Ring MCP Joint',
            'pinky':  'Pinky MCP',
        }
        flex_threshold = {
            'thumb':  -0.3,
            'index':  -0.5,
            'middle': -0.5,
            'ring':   -0.5,
            'pinky':  -0.5,
        }
        joint_name = joint_map[finger_name]
        joint_pos = row['joint_positions'].get(joint_name, 0.0)
        return joint_pos < flex_threshold[finger_name]
    
    def get_thumb_order(self, row):
        forces = {
            'thumb':  self.get_touch_force(row, 'thumb_dist_touch_s'),
            'index':  self.get_touch_force(row, 'index_dist_touch_s'),
            'middle': self.get_touch_force(row, 'middle_dist_touch_s'),
            'ring':   self.get_touch_force(row, 'ring_dist_touch_s'),
            'pinky':  self.get_touch_force(row, 'pinky_dist_touch_s'),
        }
        sorted_fingers = sorted(forces.items(), key=lambda x: x[1], reverse=True)
        thumb_rank = next(i+1 for i, (name, _) in enumerate(sorted_fingers) if name == 'thumb')
        return thumb_rank
    
    def compute_reward(self, row):
        reward = 0.0
        done = False
        
        cylinder_z = row['cylinder_pos'][2]
        cylinder_vel = np.linalg.norm(row['cylinder_vel'])
        cylinder_quat = row['cylinder_quat']
        is_upright = abs(cylinder_quat[0]) > 0.95
        lift_amount = cylinder_z - self.initial_z
        
        distal_forces = {
            'thumb':  self.get_touch_force(row, 'thumb_dist_touch_s'),
            'index':  self.get_touch_force(row, 'index_dist_touch_s'),
            'middle': self.get_touch_force(row, 'middle_dist_touch_s'),
            'ring':   self.get_touch_force(row, 'ring_dist_touch_s'),
            'pinky':  self.get_touch_force(row, 'pinky_dist_touch_s'),
        }
        
        active_distal = sum(1 for f in distal_forces.values() if f > 1.0)
        has_thumb = distal_forces['thumb'] > 1.0
        
        # 1. PER-FINGER CURL REWARD
        finger_curl_reward = 0.0
        all_curled = True
        curl_details = {}
        
        for finger_name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            is_curled = self.check_finger_curl(row, finger_name)
            is_touching = distal_forces[finger_name] > 1.0
            
            if is_curled and is_touching:
                finger_curl_reward += 0.1
                self.finger_curled[finger_name] = True
                curl_details[finger_name] = 'curled+touch'
            elif is_curled:
                curl_details[finger_name] = 'curled only'
                all_curled = False
            elif is_touching:
                curl_details[finger_name] = 'touch only'
                all_curled = False
            else:
                curl_details[finger_name] = 'none'
                all_curled = False
        
        reward += finger_curl_reward
        
        # 2. FULL CURL BONUS
        if all_curled:
            reward += 1.0
        
        # 3. THUMB ORDERING REWARD
        thumb_rank = self.get_thumb_order(row)
        if thumb_rank <= 2 and has_thumb:
            reward += 0.3
        
        # 4. CONTINUOUS LIFT REWARD
        grip_viable = active_distal >= 3 and has_thumb
        
        if grip_viable:
            grip_quality = active_distal / 5.0
            lift_reward = lift_amount * 15.0 * grip_quality
            
            if cylinder_z > self.max_z_achieved:
                height_improvement = cylinder_z - self.max_z_achieved
                lift_reward += height_improvement * 25.0
                self.max_z_achieved = cylinder_z
            
            reward += max(0, lift_reward)
        
        # 5. PERFECT GRIP BONUS
        if all_curled and grip_viable and lift_amount > 0.01:
            reward += 0.2
        
        # PENALTIES
        if lift_amount > 0.02 and cylinder_vel > 0.5:
            reward -= cylinder_vel * 0.5
        
        if lift_amount > 0.05 and not is_upright:
            reward -= 0.5
        
        cylinder_x = row['cylinder_pos'][0]
        cylinder_y = row['cylinder_pos'][1]
        lateral_drift = np.sqrt((cylinder_x - 0.28)**2 + (cylinder_y + 0.063)**2)
        if lift_amount > 0.03 and lateral_drift > 0.1:
            reward -= lateral_drift * 2.0
        
        for force in distal_forces.values():
            if force > 60.0:
                reward -= 0.05 * (force - 60.0)
        
        # DROP DETECTION
        # CHANGED: was gated on an absolute "max_z_achieved > initial_z + 0.05" (5cm)
        # requirement before ANY drop penalty could fire. That meant genuine drops
        # from a lower peak (e.g. lifted only 2cm, then fully released) went entirely
        # unpunished -- confirmed by the BC sanity-check rollout, where the object
        # peaked at ~2.2cm, then fell to -7cm with grip_viable=False, and got reward 0
        # instead of the intended -5.0 penalty.
        #
        # New logic: still requires a genuine release (active_distal < 2, i.e. grip
        # actually lost, not just slipping while still in contact -- slipping alone
        # rarely drops active_distal below 2, so natural in-hand slide during a good
        # hold still won't trigger this). Instead of an absolute 5cm bar, it checks
        # (a) any real lift attempt happened at all (>1cm above start, filters out
        # pure noise/jitter at the very start of an episode), and (b) the object has
        # fallen meaningfully (>3cm) from its own achieved peak, not from a fixed
        # world-frame threshold. This catches drops at any height, not just >5cm ones.
        grip_lost = active_distal < 2
        significant_lift_achieved = self.max_z_achieved > (self.initial_z + 0.01)
        fell_from_peak = (self.max_z_achieved - cylinder_z) > 0.03
        
        if significant_lift_achieved and fell_from_peak and grip_lost:
            reward -= 5.0
            done = True
        
        # SUCCESS: MOUNT Z LIMIT
        mount_z_pos = row['joint_positions'].get('mount_z_joint', 0.0)
        mount_z_max = 0.3
        
        if mount_z_pos >= (mount_z_max - 0.01) and grip_viable:
            self.steps_at_limit += 1
            if self.steps_at_limit >= 100:
                reward += 50.0
                done = True
        else:
            self.steps_at_limit = 0
        
        return reward, done, {
            'curl_details': curl_details,
            'thumb_rank': thumb_rank,
            'lift_amount': lift_amount,
            'all_curled': all_curled,
            'grip_viable': grip_viable,
            'grip_quality': active_distal / 5.0,
            'max_z': self.max_z_achieved,
            'cylinder_vel': cylinder_vel,
            'is_upright': is_upright,
            'lateral_drift': lateral_drift,
        }
    
    def reset(self):
        self.max_z_achieved = self.initial_z
        self.steps_at_limit = 0
        self.finger_curled = {name: False for name in [
            'thumb', 'index', 'middle', 'ring', 'pinky'
        ]}