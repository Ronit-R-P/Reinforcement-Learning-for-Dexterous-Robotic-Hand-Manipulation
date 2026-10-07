import numpy as np
import os
import glob
from reward import DumbbellLiftReward

demo_folder = "records"
demo_files = sorted(glob.glob(os.path.join(demo_folder, "*_named.npy")))

if not demo_files:
    print("No _named.npy files found!")
    exit()

print(f"Found {len(demo_files)} converted demo file(s):")
for f in demo_files:
    size_mb = os.path.getsize(f) / (1024*1024)
    print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")

for demo_idx, demo_file in enumerate(demo_files):
    print(f"\n{'='*80}")
    print(f"DEMO {demo_idx+1}: {os.path.basename(demo_file)}")
    print(f"{'='*80}")
    
    data = np.load(demo_file, allow_pickle=True)
    print(f"Total steps: {len(data)}")
    
    reward_fn = DumbbellLiftReward(initial_z=0.1357)
    reward_fn.reset()
    
    total_reward = 0.0
    peak_reward = -999
    peak_reward_step = 0
    first_contact_step = None
    first_full_curl_step = None
    first_lift_step = None
    all_rewards = []
    
    print(f"\n{'Step':<8} {'Reward':<8} {'Curled':<8} {'ThumbR':<8} {'Lift(cm)':<10} {'Forces(i/m/r/p/t)':<28} {'Events'}")
    print("-" * 115)
    
    last_print_step = -5000
    
    for i, row in enumerate(data):
        r, done, details = reward_fn.compute_reward(row)
        total_reward += r
        all_rewards.append(r)
        
        if r > peak_reward:
            peak_reward = r
            peak_reward_step = i
        
        curled_count = sum(reward_fn.finger_curled.values())
        if first_contact_step is None and curled_count >= 1:
            first_contact_step = i
        if first_full_curl_step is None and details['all_curled']:
            first_full_curl_step = i
        if first_lift_step is None and details['lift_amount'] > 0.01:
            first_lift_step = i
        
        is_key_event = (i == first_contact_step or i == first_full_curl_step or 
                       i == first_lift_step or done)
        
        if i % 5000 == 0 or is_key_event or done:
            if i - last_print_step >= 1000 or is_key_event or done:
                tf = row['touch_forces']
                force_str = f"{tf.get('index_dist_touch_s',0):.1f}/{tf.get('middle_dist_touch_s',0):.1f}/{tf.get('ring_dist_touch_s',0):.1f}/{tf.get('pinky_dist_touch_s',0):.1f}/{tf.get('thumb_dist_touch_s',0):.1f}"
                
                events = []
                if i == first_contact_step: events.append("FIRST_CONTACT")
                if i == first_full_curl_step: events.append("FULL_CURL")
                if i == first_lift_step: events.append("FIRST_LIFT")
                if done:
                    if r > 40:
                        events.append("SUCCESS")
                    else:
                        events.append("DROPPED")
                
                print(f"{i:<8} {r:<8.3f} {curled_count}/5{'':<4} {details['thumb_rank']:<8} {details['lift_amount']*100:<10.2f} {force_str:<28} {','.join(events)}")
                last_print_step = i
        
        if done:
            break
    
    print(f"\n{'─'*80}")
    print(f"SUMMARY - Demo {demo_idx+1}")
    print(f"{'─'*80}")
    print(f"  Total reward:        {total_reward:.2f}")
    print(f"  Peak step reward:    {peak_reward:.3f} (step {peak_reward_step})")
    print(f"  Average reward:      {np.mean(all_rewards):.4f}")
    print(f"  First contact:       step {first_contact_step}" if first_contact_step else "  First contact:       NEVER")
    print(f"  First full curl:     step {first_full_curl_step}" if first_full_curl_step else "  First full curl:     NEVER")
    print(f"  First lift (>1cm):   step {first_lift_step}" if first_lift_step else "  First lift (>1cm):   NEVER")
    print(f"  Max height:          {(reward_fn.max_z_achieved - 0.1357)*100:.2f} cm")
    
    if done:
        if all_rewards[-1] > 40:
            print(f"  Result:              SUCCESS (mount_z limit reached)")
        else:
            print(f"  Result:              DROPPED (dumbbell fell)")
    else:
        print(f"  Result:              TIMEOUT (demo ended naturally)")
    
    print(f"  Ended at step:       {i}")
    
    if first_contact_step and first_full_curl_step:
        idle_avg = np.mean(all_rewards[:first_contact_step])
        curl_avg = np.mean(all_rewards[first_contact_step:first_full_curl_step])
        lift_avg = np.mean(all_rewards[first_full_curl_step:])
        print(f"\n  Phase averages:")
        print(f"    Idle:      {idle_avg:.4f}")
        print(f"    Curling:   {curl_avg:.4f}")
        print(f"    Lift:      {lift_avg:.4f}")

print(f"\n{'='*80}")
print("ALL DEMOS ANALYZED")
print(f"{'='*80}")