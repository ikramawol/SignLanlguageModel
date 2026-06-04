# Run this once on a saved seq from data.pkl to verify pose layout
import pickle, numpy as np
with open('keypoint_data/data.pkl', 'rb') as f:
    d = pickle.load(f)
sample = d['data'][0, 0, :]   # frame 0 of sample 0, raw dims

# Pose x-coords should be in [0, 1] (normalized image space)
pose_x = sample[0:99:3]
print('Pose x range:', pose_x.min().round(3), '–', pose_x.max().round(3))
# Expected: roughly 0.2 – 0.8 for a centred signer