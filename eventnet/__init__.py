"""Event Tensor Network for full-waveform LiDAR ghost/glass segmentation.

See ``FW_Event_Net/initial_plan.md``. Replaces the dense T=700 waveform with a
sparse top-K transport-event tensor ``[H, W, K, 5]`` (t, delta_t, a, w, mask) and
trains a 2D U-Net to segment each event into {noise, object, glass, ghost}.
"""
