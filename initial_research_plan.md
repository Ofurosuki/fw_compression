You are helping implement an initial research experiment for full-waveform LiDAR compression.

Background:
We want to test whether full-waveform LiDAR can be replaced by a lightweight sensor-side encoder followed by a DNN decoder, while preserving the downstream performance of Ghost-FWL-style ghost detection/removal.

Prior compression papers such as CVPR2022 Compressive Single-Photon 3D Cameras and ICCV2023 Learned Compressive Representations focus mainly on depth-preserving compression. Our interest is different: we want to test whether compressed waveform representations can preserve transport-related information such as multi-peak structure, ghost returns, multipath, and secondary peaks.

Initial experiment goal:
Build a simple autoencoder-style pipeline:

full waveform x
→ simple encoder E(x)
→ compressed latent z
→ DNN decoder D(z)
→ reconstructed pseudo-waveform x_hat
→ existing Ghost-FWL pipeline
→ evaluate score degradation

The key question:
How much can we compress the waveform before Ghost-FWL downstream performance significantly degrades?

Implementation requirements:

1. Dataset / input
- Assume each waveform is a 1D tensor x of shape [T].
- Also support batched tensors [B, T].
- Later this may extend to [B, H, W, T], but start with 1D waveform compression.

2. Encoders to implement
Implement the following encoder types:

A. Coarse binning encoder
- Divide T into K bins.
- Sum or average each bin.
- Output z of shape [B, K].

B. Fixed random projection encoder
- Matrix C of shape [K, T].
- z = x @ C.T.
- Normalize rows of C.

C. DCT/Fourier low-frequency encoder
- Keep first K frequency coefficients.
- Output z of shape [B, K].

D. Learnable linear encoder
- C is nn.Parameter with shape [K, T].
- z = x @ C.T.
- Row-normalize C during forward.

3. Decoder
Implement a DNN decoder:

z [B, K]
→ MLP or lightweight 1D decoder
→ x_hat [B, T]

Start with an MLP:
Linear(K, 256) → ReLU → Linear(256, 512) → ReLU → Linear(512, T)

Optionally support deeper decoder later.

4. Training loss
Train the autoencoder first with reconstruction losses:
- MSE(x_hat, x)
- Optional peak-aware loss if peak labels exist
- Optional intensity/energy preservation loss:
  abs(sum(x_hat) - sum(x))

But structure the code so that later we can replace or add downstream Ghost-FWL loss.

5. Evaluation
Implement evaluation metrics:
- Waveform MSE
- Peak localization error
- Peak count preservation if peak detector is available
- Energy / intensity error
- Compression ratio T / K

Most importantly, save reconstructed waveforms x_hat so they can be passed into Ghost-FWL for downstream evaluation.

6. Experiment sweep
Run K sweep:
K = [8, 16, 32, 64, 128]

Compare:
- full waveform upper bound
- coarse binning
- random projection
- DCT/Fourier
- learnable linear encoder

7. Outputs
For each encoder and K, save:
- trained model checkpoint
- reconstruction metrics JSON
- example plots of original vs reconstructed waveforms
- compressed latent z
- reconstructed waveform x_hat

8. Code structure
Please create clean PyTorch code with:
- encoders.py
- decoders.py
- autoencoder.py
- train_autoencoder.py
- evaluate_autoencoder.py
- utils/metrics.py
- utils/plot.py

9. Important design philosophy
Do NOT overcomplicate the encoder.
The encoder is intended to approximate lightweight sensor-side computation.
The decoder can be DNN-heavy because it is assumed to run off-sensor.

10. Research hypothesis
We want to test whether:
- depth-oriented compression loses ghost/multipath information,
- learned lightweight compression preserves more transport information,
- Ghost-FWL downstream scores degrade gracefully with compressed pseudo-waveforms.

Please implement the initial version with synthetic/random waveform data support if real data paths are not available, but make the dataloader easy to replace with real Ghost-FWL waveform datasets.

11. Source
Papers of CVPR2022 Compressive Single-Photon 3D Cameras and ICCV2023 Learned Compressive Representations focus mainly on depth-preserving compression are located in the same direcotry named papers. You can read them to comprehend the basic concept of compression for SPAD data and details of the baselines.