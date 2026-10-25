# Small U-Net model contract

The first controlled model candidate is a compact 2D U-Net. It predicts the normalized
future radar reflectivity field from one current input tensor. It does not predict rainfall,
lightning classes or an autoregressive sequence.

## Architecture

| Stage | Channels | Spatial shape for a 256×256 input |
| --- | ---: | --- |
| Input | 2 radar-only or 8 multimodal | 256×256 |
| Encoder block 1 | 16 | 256×256 |
| Encoder block 2 | 32 | 128×128 |
| Encoder block 3 | 64 | 64×64 |
| Bottleneck | 128 | 32×32 |
| Decoder block 3 | 64 | 64×64 |
| Decoder block 2 | 32 | 128×128 |
| Decoder block 1 | 16 | 256×256 |
| Output 1×1 convolution | 1 | 256×256 |

Each encoder/bottleneck/decoder block uses two padded 3×3 convolutions, GroupNorm with
eight groups and ReLU. Max pooling halves spatial size. A 2×2 transposed convolution
upsamples, then the decoder concatenates the matching encoder feature through a skip
connection. GroupNorm does not estimate batch-wide running statistics, which is useful for
the small batches expected at 256×256 resolution.

The final 1×1 convolution has no sigmoid, softmax or clipping. Normalized reflectivity is a
continuous regression target and can be negative or positive. Physical dBZ is recovered
with the training radar mean and standard deviation.

The radar-only model has 482,593 trainable parameters. The multimodal model has 483,457;
the 864-parameter difference is exactly the six additional first-layer channels times
16 filters times a 3×3 kernel. All later layers are identical, supporting a controlled
radar-only versus multimodal comparison.

## Masked training objective

The smoke check uses masked mean squared error over valid target pixels:

\[
L = \frac{\sum_{b,i,j}M_y(b,i,j)(\hat{Y}(b,i,j)-Y(b,i,j))^2}
         {\sum_{b,i,j}M_y(b,i,j)}.
\]

`M_y` is the boolean target-valid mask. Input masks are already provided as channels; a
missing target cannot contribute to training loss. The primary validation comparison later
uses the stricter common input-radar/target domain fixed in `configs/evaluation.yaml` so it
can be compared fairly with persistence.

## Verified smoke checks

Real-data batch size one checks passed for radar-only and multimodal inputs on CPU. Shapes
were `(1,2,256,256)` or `(1,8,256,256)` in and `(1,1,256,256)` out. All 50 trainable
parameter tensors received finite gradients. The multimodal forward/backward check also
passed on Apple MPS.

The random-initialization loss and gradient norm reported by the smoke command are diagnostics,
not forecasting skill. That command creates no optimizer, performs no optimizer step and
writes no checkpoint. Official training was performed later through a separate controlled
runner and is reported in `training_results.md`.

The architecture follows the encoder-decoder and skip-connection idea introduced by
Ronneberger, Fischer and Brox, *U-Net: Convolutional Networks for Biomedical Image
Segmentation* (2015), adapted here for continuous radar regression.
