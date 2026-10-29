# Training stability repair

Changes apply only to this new source folder.

- Supervised CE sums unreduced FP32 losses and divides by the count of valid labels, clamped to one. All-ignore targets yield differentiable zero. No batch is silently discarded. Other losses still train on valid unlabeled regions.
- Nonfinite supervised logits/losses are fatal and save a diagnostic batch. Nonfinite gradient norm is fatal before optimizer update.
- BF16 remains enabled; no gradient clipping magnitude change (norm checked with infinite limit).
- Periodic atomic latest.pth every 500 iterations after optimizer and EMA updates. Contains epoch, next batch, completed iteration count, optimizer, scaler and models. Existing epoch-end checkpoint format remains readable.
- Resume skips already consumed batch indices within the epoch. This preserves optimizer step accounting but is NOT bitwise deterministic: worker RNG/prefetched augmentations are not restored.
- Fixed validation subset sizing before sampler construction.
- Pretrained weights must be supplied separately; no weights are distributed.

The historical supervised NaN cause is not conclusively established because its batch was not saved. The demonstrated all-ignore CE failure is now handled. Full-duration stability and final paper accuracy are not yet established.
