# Method and implementation map

This implementation follows *scDVD: single-cell Dual-view Diffusion for
Perturbation Prediction*, the manuscript supplied for the November 2026 refactor.
The reference numerical implementation comes from the completed final set4 runs.

| Manuscript component | Implementation |
|---|---|
| Matched training-control anchor, Eq. 1 | `scdvd/data/unpaired.py:UnpairedH5ADStore` |
| Treated/control sets, padding and regrouping | `scdvd/data/unpaired.py`, `scdvd/data/datamodule.py` |
| Drug/dose condition | `scdvd/models/condition.py:DoseGatedMolecularEncoder` |
| Response diffusion, Eq. 4 | `scdvd/diffusion/gaussian.py`, `scdvd/module.py` |
| Mean, cell and population views, Eqs. 5–6 | `scdvd/models/dual_view.py:DualViewDenoiser` |
| Positive MSE and energy, Eqs. 7–8 | `scdvd/losses.py`, `scdvd/module.py:_shared_step` |
| DDIM with classifier-free guidance | `scdvd/module.py:_sample_population` |
| Sparsity predictor and empirical masks, Eq. 9 | `scdvd/mask_module.py`, `scdvd/masking.py` |
| Condition-level ALL/DE20/DE40 metrics | `scdvd/evaluate.py`, `scdvd/evaluation.py` |

## Canonical model

The raw expression prediction is

`X_hat = A + mean_response + centered_cell_deviation + centered_population_correction`.

Both deviations are centered over all four target slots. There is no population
gate multiplier or RMS bound. Dose gating and modulation gates inside residual
blocks are retained. The unused population-gate module is initialized and then
removed, exactly as in the frozen final-run source, to preserve initialization
random-number consumption and the checkpoint parameter layout.

Historical gated/capped constructors and objectives remain supported for loading
old checkpoints. They are not the defaults in the paper configs and must not be
reported as canonical scDVD. Set16 is a historical sensitivity comparison.

## Numerical conventions

- Anchor and control spread use all matching outer-training controls, not the
  sampled four control cells. Missing context–batch control pools are errors.
- Expression is divided by the outer-training maximum and clipped to [0,1].
  This maximum is calculated before the internal train/validation split.
- The fixed 2,000-HVG representation was selected before benchmark splitting.
  Raw preprocessing and HVG selection are not repeated by the model loader.
- Target padding has zero direct reconstruction and energy-loss weight. Padded
  slots still participate in attention and centering; there is no attention
  padding mask in the final model.
- Shared timestep per set; independent Gaussian noise per cell; direct clean
  response prediction. The condition-only mean has no noisy-target or timestep
  input. `detach_population_context=true` preserves the frozen gradient routing.
- Both losses supervise the combined prediction. `mmd_weight=0.001` is the
  historical config key for the **GeomLoss energy coefficient**, not an RBF MMD
  coefficient in the paper configuration.
- Training calls `SamplesLoss(loss="energy", blur=0.05)`. This discrepancy is
  half the raw energy expression. The evaluation ED uses its established
  unbiased within-population estimator, truncation at zero and square root;
  it is not numerically identical to the training loss.
- AdamW with cosine annealing, despite the old YAML's `reduce_on_plateau`
  label. The new config labels the existing implemented scheduler correctly.
- DDIM uses 50 steps, guidance 1.5, self-conditioning, and Gaussian initialization.
  Final absolute expression is clipped before applying the empirical mask.
- Mask retrieval searches only outer-training treated cells, including internal
  validation cells. It uses the top 300 predicted-presence genes, one nearest
  empirical condition and whole binary patterns, with 0.01/0.99 thresholds.
  Candidates are not separately restricted to the queried cell type.

The supplied main manuscript refers to Supplementary Methods, but no separate
supplement was present. These details are therefore documented from actual final
configs and frozen sources, rather than inferred from an unavailable supplement.
