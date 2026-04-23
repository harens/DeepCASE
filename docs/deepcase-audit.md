# DeepCASE Paper-to-Code Audit

## Overview

This document presents an audit of the upstream [DeepCASE](https://github.com/Thijsvanede/DeepCASE) implementation, comparing the [original paper](https://thijsvane.de/static/homepage/papers/deepcase.pdf) specification with the behaviour observed in the official repository and documenting the final state of this [fork](https://github.com/harens/DeepCASE) following the audit.

The purpose of this audit is to establish a clear and reproducible account of how the released implementation behaves in practice, and to systematically identify and categorise divergences from the DeepCASE paper.

These divergences include differences in training objectives, model architecture, and data handling, many of which materially affect the training signal, learned representations, and downstream clustering behaviour.

This work was conducted as part of investigations within the [AnomaLog](https://github.com/harens/anomalog) project. The goal was to preserve upstream behaviour where possible, correct confirmed defects, and explicitly document all observed deviations from the paper.

The resulting system should be understood as:

- A corrected and stabilised implementation of the upstream repository.
- Behaviourally aligned with the upstream code, except where correctness issues were identified.
- A baseline for controlled experimentation, with deviations from the paper explicitly isolated.

---

## Methodology

The audit followed a structured approach:

1. **Specification Review**  
   The DeepCASE paper was reviewed to determine explicit and implicit behavioural requirements across preprocessing, model architecture, training, and evaluation.

2. **Implementation Inspection**  
   The [official repository](https://github.com/Thijsvanede/DeepCASE) was then examined across the different modules.

3. **Divergence Classification**  
   Observed differences were categorised according to their implications for correctness and reproducibility:
   - **Bug**: Clear violation of API expectations or internal consistency
   - **Paper divergence**: Behaviour differs from the paper specification and alters the training objective, model parameterisation, or inference procedure
   - **Ambiguous**: The paper does not specify sufficient detail to uniquely determine the implementation, and the observed behaviour therefore reflects an implicit modelling assumption
   - **Minor**: Low-impact or implementation detail unlikely to affect model behaviour or outcomes

4. **Resolution Strategy**  
   Each finding was resolved according to its classification:

   - **Bugs** were corrected to restore expected behaviour
   - **Paper divergences** were either:
      - **Accepted as deviations**, where changes would alter model behaviour without prior validation, or
      - **Isolated as experimental variants**, where paper-faithful alternatives were implemented
   - **Ambiguous behaviours** were preserved and documented as implicit modelling assumptions
   - **Minor issues** were documented without modification

### Terminology 

To ensure clarity, the following terms are used consistently throughout this document:

* **Upstream/Original implementation**
  The original DeepCASE repository as published by the authors, without modifications. Specifically v1.0.3 from [GitHub](https://github.com/Thijsvanede/DeepCASE) as opposed to v1.0.1 on [PyPi](https://pypi.org/project/deepcase/).

* **Maintained baseline/fork**
  The [main branch](https://github.com/harens/DeepCASE/tree/main) of this fork, incorporating only safe maintenance fixes (e.g. API correctness) and not intended to change model behaviour.

* **Accepted deviation**
  A difference between the upstream implementation and the paper specification that has been intentionally preserved in the maintained baseline. These are not treated as bugs, either because they are ambiguous in the paper or because changing them would alter model behaviour without prior validation.

* **Experimental variant**
  A branch or modification that intentionally changes model behaviour in order to more closely match the paper or explore an alternative design. These are isolated to allow controlled evaluation.

* **Paper-faithful**
  Refers to behaviour that more closely matches the description in the DeepCASE paper. This does not imply that the paper provides a complete or uniquely correct specification.

---

## Final State Summary

Following the audit and code changes:

- All **confirmed bugs have been resolved** in the maintained baseline without intentionally altering model semantics.
- **Paper divergences have been reviewed individually**, with only low-risk modifications applied as experimental variants.
- **Ambiguous areas of the paper have been preserved** as implemented upstream.
- **Provenance has been kept explicit at the branch and commit level** so that future AnomaLog experiments can attribute behavioural effects correctly.

This audit produced a maintained baseline implementation alongside a set of isolated experimental variants, allowing behavioural changes to be evaluated independently.

---

## Summary of Findings and Resolutions

| Finding                                                                           | Category         | Resolution                                                                                                | Status                                                                    |
| --------------------------------------------------------------------------------- | ---------------- | --------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Label smoothing uses uniform distribution instead of frequency-based smoothing    | Paper divergence | Baseline left uniform; `delta` made configurable in baseline; frequency-based variant isolated separately | Accepted deviation in baseline; experimental paper-faithful branch exists |
| Padding / `NO_EVENT` treated as a real event                                      | Paper divergence | Left unchanged in baseline; masking semantics isolated separately                                         | Accepted deviation in baseline; experimental branch exists                |
| `DecoderEvent` hidden dimension differs from paper                                | Paper divergence | Paper-aligned configurable variant implemented separately                                                 | Accepted deviation in baseline; experimental branch exists                |
| Attention decoder uses recurrent architecture                                     | Ambiguous        | Left unchanged                                                                                            | Documented                                                                |
| Manual evaluation workflow outside the scope of the package implementation                                        | Paper divergence | Outside the scope of the package implementation                                                                           | Accepted deviation                                                        |
| Workload-reduction metrics missing                                                | Paper divergence | Outside the scope of the package implementation                                                                           | Accepted deviation                                                        |
| Incremental manual update workflow missing                                        | Ambiguous        | Outside the scope of the package implementation                                                                                           | Documented                                                                |
| Preprocessor output ordering subtlety                                             | Minor            | Documented                                                                                                | Documented                                                                |
| Training epoch default differs (10 vs 100)                                        | Paper divergence | Left configurable                                                                                         | Accepted deviation                                                        |
| Attention query confidence filtering                                              | Ambiguous        | Left unchanged                                                                                            | Documented                                                                |
| Clustering grouped by target event                                                | Ambiguous        | Left unchanged                                                                                            | Documented                                                                |
| Loss progress reporting incorrect                                                 | Bug              | Fixed                                                                                                     | Resolved                                                                  |
| Label smoothing delta appeared configurable but was  hardcoded | Bug | Made configurable (commit 6676db5) | Resolved |
| `Interpreter.fit_predict()` ignores caller-provided parameters in `predict()`     | Bug              | Fixed                                                                                                     | Resolved                                                                  |
| `ContextBuilder.load()` cannot reliably restore non-default architecture settings | Bug              | Fixed in maintenance branch with compatibility fallback                                                   | Resolved                           |
| Attention vectors rounded before clustering                                       | Minor            | Documented                                                                                                | Documented                                                                |


---

## Detailed Findings

### Label Smoothing Distribution

The DeepCASE paper specifies that label smoothing should assign probability mass of `1 - delta` to the true event, with the remaining `delta` distributed across other events according to their empirical frequency distribution. This implies a non-uniform smoothing prior.

In the observed implementation, label smoothing distributes the non-target probability mass uniformly across all other classes (`delta / (n - 1)`), without incorporating any frequency information. Additionally, no class-frequency prior is computed or passed into the loss function.

This is a substantive modelling difference rather than a superficial implementation detail. Frequency-based smoothing biases the target distribution toward events that are more common in the data, whereas uniform smoothing treats all incorrect classes as equally plausible. Changing between these formulations alters the training signal and can affect the representations learned by the model.

**Resolution:**
This behaviour was not modified in the maintained baseline. Instead, a paper-faithful variant was implemented separately in the branch [`paper/label-smoothing-frequency`](https://github.com/harens/DeepCASE/tree/paper/label-smoothing-frequency).

In this variant, `LabelSmoothing` accepts an empirical event-frequency distribution, and `ContextBuilder.fit()` computes this distribution from the training targets. A compatibility fallback is retained: if no distribution is provided, the original uniform smoothing behaviour is used.

This separation was intentional. Because the change affects the training objective, it was treated as a research variant rather than a maintenance fix, allowing its impact to be evaluated independently.

**Final Status:**
Accepted deviation from the paper in the baseline; paper-faithful alternative available via a dedicated branch.


---

### Padding and `NO_EVENT` Semantics

The paper describes constructing context windows using left-padding when insufficient prior events exist but does not define padding as a semantic event.

In the implementation, padding is represented explicitly as a `NO_EVENT` token that is included in the event vocabulary. This token is processed identically to real events throughout the pipeline:
- It is embedded and passed through the model
- It contributes to attention mechanisms
- It is included in clustering via attention-derived vectors

This effectively treats padding as a meaningful event rather than a neutral placeholder, introducing a systematic bias whereby shorter histories are represented by repeated synthetic tokens that participate in attention and clustering. As a result, sequence length and padding structure can influence learned representations and cluster assignments.

**Resolution:**  
This is an important divergence, because padding then affects attention allocation and downstream cluster structure. At the same time, the paper is not explicit enough to make this a straightforward bug.

For that reason, the maintained baseline was left unchanged, while an opt-in experimental branch, [`paper/padding-mask-semantics`](https://github.com/harens/DeepCASE/tree/paper/padding-mask-semantics), was created. That branch introduces a configurable `padding_idx` approach. `Preprocessor` exposes a stable `NO_EVENT_INDEX`, `ContextBuilder` accepts an optional padding index and can mask that index from attention, and the `Interpreter` can omit that index from vectorisation when configured. When `padding_idx=None`, the original behaviour is preserved.

**Final Status:**  
Accepted deviation from the paper in the baseline; experimental branch exists for padding-masked semantics.

---

### `DecoderEvent` Hidden Dimension

The paper specifies that the event decoder includes a 128-dimensional ReLU hidden layer prior to output.

The implementation instead defines the hidden layer as:

```python
nn.Linear(input_size, input_size)
```

where `input_size` corresponds to the vocabulary size.

This results in a qualitatively different parameterisation, where model capacity scales with vocabulary size rather than being fixed. For realistic log vocabularies, this can substantially increase model size and alter the inductive bias of the decoder.

**Resolution:**  
Because changing this layer affects parameterisation and learned representations, it was not appropriate to include the paper-aligned version directly into the maintained dependency baseline without explicit evaluation.

Instead, the paper-faithful variant was implemented in [`paper/decoder-event-hidden-128`](https://github.com/harens/DeepCASE/tree/paper/decoder-event-hidden-128). On that branch, the event-decoder hidden size becomes configurable. Direct `DecoderEvent(...)` calls preserve the old vocabulary-width behaviour when no hidden size is supplied, which helps maintain local compatibility for code that constructs the decoder directly. By contrast, the main `ContextBuilder` and `DeepCASE` constructor paths default to `decoder_event_hidden_size=128`, which is the paper-faithful setting.

**Final Status:**  
Accepted deviation in the maintained baseline; paper-faithful architectural alternative available as an experimental branch.

---

### Attention Decoder Architecture

The paper describes attention computation using linear layers and softmax but does not fully specify how attention is conditioned on previous outputs.

The upstream implementation uses a recurrent decoder (GRU or LSTM), where:
- The previous decoder input is embedded
- A recurrent step is applied
- Attention weights are produced from the recurrent state

This creates a temporal dependency on prior decoder inputs and is not equivalent to a purely feedforward attention mechanism.

**Resolution:**  
The paper does not provide sufficient detail to define an alternative, and changing the architecture would significantly alter model behaviour.

Because any rewrite here would amount to a new research hypothesis rather than a safe correction, no change was made and the architecture was preserved as-is.

**Final Status:**  
Documented as an ambiguous but accepted implementation detail.

---

### Manual Analysis and Evaluation Workflow

The DeepCASE paper does not only describe a model, it also presents a broader evaluation methodology involving manual cluster sampling, coverage and reduction metrics, and temporal splits between manual and semi-automatic operation.

The package implementation provides the core clustering and prediction machinery but does not implement the full evaluation harness described in the paper. This is important for reproducibility because a researcher attempting to reproduce the reported evaluation pipeline from the package alone would not obtain the full workflow.

**Resolution:**
No such tooling was added during the audit. This gap was treated as a research-methodology divergence rather than a defect in core execution.

**Final Status:**
Accepted deviation from the paper.

---

### Preprocessor Output Ordering

The preprocessing pipeline sorts events by timestamp internally when constructing context windows. However, it returns output tensors indexed according to the original dataframe order.

This creates a subtle distinction:
- Context construction is chronological (correct)
- Output ordering is not guaranteed to be chronological

This does not affect training or inference when operating on the full dataset, but it can invalidate assumptions in downstream workflows that rely on temporal ordering. In particular, performing a naive split on the preprocessed tensors may not correspond to a temporal split.

For example:

```python
# Context matrix X, target labels y
X, y = preprocessor.sequence(df)

X_train = X[:n]
X_test  = X[n:]
```

does not guarantee that `X_train` contains earlier events than `X_test`.

To ensure valid temporal evaluation, input data should be explicitly sorted or split prior to preprocessing.

**Resolution:**  
No change to behaviour.

**Final Status:**  
Documented.

---

### Training Epoch Defaults

The paper reports training the model for 100 epochs during evaluation, whereas the implementation defaults to 10 epochs.

**Resolution:**  
No change was made, as the parameter is exposed and does not constrain reproducibility when explicitly set.

**Final Status:**  
Accepted deviation from paper defaults.

---

### Safe Maintenance Bugs

#### Label Smoothing Parameter (`delta`) Configuration

Although the paper treats `delta` as a tunable parameter, and the CLI in the original implementation provided a `--delta` option, it is instead hardcoded to a fixed value (`0.1`)

This limits reproducibility and gives a false impression of controlled experimentation with different smoothing strengths.

**Resolution:**
This was addressed in [`fix/label-smoothing-delta-config`](https://github.com/harens/DeepCASE/tree/fix/label-smoothing-delta-config) and merged into the maintained baseline, making `delta` configurable and propagating it consistently.

The default value remains unchanged, as recommended in the paper, so baseline behaviour is preserved.

Upstream PR: [#12](https://github.com/Thijsvanede/DeepCASE/pull/12)

#### `Interpreter.fit_predict()`

The method forwards user-provided parameters (`iterations`, `batch_size`, `verbose`) into `fit`, but ignores them in the subsequent `predict` call, instead using hardcoded defaults.

**Resolution:**  
This bug was fixed in branch [`fix/interpreter-api-and-logging`](https://github.com/harens/DeepCASE/tree/fix/interpreter-api-and-logging) and merged into the maintained baseline, so that the user-supplied values are preserved into the prediction step. This change may alter runtime characteristics for callers that implicitly relied on the previous hardcoded defaults, but restores consistency with the documented interface.

Upstream PR: [#14](https://github.com/Thijsvanede/DeepCASE/pull/14)

---

#### Loss Progress Reporting

Loss values were inconsistently normalised when reported during training, due to double aggregation and incorrect scaling. The optimisation loss itself was not identified as incorrect, the issue lay in how progress was summarised for the user.

**Resolution:**  
Corrected in [`fix/interpreter-api-and-logging`](https://github.com/harens/DeepCASE/tree/fix/interpreter-api-and-logging) and merged into the maintained baseline, while leaving the underlying optimisation behaviour unchanged.

Upstream PR: [#14](https://github.com/Thijsvanede/DeepCASE/pull/14)

---

#### `ContextBuilder.load()`

During loading, key architectural parameters, such as num_layers and whether an LSTM was used, were not reliably recoverable. This meant that non-default models could not be reconstructed correctly from saved checkpoints.

**Resolution:**
This issue was addressed in [`fix/contextbuilder-load-metadata`](https://github.com/harens/DeepCASE/tree/fix/contextbuilder-load-metadata) and merged into the maintained baseline, with a fallback path for legacy checkpoints.

Because this change affects model persistence, it introduces a potential compatibility boundary and should be validated with explicit round-trip tests (save → load → equivalence) before being relied upon in production or comparative experiments.

Upstream PR: [#13](https://github.com/Thijsvanede/DeepCASE/pull/13)

---

### Other Documented Behaviours

Some additional behaviours were identified and documented without modification. These behaviours are not explicitly described in the paper but are consistent with plausible design choices.

**Attention Query Filtering**  
  
For context, attention querying is applied when the Context Builder fails to correctly predict the observed event. It optimises the attention vector, with model weights fixed, to increase the probability of that observed event.

The paper describes accepting the queried attention when it reaches the confidence threshold τ confidence. The released implementation adds a stricter safeguard, retaining it only where the observed-event confidence also improves over the original attention.

This additional constraint prevents degradative optimisation by enforcing monotonic improvement in observed-event confidence, but it is not described in the paper and so constitutes an undocumented modelling choice.

**Clustering by Target Event**

In the upstream implementation, clustering is performed separately for each observed target event, with DBSCAN applied only within these per-event subsets rather than across all sequences. This prevents sequences with different target events from being grouped together and effectively decomposes the clustering problem into independent per-event subspaces.

While this behaviour is consistent with the model’s event-centric interpretation, the paper does not state this restriction explicitly, making it unclear whether it is an intended design choice or an implementation detail.

**Attention Vector Rounding**  

Attention-derived vectors are rounded to four decimal places prior to clustering. This discretises the input space, altering pairwise distances between samples and potentially changing neighbourhood connectivity at DBSCAN decision boundaries. The effect is likely minor in practice, although experimentation is required.

---

## Provenance and Reproducibility

All audit work and associated experiments were conducted with strict separation between:

- Upstream repository behaviour i.e. [`7d19eaa`](https://github.com/harens/DeepCASE/commit/7d19eaa798da257b83567aeaed78d88bd3574370) from the maintained fork and before
- Maintained fork fixes (non-behavioural)
- Experimental modifications (behavioural)

Each modification was tracked with explicit categorisation and branch-level isolation, allowing behavioural changes to be attributed to individual design differences.

This separation ensures that:
- Bug fixes are not conflated with model changes
- Experimental results can be attributed to specific modifications
- Reproducibility is preserved across iterations

---

## Conclusion

The audit shows that the released DeepCASE implementation should be understood as a specific instantiation within a broader space of plausible designs, rather than as a direct encoding of the paper specification. Several implicit modelling choices, particularly in smoothing, padding, and decoder structure, are sufficient to alter learned representations and clustering behaviour.

By isolating these choices and preserving a stable baseline, this work provides a foundation for controlled experimentation and for more reliable reproduction of prior and future results.
