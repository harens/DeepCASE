# DeepCASE Memory Fix Report

This note records the maintenance fix for the DeepCASE memory issue that
caused AnomaLog's HDFS DeepCASE run to OOM.

## Scope

The change set keeps the DeepCASE methodology intact and focuses on memory
behavior:

- Preserve CUDA support.
- Move only mini-batches to the model device.
- Avoid full-dataset `torch.as_tensor(..., device=...)` materialization.
- Chunk prediction and attention processing.
- Keep clustering semantics unchanged.

## Root Cause

The upstream implementation repeatedly materialized full datasets before
batching:

- `ContextBuilder.fit()` copied the entire dataset to the active device up
  front.
- `ContextBuilder.predict()` deduplicated and forwarded the whole input in one
  call.
- `ContextBuilder.query()` processed the entire unique set before returning.
- `Interpreter.predict()` and `Interpreter.attended_context()` built large
  global tensors before thresholding and vectorization.
- `Interpreter.vectorize()` accumulated sparse columns through repeated sparse
  additions.

That pattern made GPU/host memory scale with the full HDFS run size instead of
the requested batch size.

## Maintenance Fixes

Changed files:

- `deepcase/context_builder/context_builder.py`
- `deepcase/context_builder/optimizer.py`
- `deepcase/interpreter/interpreter.py`
- `deepcase/preprocessing/preprocessor.py`
- `deepcase/utils.py`

Key changes:

- `ContextBuilder.fit()` now keeps the training set on host memory and moves
  only each batch to the model device.
- `ContextBuilder.predict()` now chunks inference and deduplicates within each
  batch only.
- `ContextBuilder.query()` now runs in eval mode for deterministic chunked
  attention optimization.
- `Interpreter.predict()` now processes the input in bounded chunks rather than
  materializing the full prediction set first.
- `Interpreter.attended_context()` now streams attention and sparse vector
  creation per batch.
- `Interpreter.vectorize()` now builds sparse matrices in a single pass.
- `VarAdam.step()` now matches the base optimizer signature.
- `deepcase.utils` now uses the correct PyTorch `dim=` keyword and only
  constructs the mapping vectorizer when a mapping is present.
- The preprocessing CLI now passes `length=args.context` to
  `Preprocessor(...)`.

## Verification

Environment:

- Python 3.10 from `env/`
- `torch`, `numpy`, `scipy`, `scikit-learn`, `tqdm`, `pandas`, `argformat`
  installed into that env

Checks:

- `env/bin/ty check --python env deepcase` passed.
- `env/bin/python -m py_compile ...` passed for the touched modules.

Synthetic results:

- Fit smoke test on generated data passed.
- `ContextBuilder.predict()` matched across batch sizes to numerical tolerance:
  - max abs diff in confidence: `4.77e-7`
  - max abs diff in attention: `2.98e-8`
- `Interpreter.attended_context()` preserved the mask across batch sizes and
  produced close sparse vectors on deterministic synthetic data:
  - mask equality: `True`
  - max abs vector diff: `0.0206`
  - mean abs vector diff: `0.0003818`
- Memory proxy on 50k synthetic contexts showed the bounded path keeping the
  largest forward batch at `256` rather than `50000`:
  - batched RSS: `481.75 MB`
  - single-batch RSS: `995.92 MB`

These synthetic tests demonstrate that the full-batch materialization problem is
fixed. They do not prove the specific HDFS job is fixed; the downstream rerun
still needs to validate the real dataset path.

## Recommendation

This should be treated as an **accepted deviation in the maintained baseline**,
not an experimental branch.

Reasoning:

- The patch fixes a memory-safety bug and does not change the scientific model
  meaning.
- CUDA support remains intact.
- The behavior is methodologically equivalent, just bounded by chunk size.
- Splitting this into an experimental branch would add unnecessary
  maintenance overhead for a correctness fix.

AnomaLog should still rerun the HDFS DeepCASE job after merging this baseline
fix and confirm the full production path stays within memory limits.
