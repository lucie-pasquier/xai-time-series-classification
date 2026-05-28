
* CLASS IMBALANCE
ECG200 is mildly imbalanced (~65/35) and remember this when you start training models. Accuracy alone is a weak metric on imbalanced data — a model predicting "always normal" would score 64% on the test set without learning anything. When you get to Model 1, you'll want to report F1 or balanced accuracy alongside raw accuracy

* NORMALISATION
One subtle thing worth noticing, because it's interesting and you'll want to mention it in your thesis: the raw and normalised plots look extremely similar here. That's actually informative. It means the raw ECG200 data is already roughly zero-centred and unit-scaled — the UCR archive often pre-normalises its datasets. Per-sample z-scoring is still doing real work (it's removing any residual per-sample drift, and it's a defensive practice for reproducibility), but on this particular dataset the visual difference is small. On a raw clinical ECG dataset with millivolt-scale signals and electrode-offset drift, the difference would be dramatic. Worth a sentence in your final report explaining the choice and acknowledging it's a small effect on this dataset

