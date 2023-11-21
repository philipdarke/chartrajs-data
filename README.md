# Prepare CharacterTrajectories data set

Prepare the [CharacterTrajectories](https://archive.ics.uci.edu/dataset/175/character+trajectories) data set for modelling. The code:

- removes padding
- handles cases with inconsistent channel lengths
- splits the data into the same training/test splits used in the [Time Series Classification](https://www.timeseriesclassification.com/) (TSC) repository version of the data set
- saves the output in [aeon](https://aeon-toolkit.org/) `.ts` [format](https://www.aeon-toolkit.org/en/latest/api_reference/file_specifications/ts.html)

This resolves issues [time-series-machine-learning/tsml-repo#92](https://github.com/time-series-machine-learning/tsml-repo/issues/92) and [aeon-toolkit/aeon#853](https://github.com/aeon-toolkit/aeon/issues/853).

## Instructions

1. Download the UCI version of the data from [here](https://archive.ics.uci.edu/dataset/175/character+trajectories), extract and save in `data\uci`
2. Set up a Python virtual environment with the dependencies in `requirements.txt`
3. Run `process_data.py` to save `CharacterTrajectories_TRAIN.ts` and `CharacterTrajectories_TEST.ts` in the `out\` directory.

The file paths can be changed in `constants.py`.

## Handling cases with inconsistent channel lengths

39 cases have shorter *x* or *y* channels after removing padding. The shorter channels have a final observation in the range 1e-12 to 1e-14. The data was differentiated therefore small *x* and *y* values correspond to a broadly stationary pen after drawing the character.

It is hypothesised that the smoothing carried out prior to differentiation resulted in final values that were zero (or small enough for numeric underflow). These shorter channels have been filled with trailing zero values such that all channels are the same length.

## Training and test splits

The `out/train_indices.csv` and `out/test_indices.csv` files include the UCI index of each TSC training/test case. These are used to split the data consistently with the TSC repository.

The first (non-zero) *x*, *y* and *z* observation for each training/test case was compared with each UCI case. A match was found where a) the class label was the same and b) the value of each channel was within 1e-6 to allow for rounding differences. To reproduce the UCI indices:

1. Download the *Univariate aeon formatted ts files* from [here](http://www.timeseriesclassification.com/aeon-toolkit/Archives/Univariate2018_ts.zip), extract and save the contents of the `Multivariate_ts\CharacterTrajectories\` directory to `data\aeon`
2. Run `infer_indices.py`

## Licence

Made available under the MIT License. See the above repositories for licensing of the original data.
