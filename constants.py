# Path to MATLAB file with UCI data
UCI_PATH = "data/uci/mixoutALL_shifted.mat"

# Location of train/test split indices
TRAIN_INDICES = "out/train_indices.csv"
TEST_INDICES = "out/test_indices.csv"

# Save .ts output to
TRAIN_OUT = "out/CharacterTrajectories_TRAIN.ts"
TEST_OUT = "out/CharacterTrajectories_TEST.ts"

# Comment to add to .ts files
COMMENT = """# Version 3: November 2023
# ------------------------
#
# Updated to remove padding and fix the 39 trajectories with unequal channel lengths.
#
# The 39 cases have shorter *x* or *y* channels after removing padding. The shorter
# channels have a final observation in the range 1e-12 to 1e-14. The data was
# differentiated therefore small *x* and *y* values correspond to a broadly stationary
# pen after drawing the character.
#
# It is hypothesised that the smoothing carried out prior to differentiation resulted in
# final values that were zero (or small enough for numeric underflow). These shorter
# channels have been filled with trailing zero values such that all channels are the 
# same length.
#
# Note that the numeric accuracy in this version is the same as the UCI data. This is
# greater than version 1.
#
# Philip Darke
# 21 November 2023
#
# Version 2: October 2023
# ------------------------
#
# Reverted to UCI version of data set with initial and trailing zero padding.
#
# Tony Bagnall
# 25 October 2023
#
# Version 1: 2018
# ---------------
#
# Ben H Williams 
# School of Informatics, 
# University of Edinburgh, 
# ben.williams '@' ed.ac.uk 
#
# Data Set Information:
#
# The characters here were used for a PhD study on primitive extraction using HMM based 
# models. The data consists of 2858 character samples, The data was captured using a
# WACOM tablet. 3 Dimensions were kept - x, y, and pen tip force. The data has been
# numerically differentiated and Gaussian smoothed, with a sigma value of 2. Data was
# captured at 200Hz. The data was normalised. Only characters with a single 'PEN-DOWN'
# segment were considered. 
#
# Character segmentation was performed using a pen tip force cut-off point. The
# characters have also been shifted so that their velocity profiles best match the mean
# of the set.
#
# Each instance is a 3-dimensional pen tip velocity trajectory. The original data has
# different length cases. The class label is one of 20 characters: a, b, c, d, e, g, h,
# l, m, n, o, p, q, r, s, t, u, v, w, y, z
#
# To conform with the repository, we have truncated all series to the length of the
# shortest, which is 182, which will no doubt make classification harder. 
#
# Relevant Papers:
#
# B.H. Williams, M.Toussaint, and A.J. Storkey. Extracting motion primitives from natural handwriting data. In ICANN, volume 2, pages 634-643, 2006. 
#
# B.H. Williams, M.Toussaint, and A.J. Storkey. A primitive based generative model to infer timing information in unpartitioned handwriting data. In IJCAI, pages 1119-1124, 2007. 
#
# B.H. Williams, M. Toussaint, and A.J. Storkey. Modelling motion primitives and their timing in biologically executed movements. In J.C. Platt, D. Koller, Y. Singer, and S. Roweis, editors, Advances in Neural Information Processing Systems 20, pages 1609-1616. MIT Press, Cambridge, MA, 2008.
#
"""
