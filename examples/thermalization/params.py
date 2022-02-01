from ising.utils.file_utils import figPath

VERBOSE = 4

# ===================================
# Parameters for models
# ===================================
INTEGRABLE = False
# Ls = [8, 10, 12, 14, 16, 18]
Ls = [16, 18]
MODELS = ['MFIM', 'XXZ', 'ZXXXXZZ']


figBasicPath = figPath+'thermalization/basic/'
figLargeOpsPath = figPath+'thermalization/largeops/'
