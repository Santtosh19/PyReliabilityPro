"""
The core module of PyReliabilityPro, providing fundamental reliability
calculations and distribution functions.
"""

from .distributions import weibull_pdf, weibull_cdf, weibull_sf, weibull_hf, weibull_fit

from .metrics import calculate_mttf_exponential, weibull_mttf

# You can define what gets imported when
# a user does 'from pyreliabilitypro.core import *'
# It's generally good practice to be
#  explicit with __all__ if you use 'import *',
# though direct imports like
# 'from pyreliabilitypro.core import weibull_pdf' are preferred.
__all__ = [
    "weibull_pdf",
    "weibull_cdf",
    "weibull_sf",
    "weibull_hf",
    "weibull_fit",
    "calculate_mttf_exponential",
    "weibull_mttf",
]
