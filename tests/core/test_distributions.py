# tests/core/test_distributions.py

import pytest
import numpy as np
from pyreliabilitypro.core.distributions import weibull_pdf
from pyreliabilitypro.core.distributions import weibull_cdf  
from pyreliabilitypro.core.distributions import weibull_sf
from pyreliabilitypro.core.distributions import weibull_hf
from pyreliabilitypro.core.distributions import weibull_fit
import scipy.stats as stats


# ---- Test Case 1: Basic 2-parameter Weibull ----
# Lines 7-12: Comments explaining the source of 
# expected values or manual calculation.
# For x=100, β=2, η=100: 
# PDF formula f(x) = (β/η) * (x/η)^(β-1) * exp(-(x/η)^β)
# (2/100) * (100/100)^(2-1) * exp(-(100/100)^2) = 0.02 *
#  1^1 * exp(-1^2) = 0.02 * exp(-1)
# exp(-1) is approx 0.367879.
#  So, 0.02 * 0.367879 = 0.00735758.

def test_weibull_pdf_basic_2_parameter():
    # What: A test for a simple case with a 
    # scalar `x` and 2-parameter Weibull (gamma=0).
    beta = 2.0
    eta = 100.0
    x = 100.0
    expected_pdf = (
        (beta / eta) * ((x / eta) ** (beta - 1)) * np.exp(-((x / eta) ** beta))
    )
    assert weibull_pdf(x, beta, eta) == pytest.approx(expected_pdf, abs=1e-6)
    # Assert: Compares the result with the `expected_pdf`.
    # `pytest.approx(..., abs=1e-6)`: 
    # Compares floats approximately with an absolute tolerance of 1e-6.
    # Useful because floating point math can have tiny precision errors.


# ---- Test Case 2: Array input for x (2-parameter) ----
def test_weibull_pdf_array_x_2_parameter():
    beta = 2.0
    eta = 100.0
    x_arr = np.array([50.0, 100.0, 150.0])
    expected_pdfs = np.array(
        [  # Manually calculating expected PDF for each x in x_arr
            (beta / eta) * ((50 / eta) ** (beta - 1)) * np.exp(-((50 / eta) ** beta)),
            (beta / eta) * ((100 / eta) ** (beta - 1)) * np.exp(-((100 / eta) ** beta)),
            (beta / eta) * ((150 / eta) ** (beta - 1)) * np.exp(-((150 / eta) ** beta)),
        ]
    )
    result = weibull_pdf(x_arr, beta, eta)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected_pdfs, atol=1e-6)


# ---- Test Cases 3 & 4 & 5: 3-Parameter Weibull (gamma != 0) ----
# These tests explore scenarios with a non-zero location parameter `gamma`.

def test_weibull_pdf_3_parameter_at_gamma_beta_gt_1():
    # What: Test case for 3-parameter Weibull where x 
    # is exactly gamma, and beta > 1.
    # Why: For β > 1, the PDF is defined to be 0 at x = γ.
    beta = 2.5  # beta > 1
    eta = 100.0
    gamma = 50.0
    x = 50.0  # x is equal to gamma
    # Expected PDF is 0
    assert weibull_pdf(x, beta, eta, gamma) == pytest.approx(0.0, abs=1e-9)

def test_weibull_pdf_3_parameter_at_gamma_beta_eq_1():
    # What: Test case for 3-parameter Weibull where x is gamma, and beta = 1.
    # Why: When β = 1, the Weibull distribution 
    # reduces to an Exponential distribution
    # (shifted by gamma). The PDF of an exponential 
    # distribution f(t) = λ * exp(-λt)
    # at t=0 (which corresponds to x=gamma here, 
    # since x' = x-gamma) is λ.
    # Here, λ (failure rate) for Weibull with β=1 is 1/η.
    beta = 1.0  # This makes it behave like an Exponential distribution
    eta = 100.0
    gamma = 50.0
    x = 50.0  # x is equal to gamma
    expected_pdf = (
        1.0 / eta
    )  
    # For an exponential, PDF at 
    # the start (t=0, or x=gamma) is 1/scale.
    assert weibull_pdf(x, beta, eta, gamma) == pytest.approx(expected_pdf, abs=1e-6)

def test_weibull_pdf_3_parameter_above_gamma():
    # What: Tests a 3-parameter Weibull where x is greater than gamma.
    # Why: To ensure the (x-gamma) shift is 
    # handled correctly by our function (via SciPy).
    beta = 2.0
    eta = 100.0
    gamma = 20.0
    x = 25.0  # x > gamma
    # For a 3-parameter Weibull, f(x; β, η, γ) = f_2p(x-γ; β, η), 
    # where f_2p is the 2-param PDF.
    # So we calculate expected using x' = x - gamma.
    x_prime = x - gamma  # x_prime = 5.0
    expected_pdf = (
        (beta / eta)
        * ((x_prime / eta) ** (beta - 1))
        * np.exp(-((x_prime / eta) ** beta))
    )
    assert weibull_pdf(x, beta, eta, gamma) == pytest.approx(expected_pdf, abs=1e-6)


# ---- Test Cases for Input Validations ----

def test_weibull_pdf_invalid_beta():
    # What: Tests that a ValueError is raised for non-positive beta.
    with pytest.raises(
        ValueError, match="Shape parameter beta \\(β\\) must be greater than 0."
    ):
        weibull_pdf(x=100, beta=0, eta=100)
    with pytest.raises(
        ValueError, match="Shape parameter beta \\(β\\) must be greater than 0."
    ):
        weibull_pdf(x=100, beta=-1, eta=100)  # Test with another invalid beta

def test_weibull_pdf_invalid_eta():
    # What: Tests for non-positive eta.
    with pytest.raises(
        ValueError, match="Scale parameter eta \\(η\\) must be greater than 0."
    ):
        weibull_pdf(x=100, beta=2, eta=0)
    with pytest.raises(
        ValueError, match="Scale parameter eta \\(η\\) must be greater than 0."
    ):
        weibull_pdf(x=100, beta=2, eta=-10)


def test_weibull_pdf_x_less_than_gamma():
    with pytest.raises(
        ValueError,
        match="All values of x must be greater than or equal to the location parameter gamma",
    ):
        weibull_pdf(x=50, beta=2, eta=100, gamma=60)  # scalar x < gamma
    with pytest.raises(
        ValueError,
        match="All values of x must be greater than or equal to the location parameter gamma",
    ):
        # Testing with an array where one element is invalid
        weibull_pdf(
            x=np.array([50, 55, 60]), beta=2, eta=100, gamma=58
        )  # 50 & 55 are < 58


# ---- Test Cases for Return Types ----
# Why: To explicitly verify that our function 
# returns a standard Python float for scalar input
# and a NumPy ndarray for array input, as per its design.

def test_weibull_pdf_returns_float_for_scalar_input():
    result = weibull_pdf(x=100.0, beta=2.0, eta=100.0)
    assert isinstance(result, float)


def test_weibull_pdf_returns_ndarray_for_array_input():
    # What: Checks return type for array x input.
    result = weibull_pdf(x=np.array([100.0, 110.0]), beta=2.0, eta=100.0)
    assert isinstance(result, np.ndarray)


# -------------------------------------------------------------------------------------------------
# --- Test cases for weibull_cdf ---


# 1. Basic test with known values (2-parameter Weibull, gamma=0)
#    CDF(x) = 1 - exp(-(x/η)^β)
#    For beta=2, eta=100:
#    At x = eta (characteristic life), CDF = 1 - 
#    exp(- (eta/eta)^beta ) = 1 - exp(-1^beta) = 1 - exp(-1)
#    1 - exp(-1) is approx 1 - 0.367879 = 0.632121
def test_weibull_cdf_basic_2_parameter_at_eta():
    beta = 2.0
    eta = 100.0
    x = 100.0  # At characteristic life
    expected_cdf = 1.0 - np.exp(-((x / eta) ** beta))
    assert weibull_cdf(x, beta, eta) == pytest.approx(expected_cdf, abs=1e-6)
    assert 0.0 <= weibull_cdf(x, beta, eta) <= 1.0  # CDF must be between 0 and 1


def test_weibull_cdf_basic_2_parameter_various_x():
    beta = 2.0
    eta = 100.0
    # x well below eta
    x_low = 10.0
    expected_cdf_low = 1.0 - np.exp(-((x_low / eta) ** beta))
    assert weibull_cdf(x_low, beta, eta) == pytest.approx(expected_cdf_low, abs=1e-6)
    assert 0.0 <= weibull_cdf(x_low, beta, eta) <= 1.0

    # x well above eta (probability should be close to 1)
    x_high = 300.0
    expected_cdf_high = 1.0 - np.exp(-((x_high / eta) ** beta))
    assert weibull_cdf(x_high, beta, eta) == pytest.approx(expected_cdf_high, abs=1e-6)
    assert 0.0 <= weibull_cdf(x_high, beta, eta) <= 1.0


# 2. Test CDF with an array of x values (2-parameter)
def test_weibull_cdf_array_x_2_parameter():
    beta = 2.0
    eta = 100.0
    x_arr = np.array([10.0, 100.0, 300.0])
    expected_cdfs = 1.0 - np.exp(-((x_arr / eta) ** beta))  # Vectorized calculation

    result = weibull_cdf(x_arr, beta, eta)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected_cdfs, atol=1e-6)
    assert np.all(result >= 0) and np.all(
        result <= 1
    )  # All CDF values should be in [0,1]


# 3. Test CDF with location parameter gamma 
# (3-parameter Weibull)
#    CDF for x < gamma should be 0.
#    CDF for x = gamma should be 0.
def test_weibull_cdf_3_parameter_at_and_below_gamma():
    beta = 2.0
    eta = 100.0
    gamma = 50.0

    # Test at x = gamma
    assert weibull_cdf(gamma, beta, eta, gamma) == pytest.approx(0.0, abs=1e-9)
    # Test at x < gamma
    assert weibull_cdf(gamma - 10, beta, eta, gamma) == pytest.approx(0.0, abs=1e-9)
    # Test with an array including values below, 
    # at, and above gamma
    x_arr = np.array([gamma - 20, gamma, gamma + 20])
    expected_cdfs_arr = np.array(
        [
            0.0,
            0.0,
            1.0 - np.exp(-(((gamma + 20 - gamma) / eta) ** beta)),  # x_prime = 20
        ]
    )
    result_arr = weibull_cdf(x_arr, beta, eta, gamma)
    assert np.allclose(result_arr, expected_cdfs_arr, atol=1e-6)


# 4. Test CDF value for very large x (should approach 1)
def test_weibull_cdf_large_x():
    beta = 1.5
    eta = 500.0
    # A very large x relative to eta (e.g., 10 * eta or more)
    # For Weibull, if x = 7*eta, CDF > 0.999... for most betas
    # Let's use x significantly larger to be very close to 1
    large_x = eta * 20  # 10000
    assert weibull_cdf(large_x, beta, eta) == pytest.approx(1.0, abs=1e-6)


# --- Input Validation Tests (beta, eta -
#  gamma does not cause error for x < gamma here) ---
def test_weibull_cdf_invalid_beta():
    with pytest.raises(
        ValueError, match="Shape parameter beta \\(β\\) must be greater than 0."
    ):
        weibull_cdf(x=100, beta=0, eta=100)


def test_weibull_cdf_invalid_eta():
    with pytest.raises(
        ValueError, match="Scale parameter eta \\(η\\) must be greater than 0."
    ):
        weibull_cdf(x=100, beta=2, eta=0)


# --- Return Type Tests ---
def test_weibull_cdf_returns_float_for_scalar_input():
    result = weibull_cdf(x=100.0, beta=2.0, eta=100.0)
    assert isinstance(result, float)


def test_weibull_cdf_returns_ndarray_for_array_input():
    result = weibull_cdf(x=np.array([100.0, 110.0]), beta=2.0, eta=100.0)
    assert isinstance(result, np.ndarray)


# ----------------------------------------------------------------------------------------------------


# --- Test cases for weibull_sf ---


# 1. Basic test with known values (2-parameter Weibull, gamma=0)
#    SF(x) = exp(-(x/η)^β)
#    At x = eta (characteristic life), SF = 
#    exp(- (eta/eta)^beta ) = exp(-1^beta) = exp(-1)
#    exp(-1) is approx 0.367879. This is 1 - CDF_at_eta.
def test_weibull_sf_basic_2_parameter_at_eta():
    beta = 2.0
    eta = 100.0
    x = 100.0  # At characteristic life
    expected_sf = np.exp(-((x / eta) ** beta))
    # Or, verify against SciPy:
    # import scipy.stats as stats
    # expected_sf_scipy = stats.weibull_min.sf(x, c=beta, scale=eta, loc=0)
    # print(f"Expected manual SF: {expected_sf}, SciPy SF: {expected_sf_scipy}")
    assert weibull_sf(x, beta, eta) == pytest.approx(expected_sf, abs=1e-6)
    assert 0.0 <= weibull_sf(x, beta, eta) <= 1.0  # SF must be between 0 and 1


def test_weibull_sf_basic_2_parameter_various_x():
    beta = 2.0
    eta = 100.0
    # x well below eta (SF should be close to 1)
    x_low = 10.0
    expected_sf_low = np.exp(-((x_low / eta) ** beta))
    assert weibull_sf(x_low, beta, eta) == pytest.approx(expected_sf_low, abs=1e-6)
    assert 0.0 <= weibull_sf(x_low, beta, eta) <= 1.0

    # x well above eta (SF should be close to 0)
    x_high = 300.0
    expected_sf_high = np.exp(-((x_high / eta) ** beta))
    assert weibull_sf(x_high, beta, eta) == pytest.approx(expected_sf_high, abs=1e-6)
    assert 0.0 <= weibull_sf(x_high, beta, eta) <= 1.0


# 2. Test SF with an array of x values (2-parameter)
def test_weibull_sf_array_x_2_parameter():
    beta = 2.0
    eta = 100.0
    x_arr = np.array([10.0, 100.0, 300.0])
    expected_sfs = np.exp(-((x_arr / eta) ** beta))  # Vectorized calculation

    result = weibull_sf(x_arr, beta, eta)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected_sfs, atol=1e-6)
    assert np.all(result >= 0) and np.all(
        result <= 1
    )  # All SF values should be in [0,1]


# 3. Test SF with location parameter gamma (3-parameter Weibull)
#    SF for x < gamma should be 1.0.
#    SF for x = gamma should be 1.0.
def test_weibull_sf_3_parameter_at_and_below_gamma():
    beta = 2.0
    eta = 100.0
    gamma = 50.0

    # Test at x = gamma
    assert weibull_sf(gamma, beta, eta, gamma) == pytest.approx(1.0, abs=1e-9)
    # Test at x < gamma
    assert weibull_sf(gamma - 10, beta, eta, gamma) == pytest.approx(1.0, abs=1e-9)
    # Test with an array including values below, 
    # at, and above gamma
    x_arr = np.array([gamma - 20, gamma, gamma + 20])
    expected_sfs_arr = np.array(
        [1.0, 1.0, np.exp(-(((gamma + 20 - gamma) / eta) ** beta))]  # x_prime = 20
    )
    result_arr = weibull_sf(x_arr, beta, eta, gamma)
    assert np.allclose(result_arr, expected_sfs_arr, atol=1e-6)


# 4. Test SF value for very large x (should approach 0)
def test_weibull_sf_large_x():
    beta = 1.5
    eta = 500.0
    # A very large x relative to eta
    large_x = eta * 20  # 10000
    assert weibull_sf(large_x, beta, eta) == pytest.approx(0.0, abs=1e-6)


# 5. Test relationship: SF(x) + CDF(x) = 1.0
def test_weibull_sf_plus_cdf_equals_one():
    beta = 2.5
    eta = 150.0
    gamma = 10.0
    x_values = np.array(
        [5.0, 10.0, 50.0, 150.0, 500.0]
    )  # Covers x < gamma, x = gamma, x > gamma

    for x_val in x_values:
        sf_val = weibull_sf(x_val, beta, eta, gamma)
        cdf_val = weibull_cdf(
            x_val, beta, eta, gamma
        )  # Assuming weibull_cdf is correct
        assert sf_val + cdf_val == pytest.approx(1.0, abs=1e-7)


# --- Input Validation Tests (beta, eta) ---
def test_weibull_sf_invalid_beta():
    with pytest.raises(
        ValueError, match="Shape parameter beta \\(β\\) must be greater than 0."
    ):
        weibull_sf(x=100, beta=0, eta=100)


def test_weibull_sf_invalid_eta():
    with pytest.raises(
        ValueError, match="Scale parameter eta \\(η\\) must be greater than 0."
    ):
        weibull_sf(x=100, beta=2, eta=0)


# --- Return Type Tests ---
def test_weibull_sf_returns_float_for_scalar_input():
    result = weibull_sf(x=100.0, beta=2.0, eta=100.0)
    assert isinstance(result, float)


def test_weibull_sf_returns_ndarray_for_array_input():
    result = weibull_sf(x=np.array([100.0, 110.0]), beta=2.0, eta=100.0)
    assert isinstance(result, np.ndarray)


# ------------------------------------------------------------------------------------------------------


# --- Test cases for weibull_hf ---

# Hazard function h(x) = (beta/eta) * 
# ((x-gamma)/eta)**(beta-1) for x >= gamma
# h(x) = 0 for x < gamma


# 1. Basic 2-parameter (gamma=0)
def test_weibull_hf_basic_2_parameter():
    beta = 2.0  # IFR (Increasing Failure Rate)
    eta = 100.0
    x = 100.0
    # h(100) = (2/100) * (100/100)^(2-1) 
    # = 0.02 * 1^1 = 0.02
    expected_hf = (beta / eta) * ((x / eta) ** (beta - 1.0))
    assert weibull_hf(x, beta, eta) == pytest.approx(expected_hf)


def test_weibull_hf_basic_2_parameter_beta_eq_1():
    beta = 1.0  # CFR (Constant Failure Rate) - Exponential
    eta = 100.0  # MTTF = 100, failure rate = 1/100 = 0.01
    x = 50.0
    # h(x) = beta/eta = 1/eta
    expected_hf = beta / eta
    assert weibull_hf(x, beta, eta) == pytest.approx(expected_hf)
    x_arr = np.array([10.0, 100.0, 200.0])
    results = weibull_hf(x_arr, beta, eta)
    assert np.allclose(results, expected_hf)  # Should be constant


def test_weibull_hf_basic_2_parameter_beta_lt_1():
    beta = 0.5  # DFR (Decreasing Failure Rate)
    eta = 100.0
    x = 25.0  # (x/eta) = 0.25, beta-1 = -0.5
    # h(25) = (0.5/100) * (25/100)^(-0.5) = 
    # 0.005 * (0.25)^(-0.5) = 0.005 * (1/sqrt(0.25)) 
    # = 0.005 * (1/0.5) = 0.005 * 2 = 0.01
    expected_hf = (beta / eta) * ((x / eta) ** (beta - 1.0))
    assert weibull_hf(x, beta, eta) == pytest.approx(expected_hf)


# 2. Test HF with an array of x values (2-parameter)
def test_weibull_hf_array_x_2_parameter():
    beta = 2.0
    eta = 100.0
    x_arr = np.array([50.0, 100.0, 150.0])
    # For gamma=0, h(x) = (beta/eta) * (x/eta)**(beta-1)
    expected_hfs = (beta / eta) * ((x_arr / eta) ** (beta - 1.0))

    result = weibull_hf(x_arr, beta, eta)
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected_hfs)


# 3. Test HF with location parameter gamma (3-parameter Weibull)
def test_weibull_hf_3_parameter_various_cases():
    eta = 100.0
    gamma = 50.0

    # Case 1: x < gamma -> HF = 0
    assert weibull_hf(x=gamma - 10, beta=2.0, eta=eta, gamma=gamma) == pytest.approx(
        0.0
    )

    # Case 2: x = gamma
    #   If beta > 1, HF = 0
    assert weibull_hf(x=gamma, beta=2.0, eta=eta, gamma=gamma) == pytest.approx(0.0)
    #   If beta = 1, HF = 1/eta
    assert weibull_hf(x=gamma, beta=1.0, eta=eta, gamma=gamma) == pytest.approx(
        1.0 / eta
    )
    #   If beta < 1, HF = inf
    assert weibull_hf(x=gamma, beta=0.5, eta=eta, gamma=gamma) == np.inf

    # Case 3: x > gamma
    beta_val = 1.5
    x_val = gamma + 25.0  # shifted_x = 25
    expected_hf = (beta_val / eta) * (((x_val - gamma) / eta) ** (beta_val - 1.0))
    assert weibull_hf(x_val, beta_val, eta, gamma) == pytest.approx(expected_hf)


# 4. Test HF relationship with PDF and SF: h(x) = f(x) / S(x)
#    This test can be tricky due to 
#    potential division by zero if SF is very small.
#    It's good to test where SF is reasonably > 0.
def test_weibull_hf_relation_to_pdf_sf():
    beta = 2.0
    eta = 100.0
    gamma = 0.0  # 2-parameter for simplicity here
    x_values = np.array([50.0, 100.0, 150.0])  # SF will be > 0 for these

    for x_val in x_values:
        pdf_val = weibull_pdf(x_val, beta, eta, gamma)
        sf_val = weibull_sf(x_val, beta, eta, gamma)
        hf_val_from_func = weibull_hf(x_val, beta, eta, gamma)

        if sf_val > 1e-9:  # Avoid division by zero if SF is practically zero
            expected_hf_from_definition = pdf_val / sf_val
            assert hf_val_from_func == pytest.approx(
                expected_hf_from_definition, rel=1e-5
            )
        else:  # If SF is too small, check if HF is very large or if PDF is also ~0
            if pdf_val < 1e-9:  
    # Both PDF and SF are near zero
                pass  
    # HF could be NaN or some limit, tricky to 
    #assert generic equality. Our direct formula should be more robust.
            else:  
    # PDF non-zero, SF zero -> HF should be infinite
                assert hf_val_from_func == np.inf


# --- Input Validation Tests ---
def test_weibull_hf_invalid_beta():
    # Corrected match string
    with pytest.raises(
        ValueError, match="Shape parameter beta \\(β\\) must be a positive number."
    ):
        weibull_hf(x=100, beta=0, eta=100)
    # You can also add the second test case back if you like,
    #  ensuring it uses the same corrected match string
    with pytest.raises(
        ValueError, match="Shape parameter beta \\(β\\) must be a positive number."
    ):
        weibull_hf(x=100, beta=-1, eta=100)


def test_weibull_hf_invalid_eta():
    # Corrected match string
    with pytest.raises(
        ValueError, match="Scale parameter eta \\(η\\) must be a positive number."
    ):
        weibull_hf(x=100, beta=2, eta=0)
    # You can also add the second test case back if you like
    with pytest.raises(
        ValueError, match="Scale parameter eta \\(η\\) must be a positive number."
    ):
        weibull_hf(x=100, beta=2, eta=-10)


# --- Return Type Tests ---
def test_weibull_hf_returns_float_for_scalar_input():
    result = weibull_hf(x=100.0, beta=2.0, eta=100.0)
    assert isinstance(result, float)


def test_weibull_hf_returns_ndarray_for_array_input():
    result = weibull_hf(x=np.array([100.0, 110.0]), beta=2.0, eta=100.0)
    assert isinstance(result, np.ndarray)


# -----------------------------------------------------------------------------------------------------


# --- Test cases for weibull_fit ---


def test_weibull_fit_2_parameter_basic():
    # Known parameters for data generation
    true_beta = 2.0
    true_eta = 100.0
    true_gamma = 0.0  # For 2-parameter fit
    sample_size = 500  # A reasonable sample size for fitting

    # Generate random data from weibull_min with these parameters
    # SciPy parameters: c=beta, loc=gamma, scale=eta
    np.random.seed(42)  # For reproducibility of test data
    failure_data = stats.weibull_min.rvs(
        c=true_beta, loc=true_gamma, scale=true_eta, size=sample_size
    )

    # Fit the data using our function (2-parameter fit by default)
    est_beta, est_eta, est_gamma = weibull_fit(failure_data, fit_gamma=False)

    # Check if estimated parameters are
    #  reasonably close to true parameters
    # For MLE, estimates get closer with
    #  larger sample sizes.
    # Relative tolerance might be appropriate here.
    assert est_beta == pytest.approx(
        true_beta, rel=0.2
    )  # Allow up to 20% relative error for beta
    assert est_eta == pytest.approx(
        true_eta, rel=0.2
    )  # Allow up to 20% relative error for eta
    assert est_gamma == pytest.approx(
        true_gamma, abs=1e-3
    )  # Gamma should be very close to 0

    print(
        f"\n2P Fit: True (β,η,γ)=({true_beta},{true_eta},{true_gamma}), Estimated=({est_beta:.2f},{est_eta:.2f},{est_gamma:.2f})"
    )


def test_weibull_fit_3_parameter_basic():
    # Known parameters for data generation
    true_beta = 1.5
    true_eta = 200.0
    true_gamma = 50.0  # Non-zero location
    sample_size = 1000  # Fitting 3 params usually needs more data

    np.random.seed(43)  # Different seed for different data
    failure_data = stats.weibull_min.rvs(
        c=true_beta, loc=true_gamma, scale=true_eta, size=sample_size
    )
    # Ensure all generated data is > 0 if we're 
    # not expecting negative gamma.
    # Or, if data can be <= gamma, fit function
    #  should handle it (SciPy's fit can sometimes).
    # If failure_data can be very close to gamma, fitting gamma can be sensitive.

    # Fit the data using our function, attempting to fit gamma
    est_beta, est_eta, est_gamma = weibull_fit(failure_data, fit_gamma=True)

    # Tolerances might need to be larger for 
    # 3-parameter fits, especially for gamma.
    assert est_beta == pytest.approx(true_beta, rel=0.25)
    assert est_eta == pytest.approx(true_eta, rel=0.25)
    assert est_gamma == pytest.approx(
        true_gamma, rel=0.25, abs=5.0
    )  # Gamma can be harder to pinpoint

    print(
        f"\n3P Fit: True (β,η,γ)=({true_beta},{true_eta},{true_gamma}), Estimated=({est_beta:.2f},{est_eta:.2f},{est_gamma:.2f})"
    )


# --- Input Validation Tests for weibull_fit ---
# THIS IS THE NEW, CORRECTED VERSION TO USE
def test_weibull_fit_insufficient_data():
    # Test with an empty list
    with pytest.raises(ValueError, match="At least two data points are required to fit a distribution."):
        weibull_fit([])
        
    # Test with a single data point
    with pytest.raises(ValueError, match="At least two data points are required to fit a distribution."):
        weibull_fit([100.0])


def test_weibull_fit_non_positive_data_2_parameter():
    with pytest.raises(
        ValueError, match="strictly positive for a 2-parameter Weibull fit"
    ):
        weibull_fit([10, 0, 30], fit_gamma=False)
    with pytest.raises(
        ValueError, match="strictly positive for a 2-parameter Weibull fit"
    ):
        weibull_fit([10, -5, 30], fit_gamma=False)


def test_weibull_fit_nan_or_inf_data():
    with pytest.raises(
        ValueError, match="Input 'failure_times' must not contain NaN or Inf values."
    ):
        weibull_fit([10, np.nan, 30])
    with pytest.raises(
        ValueError, match="Input 'failure_times' must not contain NaN or Inf values."
    ):
        weibull_fit([10, np.inf, 30])


def test_weibull_fit_wrong_input_type():
    with pytest.raises(
        TypeError, match="Input 'failure_times' must be a list or NumPy array."
    ):
        weibull_fit("not data")  # type: ignore


def test_weibull_fit_wrong_dimension():
    with pytest.raises(
        ValueError, match="'failure_times' must be a 1-dimensional array or list."
    ):
        weibull_fit(np.array([[1, 2], [3, 4]]))
