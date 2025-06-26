# tests/core/test_metrics.py
import pytest
from pyreliabilitypro.core.metrics import calculate_mttf_exponential
from pyreliabilitypro.core.metrics import weibull_mttf  # Add this import
from scipy.special import gamma as gamma_function  # For test verification if needed


# ---- Test Case 1: Basic Valid Input (Happy Path) ----


def test_calculate_mttf_exponential_basic():
    failure_times = [100, 150, 200, 120, 180]
    expected_mttf = (100 + 150 + 200 + 120 + 180) / 5
    assert calculate_mttf_exponential(failure_times) == pytest.approx(expected_mttf)


# ---- Test Case 2: Single Failure Time ----


def test_calculate_mttf_exponential_single_value():
    # What: Tests if the function handles a list
    # containing just one failure time.
    # Why: This is an "edge case" – a simple input
    # that could sometimes be overlooked
    # or handled incorrectly if the logic isn't general enough.
    failure_times = [150.5]
    expected_mttf = 150.5  # The mean of a single number is the number itself.
    assert calculate_mttf_exponential(failure_times) == pytest.approx(expected_mttf)


# ---- Test Case 3: Floating Point Numbers ----
def test_calculate_mttf_exponential_float_values():
    # What: Ensures the function works correctly when
    # failure times are floats (decimal numbers).
    # Why: While our type hints `Union[int, float]`
    # allow this, it's good to explicitly test it.
    failure_times = [100.2, 150.7, 199.1]
    expected_mttf = (100.2 + 150.7 + 199.1) / 3
    assert calculate_mttf_exponential(failure_times) == pytest.approx(expected_mttf)


# ---- Test Case 4: Empty List Input (Testing for Expected Errors) ----
def test_calculate_mttf_exponential_empty_list():
    # What: This test checks if our function
    # correctly raises a `ValueError` when an
    # empty list is provided as input,
    # which is one of the error conditions we
    # defined in the function's docstring
    # and implemented with `raise ValueError(...)`.
    # Why: It's just as important to test that our
    # error handling works as it is to test
    # that the correct calculations happen
    # for valid inputs.

    with pytest.raises(ValueError, match="Input 'failure_times' cannot be empty."):
        # `with ... :`: This is a "context manager" block in Python.
        #  Pytest's `raises` is used as a context manager here.
        # `pytest.raises(ExpectedExceptionType,
        #  match="optional_regex_for_message")`:
        #  - `pytest.raises()`: This Pytest helper is
        # specifically for testing exceptions.
        #  - `ValueError`: The first argument is the type of
        #  exception we *expect* to be raised
        #  by the code inside the `with` block.
        #  - `match="Input 'failure_times' cannot be empty."`:
        #  This is an optional second argument.
        #  It's a string (or a regular expression) that
        #  Pytest will try to match against the
        #  error message of the raised exception.
        #  This is very useful because it verifies
        #  not only that the *correct type* of error was raised,
        #  but also that it was raised
        #  for the *correct reason* (as indicated by the message).

        calculate_mttf_exponential([])
        # This is the line of code that we expect to cause the `ValueError`.


# ---- Test Case 5: Input with a Negative Value ----
def test_calculate_mttf_exponential_negative_value():
    # What: Checks if a `ValueError` is raised
    # when a negative failure time is included.
    failure_times = [100, -50, 200]
    with pytest.raises(
        ValueError, match="Failure times must be non-negative. Found: -50"
    ):
        calculate_mttf_exponential(failure_times)


# ---- Test Case 6: Input that is Not a List ----
def test_calculate_mttf_exponential_not_a_list():
    # What: Checks if a `TypeError` is raised
    # when the input is not a list at all.
    failure_times = "not a list"  # type: ignore
    # We intentionally pass a string where a list is expected.
    # `# type: ignore`: This comment tells MyPy
    # (our static type checker, which we'll run later)
    # to ignore the type error on this
    # specific line. We do this because,
    # in a test, we *deliberately* want to pass
    # the wrong type to ensure
    # our function handles it by raising a `TypeError`.
    with pytest.raises(TypeError, match="Input 'failure_times' must be a list."):
        calculate_mttf_exponential(failure_times)


# ---- Test Case 7: Input List with a Non-Numeric Value ----
def test_calculate_mttf_exponential_non_numeric_in_list():
    # What: Checks if a `TypeError` is raised if
    # one of the items *inside* the list is not a number.
    failure_times = [100, "oops", 200]  # type: ignore
    with pytest.raises(
        TypeError, match="All elements in 'failure_times' must be numbers."
    ):
        calculate_mttf_exponential(failure_times)


# ---- Test Case 8: Zero Failure Times ----
def test_calculate_mttf_exponential_zero_values():
    # What: Tests if the function correctly handles
    # cases where failure times are zero.
    # Why: Zero is a valid non-negative number.
    # The MTTF of items that fail instantly is 0.
    # This is another simple edge case.
    failure_times = [0, 0, 0]
    expected_mttf = 0.0
    assert calculate_mttf_exponential(failure_times) == pytest.approx(expected_mttf)


# ------------------------------------------------------------------------------------------------------

# --- Test cases for weibull_mttf ---


# 1. Test case: Beta = 1 (should match exponential behavior: MTTF = gamma + eta)
def test_weibull_mttf_beta_equals_1():
    beta = 1.0
    eta = 100.0
    gamma_loc = 50.0
    # Expected: gamma_loc + eta * Gamma(1 + 1/1) = gamma_loc + eta * Gamma(2)
    # Gamma(2) = 1! = 1
    expected_mttf = gamma_loc + eta * 1.0
    assert weibull_mttf(beta, eta, gamma_loc) == pytest.approx(expected_mttf)

    # Test 2-parameter case (gamma_loc = 0)
    gamma_loc_0 = 0.0
    expected_mttf_0 = gamma_loc_0 + eta * 1.0
    assert weibull_mttf(beta, eta) == pytest.approx(
        expected_mttf_0
    )  # gamma_loc defaults to 0


# 2. Test case: Beta = 2
def test_weibull_mttf_beta_equals_2():
    beta = 2.0
    eta = 100.0
    gamma_loc = 0.0
    # Expected: eta * Gamma(1 + 1/2) = eta * Gamma(1.5)
    # Gamma(1.5) = Gamma(0.5 + 1) = 0.5 * Gamma(0.5) = 0.5 * sqrt(pi)
    # sqrt(pi) is approx 1.77245385
    # Gamma(1.5) approx 0.5 * 1.77245385 = 0.886226925
    expected_mttf_val = eta * gamma_function(
        1.0 + 1.0 / beta
    )  # Use scipy.special.gamma for accuracy
    assert weibull_mttf(beta, eta, gamma_loc) == pytest.approx(
        expected_mttf_val, rel=1e-6
    )


# 3. Test case: Beta < 1
def test_weibull_mttf_beta_less_than_1():
    beta = 0.8
    eta = 100.0
    gamma_loc = 20.0
    expected_mttf_val = gamma_loc + eta * gamma_function(1.0 + 1.0 / beta)
    assert weibull_mttf(beta, eta, gamma_loc) == pytest.approx(
        expected_mttf_val, rel=1e-6
    )


# 4. Test case: Beta > 2 (just another value)
def test_weibull_mttf_beta_greater_than_2():
    beta = 3.5
    eta = 100.0
    gamma_loc = 0.0
    expected_mttf_val = gamma_loc + eta * gamma_function(1.0 + 1.0 / beta)
    assert weibull_mttf(beta, eta, gamma_loc) == pytest.approx(
        expected_mttf_val, rel=1e-6
    )


# --- Input Validation Tests ---
def test_weibull_mttf_invalid_beta():
    with pytest.raises(
        ValueError, match="Shape parameter beta \\(β\\) must be greater than 0."
    ):
        weibull_mttf(beta=0, eta=100)
    with pytest.raises(
        ValueError, match="Shape parameter beta \\(β\\) must be greater than 0."
    ):
        weibull_mttf(beta=-1, eta=100)


def test_weibull_mttf_invalid_eta():
    with pytest.raises(
        ValueError, match="Scale parameter eta \\(η\\) must be greater than 0."
    ):
        weibull_mttf(beta=2, eta=0)
    with pytest.raises(
        ValueError, match="Scale parameter eta \\(η\\) must be greater than 0."
    ):
        weibull_mttf(beta=2, eta=-10)
