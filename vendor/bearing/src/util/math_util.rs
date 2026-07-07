// SPDX-License-Identifier: Apache-2.0

/// Returns a relative error bound for a sum of `num_values` positive doubles,
/// computed using recursive summation, ie. sum = x1 + ... + xn.
///
/// NOTE: This only works if all values are POSITIVE so that Σ |xi| == |Σ xi|.
/// Uses formula 3.5 from Higham, Nicholas J. (1993), "The accuracy of floating
/// point summation", SIAM Journal on Scientific Computing.
pub fn sum_relative_error_bound(num_values: i32) -> f64 {
    if num_values <= 1 {
        return 0.0;
    }
    // u = unit roundoff in the paper, also called machine precision or machine epsilon
    let u = f64::from_bits(((1023 - 52) as u64) << 52); // Math.scalb(1.0, -52)
    (num_values - 1) as f64 * u
}

/// Returns the maximum possible sum across `num_values` non-negative doubles,
/// assuming one sum yielded `sum`.
pub fn sum_upper_bound(sum: f64, num_values: i32) -> f64 {
    if num_values <= 2 {
        // When there are only two clauses, the sum is always the same regardless
        // of the order.
        return sum;
    }

    // The error of sums depends on the order in which values are summed up. In
    // order to avoid this issue, we compute an upper bound of the value that
    // the sum may take. If the max relative error is b, then it means that two
    // sums are always within 2*b of each other.
    // For conjunctions, we could skip this error factor since the order in which
    // scores are summed up is predictable, but in practice, this wouldn't help
    // much since the delta that is introduced by this error factor is usually
    // cancelled by the float cast.
    let b = sum_relative_error_bound(num_values);
    (1.0 + 2.0 * b) * sum
}

/// Returns the min of the two given integers, treating them as unsigned.
pub fn unsigned_min(a: i32, b: i32) -> i32 {
    if (a as u32) < (b as u32) { a } else { b }
}

#[cfg(test)]
mod tests {
    use super::*;
    use assertables::*;

    #[test]
    fn test_sum_relative_error_bound_zero_values() {
        assert_eq!(sum_relative_error_bound(0), 0.0);
    }

    #[test]
    fn test_sum_relative_error_bound_one_value() {
        assert_eq!(sum_relative_error_bound(1), 0.0);
    }

    #[test]
    fn test_sum_relative_error_bound_two_values() {
        let bound = sum_relative_error_bound(2);
        assert_gt!(bound, 0.0);
        // 1 * 2^-52
        assert_in_delta!(bound, 2.220446049250313e-16, 1e-30);
    }

    #[test]
    fn test_sum_relative_error_bound_ten_values() {
        let bound = sum_relative_error_bound(10);
        // 9 * 2^-52
        assert_in_delta!(bound, 9.0 * 2.220446049250313e-16, 1e-29);
    }

    #[test]
    fn test_sum_upper_bound_two_values() {
        // Two values: returns sum unchanged
        assert_eq!(sum_upper_bound(10.0, 2), 10.0);
    }

    #[test]
    fn test_sum_upper_bound_one_value() {
        assert_eq!(sum_upper_bound(5.0, 1), 5.0);
    }

    #[test]
    fn test_sum_upper_bound_three_values() {
        let result = sum_upper_bound(10.0, 3);
        assert_gt!(result, 10.0);
        // Should be very close to 10.0 (tiny error bound)
        assert_in_delta!(result, 10.0, 1e-13);
    }

    #[test]
    fn test_unsigned_min_both_positive() {
        assert_eq!(unsigned_min(3, 5), 3);
        assert_eq!(unsigned_min(5, 3), 3);
    }

    #[test]
    fn test_unsigned_min_negative_is_larger() {
        // -1 as unsigned is u32::MAX, so positive value should be min
        assert_eq!(unsigned_min(-1, 5), 5);
        assert_eq!(unsigned_min(5, -1), 5);
    }

    #[test]
    fn test_unsigned_min_both_negative() {
        // -1 (0xFFFFFFFF) vs -2 (0xFFFFFFFE): -2 is smaller unsigned
        assert_eq!(unsigned_min(-1, -2), -2);
    }
}
