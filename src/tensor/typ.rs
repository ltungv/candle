//! Traits for types that can be used as elements in a tensor.

use std::ops::{Add, Div, Mul, Sub};

/// Tensor elements.
pub trait Elem: 'static + Clone {}

impl Elem for bool {}
impl Elem for f32 {}
impl Elem for i32 {}

/// Boolean-like values
pub trait Bool: 'static + Elem + PartialEq + PartialOrd + From<bool> {
    /// The zero value of the type.
    const ZERO: Self;

    /// The one value of the type.
    const ONE: Self;

    /// The min value of the type.
    const MIN: Self;
}

impl Bool for bool {
    const ZERO: Self = false;
    const ONE: Self = true;
    const MIN: Self = false;
}

impl Bool for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
    const MIN: Self = Self::MIN;
}

impl Bool for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
    const MIN: Self = Self::MIN;
}

/// Numeric values.
pub trait Num:
    'static
    + Bool
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Mul<Self, Output = Self>
    + Div<Self, Output = Self>
{
}

impl Num for f32 {}
impl Num for i32 {}

/// Floating values.
pub trait Float: 'static + Num {
    /// Return whether `self` is `NaN`.
    #[must_use]
    fn is_nan(&self) -> bool;

    /// Return the absolute value of `self`.
    #[must_use]
    fn abs(self) -> Self;

    /// Return the exponential of `self`, i.e., `e^self`.
    #[must_use]
    fn exp(self) -> Self;

    /// Return the natural logarithm of `self`.
    #[must_use]
    fn ln(self) -> Self;

    /// Return `self` raised by `by`.
    #[must_use]
    fn pow(self, by: Self) -> Self;
}

impl Float for f32 {
    fn is_nan(&self) -> bool {
        Self::is_nan(*self)
    }

    fn abs(self) -> Self {
        Self::abs(self)
    }

    fn exp(self) -> Self {
        Self::exp(self)
    }

    fn ln(self) -> Self {
        Self::ln(self)
    }

    fn pow(self, by: Self) -> Self {
        Self::powf(self, by)
    }
}
