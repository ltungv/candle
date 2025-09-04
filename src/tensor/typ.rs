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
pub trait Float: 'static + Num + num::Float {}

impl Float for f32 {}
