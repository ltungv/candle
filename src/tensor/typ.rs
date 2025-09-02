//! Traits for types that can be used as elements in a tensor.

/// Tensor element values.
pub trait Elem: 'static + Clone {}

impl Elem for bool {}
impl Elem for f32 {}
impl Elem for i32 {}

/// Boolean-like values
pub trait Bool: 'static + Elem + PartialEq + PartialOrd + From<bool> {
    /// Return the minimum value of the type.
    fn min_value() -> Self;
}

impl Bool for bool {
    fn min_value() -> Self {
        false
    }
}

impl Bool for f32 {
    fn min_value() -> Self {
        Self::MIN
    }
}

impl Bool for i32 {
    fn min_value() -> Self {
        Self::MIN
    }
}

/// Numeric values
pub trait Num: 'static + Bool + num::Num {}

impl Num for f32 {}
impl Num for i32 {}

/// Floating values.
pub trait Float: 'static + Num + num::Float {}

impl Float for f32 {}
