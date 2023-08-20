//! Implementations of various machine learning data structures and algorithms.

#![deny(unsafe_code, rust_2018_idioms, rust_2021_compatibility)]
#![warn(missing_docs)]

pub mod autodiff;
pub mod tensor;

/// A sample input and output data.
#[derive(Debug, Clone)]
pub struct Sample<const M: usize, const N: usize> {
    /// The input data.
    pub input: [f64; M],
    /// The output data.
    pub output: [f64; N],
}
