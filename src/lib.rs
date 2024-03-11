//! Implementations of various machine learning data structures and algorithms.

#![warn(
    clippy::cargo,
    clippy::nursery,
    clippy::pedantic,
    missing_debug_implementations,
    rustdoc::all
)]
#![deny(
    clippy::all,
    missing_docs,
    rust_2018_idioms,
    rust_2021_compatibility,
    unsafe_code
)]

pub mod dataset;
pub mod scalar;
pub mod tensor;
