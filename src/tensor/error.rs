//! Defines tensor errors.

use std::{error, fmt};

/// An error for operations on tensor(s).
#[derive(Clone, Debug)]
pub enum TensorError {
    /// An operation was performed on 2 objects with incompatible shapes.
    ShapeMismatch(Vec<usize>, Vec<usize>),
}

impl error::Error for TensorError {}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShapeMismatch(l, r) => write!(f, "Shape mismatch {:?} and {:?}", l, r),
        }
    }
}
