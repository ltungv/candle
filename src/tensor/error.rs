//! Defines tensor errors.

use std::{error, fmt};

/// An error type for all operations on tensors.
#[derive(Clone, Debug)]
pub enum TensorError {
    /// An operation was performed on 2 objects with incompatible shapes.
    IncompatibleShapes(Vec<usize>, Vec<usize>),
    /// An operation was performed with an axis that does not exist.
    UnknownAxis(usize),
    /// A custom error message.
    Custom(String),
}

impl error::Error for TensorError {}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IncompatibleShapes(l, r) => write!(f, "Incompatible shapes {:?} and {:?}.", l, r),
            Self::UnknownAxis(d) => write!(f, "Unknown axis {}.", d),
            Self::Custom(s) => write!(f, "{}", s),
        }
    }
}

impl From<String> for TensorError {
    fn from(err: String) -> Self {
        Self::Custom(err)
    }
}
