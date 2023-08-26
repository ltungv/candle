use std::{error, fmt};

#[derive(Clone, Debug)]
pub enum TensorError {
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
