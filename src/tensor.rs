//! An N-dimension tensor.

pub mod error;
pub mod layout;

use std::sync::Arc;

use self::{error::TensorError, layout::Layout};

/// An N-dimension tensor.
#[derive(Debug)]
pub struct Tensor {
    data: Arc<Vec<f32>>,
    layout: Layout,
}

impl From<&[f32]> for Tensor {
    fn from(data: &[f32]) -> Self {
        Self {
            data: Arc::new(data.to_vec()),
            layout: Layout::from(&[data.len()]),
        }
    }
}

impl<const N: usize> From<[f32; N]> for Tensor {
    fn from(data: [f32; N]) -> Self {
        Self {
            data: Arc::new(data.to_vec()),
            layout: Layout::from(&[data.len()]),
        }
    }
}

impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        let data_len = data.len();
        Self {
            data: Arc::new(data),
            layout: Layout::from(&[data_len]),
        }
    }
}

impl<'a> IntoIterator for &'a Tensor {
    type Item = &'a f32;

    type IntoIter = TensorIterator<'a>;

    fn into_iter(self) -> Self::IntoIter {
        TensorIterator::from(self)
    }
}

impl Tensor {
    /// Creates a new tensor using the given data and layout.
    pub fn new(data: &[f32], layout: Layout) -> Result<Self, TensorError> {
        if layout.elems() != data.len() {
            return Err(TensorError::ShapeMismatch(
                layout.shape().to_vec(),
                vec![data.len()],
            ));
        }
        Ok(Self {
            data: Arc::new(data.to_vec()),
            layout,
        })
    }

    /// Expands the tensor to the given shape without cloning its data.
    pub fn expand(&self, shape: &[usize]) -> Result<Self, TensorError> {
        let layout = self.layout.expand(shape)?;
        Ok(Self {
            data: self.data.clone(),
            layout,
        })
    }
}

/// An iterator over a tensor.
pub struct TensorIterator<'a> {
    tensor: &'a Tensor,
    index: Vec<usize>,
    exhausted: bool,
}

impl<'a> Iterator for TensorIterator<'a> {
    type Item = &'a f32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let position = self.tensor.layout.index_to_position(&self.index);
        for (i, s) in self.tensor.layout.shape().iter().enumerate().rev() {
            self.index[i] += 1;
            if self.index[i] < *s {
                break;
            }
            self.index[i] = 0;
        }
        self.exhausted = self.index.iter().all(|e| *e == 0);
        Some(&self.tensor.data[position])
    }
}

impl<'a> From<&'a Tensor> for TensorIterator<'a> {
    fn from(tensor: &'a Tensor) -> Self {
        Self {
            tensor,
            index: vec![0; tensor.layout.shape().len()],
            exhausted: false,
        }
    }
}
