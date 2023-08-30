//! An N-dimension tensor.

pub mod error;
pub mod layout;

use std::{ops::Index, rc::Rc};

use rand::Rng;
use rand_distr::Distribution;

use self::{
    error::TensorError,
    layout::{PositionIterator, TensorLayout},
};

/// An N-dimension array holding elements row-major order. Tensors are immutable and new ones are
/// created each time we perform an operation. Tensors' underlying data is shared using reference
/// counting and only cloned when an operations can't be performed without modifying the data.
#[derive(Debug)]
pub struct Tensor {
    data: Rc<Vec<f32>>,
    layout: TensorLayout,
}

impl From<Vec<f32>> for Tensor {
    fn from(data: Vec<f32>) -> Self {
        let data_len = data.len();
        Self {
            data: Rc::new(data),
            layout: TensorLayout::from(&[data_len]),
        }
    }
}

impl<const N: usize> From<[f32; N]> for Tensor {
    fn from(data: [f32; N]) -> Self {
        Tensor::from(data.to_vec())
    }
}

impl<const N: usize> From<&[f32; N]> for Tensor {
    fn from(data: &[f32; N]) -> Self {
        Tensor::from(data.to_vec())
    }
}

impl From<&[f32]> for Tensor {
    fn from(data: &[f32]) -> Self {
        Tensor::from(data.to_vec())
    }
}

impl Index<usize> for &Tensor {
    type Output = f32;

    fn index(&self, pos: usize) -> &Self::Output {
        &self.data[pos]
    }
}

impl Index<&[usize]> for &Tensor {
    type Output = f32;

    fn index(&self, index: &[usize]) -> &Self::Output {
        let pos = self.layout.index_to_position(index);
        &self[pos]
    }
}

impl Index<Vec<usize>> for &Tensor {
    type Output = f32;

    fn index(&self, index: Vec<usize>) -> &Self::Output {
        &self[index.as_slice()]
    }
}

impl<const N: usize> Index<[usize; N]> for &Tensor {
    type Output = f32;

    fn index(&self, index: [usize; N]) -> &Self::Output {
        &self[&index]
    }
}

impl<const N: usize> Index<&[usize; N]> for &Tensor {
    type Output = f32;

    fn index(&self, index: &[usize; N]) -> &Self::Output {
        &self[index.as_slice()]
    }
}

impl<'a> IntoIterator for &'a Tensor {
    type Item = &'a f32;

    type IntoIter = TensorRowIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        Self::IntoIter {
            tensor: self,
            position_iterator: self.layout.iter_position(),
        }
    }
}

impl Tensor {
    /// Creates a new tensor using the given data and layout.
    pub fn new(data: &[f32], shape: &[usize]) -> Result<Self, TensorError> {
        let layout = TensorLayout::from(shape);
        if layout.elems() != data.len() {
            return Err(TensorError::IncompatibleShapes(
                shape.to_vec(),
                vec![data.len()],
            ));
        }
        Ok(Self {
            data: Rc::new(data.to_vec()),
            layout,
        })
    }

    /// Creates a new tensor with randomized data.
    pub fn rand<R, D>(rng: R, distribution: D, shape: &[usize]) -> Self
    where
        R: Rng,
        D: Distribution<f32>,
    {
        let layout = TensorLayout::from(shape);
        let data = rng.sample_iter(distribution).take(layout.elems()).collect();
        Self {
            data: Rc::new(data),
            layout,
        }
    }

    /// Returns the layout of this tensor.
    pub fn layout(&self) -> &TensorLayout {
        &self.layout
    }

    /// Applies the unary function `op` to all elements in the tensor.
    pub fn map<F>(&self, op: F) -> Self
    where
        F: Fn(&f32) -> f32,
    {
        let mut res = Vec::with_capacity(self.layout.elems());
        for x in self.into_iter() {
            res.push(op(x));
        }
        Self {
            data: Rc::new(res),
            layout: TensorLayout::from(self.layout.shape()),
        }
    }

    /// Applies the binary function `op` by pairing each element in `self` and `other` and applying
    /// broadcast when necessary. See [NumPy's broadcasting] for more information.
    ///
    /// [NumPy's broadcasting]: https://numpy.org/doc/stable/user/basics.broadcasting.html
    pub fn zip<F>(&self, other: &Self, op: F) -> Result<Self, TensorError>
    where
        F: Fn(&f32, &f32) -> f32,
    {
        let (lhs, rhs) = self.broadcast(other)?;
        let mut res = Vec::with_capacity(lhs.layout.elems());
        for (x, y) in lhs.into_iter().zip(rhs.into_iter()) {
            res.push(op(x, y));
        }
        Ok(Self {
            data: Rc::new(res),
            layout: TensorLayout::from(lhs.layout.shape()),
        })
    }

    /// Reduces all elements along the given axis into a single element using the given operation.
    /// This effectively reduces the rank of the tensor by one. See [NumPy's reduce] for more
    /// information.
    ///
    /// [NumPy's reduce]: https://numpy.org/doc/stable/reference/generated/numpy.ufunc.reduce.html#numpy-ufunc-reduce
    pub fn reduce<F>(&self, axis: &[usize], default: f32, op: F) -> Result<Self, TensorError>
    where
        F: Fn(&f32, &f32) -> f32,
    {
        let (layout, reducer) = self.layout.reduce(axis)?;
        let mut res = vec![default; layout.elems()];
        for idx in self.layout.iter_index() {
            let src_pos = self.layout.index_to_position(&idx);
            let dst_pos = reducer.index_to_position(&idx);
            res[dst_pos] = op(&res[dst_pos], &self.data[src_pos]);
        }
        Ok(Self {
            data: Rc::new(res),
            layout,
        })
    }

    /// Removes all singleton dimensions from the tensor.
    pub fn squeeze(&self) -> Self {
        let layout = self.layout.squeeze();
        Self {
            data: self.data.clone(),
            layout,
        }
    }

    /// Swaps 2 axis of the tensor without cloning its data.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, TensorError> {
        let layout = self.layout.transpose(dim0, dim1)?;
        Ok(Self {
            data: self.data.clone(),
            layout,
        })
    }

    /// Permutes the tensor axis according to the given axis ordering without cloning its data.
    pub fn permute(&self, permutation: &[usize]) -> Result<Self, TensorError> {
        let layout = self.layout.permute(permutation)?;
        Ok(Self {
            data: self.data.clone(),
            layout,
        })
    }

    /// Reshapes the tensor to the given shape. This might clone the data if the new shape can't be
    /// represented contiguously basing on the current layout.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError> {
        match self.layout.reshape(shape)? {
            Some(layout) => Ok(Self {
                data: self.data.clone(),
                layout,
            }),
            None => Self::from(self.data.as_ref().clone()).reshape(shape),
        }
    }

    /// Broadcast the tensors and returns their broadcasted versions. See [TensorLayout::broadcast]
    /// for more details.
    fn broadcast(&self, other: &Self) -> Result<(Self, Self), TensorError> {
        let (lhs_layout, rhs_layout) = self.layout.broadcast(&other.layout)?;
        let lhs = Self {
            data: self.data.clone(),
            layout: lhs_layout,
        };
        let rhs = Self {
            data: other.data.clone(),
            layout: rhs_layout,
        };
        Ok((lhs, rhs))
    }
}

/// A row-major iterator over a tensor.
pub struct TensorRowIter<'a> {
    tensor: &'a Tensor,
    position_iterator: PositionIterator<'a>,
}

impl<'a> Iterator for TensorRowIter<'a> {
    type Item = &'a f32;

    fn next(&mut self) -> Option<Self::Item> {
        self.position_iterator
            .next()
            .map(|pos| &self.tensor.data[pos])
    }
}
