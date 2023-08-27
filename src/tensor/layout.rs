//! Defines tensor layout.

use std::iter;

use super::error::TensorError;

/// A description of how to translate between a contiguous memory array and an N-dimention array.
#[derive(Debug, PartialEq)]
pub struct TensorLayout {
    /// The number of elements in each dimension.
    shape: Vec<usize>,
    /// The number of elements in the memory array that need to be skipped to move to the next
    /// element in each dimension.
    strides: Vec<usize>,
}

impl From<Vec<usize>> for TensorLayout {
    fn from(shape: Vec<usize>) -> Self {
        let mut strides = vec![1; shape.len()];
        for (i, s) in shape.iter().skip(1).enumerate().rev() {
            strides[i] = strides[i + 1] * s;
        }
        Self { shape, strides }
    }
}

impl From<&[usize]> for TensorLayout {
    fn from(shape: &[usize]) -> Self {
        TensorLayout::from(shape.to_vec())
    }
}

impl<const N: usize> From<&[usize; N]> for TensorLayout {
    fn from(shape: &[usize; N]) -> Self {
        TensorLayout::from(shape.to_vec())
    }
}

impl TensorLayout {
    /// Returns the shape of a tensor.
    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    /// Returns the strides of a tensor.
    pub fn strides(&self) -> &[usize] {
        self.strides.as_slice()
    }

    /// Returns the number of elements in a tensor.
    pub fn elems(&self) -> usize {
        self.shape.iter().product()
    }

    /// Returns a new layout where all singleton dimensions are removed.
    pub fn squeeze(&self) -> Self {
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        for (size, stride) in self.shape.iter().zip(self.strides.iter()) {
            if *size != 1 {
                shape.push(*size);
                strides.push(*stride);
            }
        }
        Self { shape, strides }
    }

    /// Returns a new layout where the dimensions are transposed.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, TensorError> {
        if dim0 >= self.shape.len() {
            return Err(TensorError::UnknownAxis(dim0));
        }
        if dim1 >= self.shape.len() {
            return Err(TensorError::UnknownAxis(dim1));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Ok(Self { shape, strides })
    }

    /// Returns a new layout where the dimensions are permuted.
    pub fn permute(&self, permutation: &[usize]) -> Result<Self, TensorError> {
        for axis in permutation {
            if *axis >= self.shape.len() {
                return Err(TensorError::UnknownAxis(*axis));
            }
        }
        let elems = permutation.len();
        if elems * (elems - 1) / 2 != permutation.iter().sum() {
            return Err(TensorError::InvalidArgument(
                "each axis must be specified exactly once".to_string(),
            ));
        }
        let mut shape = Vec::with_capacity(self.shape.len());
        let mut strides = Vec::with_capacity(self.strides.len());
        for i in permutation {
            shape.push(self.shape[*i]);
            strides.push(self.strides[*i]);
        }
        Ok(Self { shape, strides })
    }

    /// Returns a new layout for a tensor with singleton dimensions expanded to a larger size.
    /// Tensor can be also expanded to a larger number of dimensions, and the new ones will be
    /// appended at the front. For the new dimensions, the size cannot be set to -1. Expanding
    /// a tensor does not allocate new memory, but only creates a new view on the existing
    /// tensor where a dimension of size one is expanded to a larger size by setting the stride
    /// to 0. Any dimension of size 1 can be expanded to an arbitrary value without allocating
    /// new memory.
    pub fn expand(&self, shape: &[usize]) -> Result<Self, TensorError> {
        let mut new_shape = shape.to_vec();
        let mut new_strides = vec![0; shape.len()];
        for (old_dim, new_dim) in (0..self.shape.len()).rev().zip((0..shape.len()).rev()) {
            let old_sz = self.shape[old_dim];
            let new_sz = shape[new_dim];
            if old_sz == new_sz {
                new_shape[new_dim] = old_sz;
                new_strides[new_dim] = self.strides[old_dim];
            } else if old_sz == 1 {
                new_shape[new_dim] = new_sz;
                new_strides[new_dim] = 0;
            } else {
                return Err(TensorError::IncompatibleShapes(
                    self.shape.to_vec(),
                    shape.to_vec(),
                ));
            }
        }
        Ok(Self {
            shape: new_shape,
            strides: new_strides,
        })
    }

    /// Returns a new layout for a tensor having the same number of elements
    /// but with a different shape. This function returns an error if the new
    /// layout can't be accomodated without copying data.
    pub fn reshape(&self, shape: &[usize]) -> Result<Self, TensorError> {
        if self.elems() != shape.iter().product() {
            return Err(TensorError::IncompatibleShapes(
                self.shape.to_vec(),
                shape.to_vec(),
            ));
        }
        // 1. Check the new shape for joins/splits of the dimensions.
        // 2. Ensure the joined/splitted dimensions are contiguous.
        todo!()
    }

    /// Translates a tensor index into a position in the data buffer.
    pub fn index_to_position(&self, index: &[usize]) -> usize {
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(x, s)| x * s)
            .sum()
    }

    /// Translates a position in the data buffer into a tensor index.
    pub fn position_to_index(&self, position: usize) -> Vec<usize> {
        let mut index = Vec::with_capacity(self.shape.len());
        let mut remainder = position;
        for s in &self.strides {
            index.push(remainder / s);
            remainder %= s;
        }
        index
    }

    /// Returns an iterator over the indices of a tensor.
    pub fn iter_index(&self) -> IndexIterator<'_> {
        IndexIterator::from(self)
    }

    /// Returns an iterator over the internal buffer positions of a tensor.
    pub fn iter_position(&self) -> PositionIterator<'_> {
        PositionIterator::from(self)
    }
}

/// Returns the shape of a broadcast between this layout and another shape.
pub fn broadcast_shape(lhs: &[usize], rhs: &[usize]) -> Result<Vec<usize>, TensorError> {
    let (small, large) = if lhs.len() < rhs.len() {
        (lhs, rhs)
    } else {
        (rhs, lhs)
    };
    let mut new_shape = Vec::with_capacity(large.len());
    for sizes in small
        .iter()
        .rev()
        .chain(iter::once(&1usize).cycle())
        .zip(large.iter().rev())
    {
        match sizes {
            (1, d) => new_shape.push(*d),
            (d, 1) => new_shape.push(*d),
            (dx, dy) if dx == dy => new_shape.push(*dx),
            _ => return Err(TensorError::IncompatibleShapes(lhs.to_vec(), rhs.to_vec())),
        }
    }
    new_shape.reverse();
    Ok(new_shape)
}

/// An iterator over a tensor's indices.
pub struct IndexIterator<'a> {
    layout: &'a TensorLayout,
    index: Vec<usize>,
    exhausted: bool,
}

impl<'a> Iterator for IndexIterator<'a> {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.exhausted {
            return None;
        }
        let index = self.index.clone();
        for (i, s) in self.layout.shape.iter().enumerate().rev() {
            self.index[i] += 1;
            if self.index[i] < *s {
                break;
            }
            self.index[i] = 0;
        }
        self.exhausted = self.index.iter().all(|e| *e == 0);
        Some(index)
    }
}

impl<'a> From<&'a TensorLayout> for IndexIterator<'a> {
    fn from(layout: &'a TensorLayout) -> Self {
        Self {
            layout,
            index: vec![0; layout.shape.len()],
            exhausted: false,
        }
    }
}

/// An iterator over a tensor's internal buffer positions.
pub struct PositionIterator<'a> {
    layout: &'a TensorLayout,
    index_iterator: IndexIterator<'a>,
}

impl<'a> Iterator for PositionIterator<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        self.index_iterator
            .next()
            .map(|index| self.layout.index_to_position(&index))
    }
}

impl<'a> From<&'a TensorLayout> for PositionIterator<'a> {
    fn from(layout: &'a TensorLayout) -> Self {
        Self {
            layout,
            index_iterator: IndexIterator::from(layout),
        }
    }
}
