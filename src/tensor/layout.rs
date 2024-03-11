//! Describes how N-dimension arrays are laid out in memory.

use std::iter;

use super::error::Error;

/// A description of how to translate between a contiguous memory array and an N-dimension array.
#[derive(Clone, Debug, PartialEq)]
pub struct Layout {
    /// The number of elements in each dimension.
    shape: Vec<usize>,
    /// The number of elements in the memory array that need to be skipped to move to the next
    /// element in each dimension.
    strides: Vec<usize>,
}

/// Creates a contiguous layout based on the given shape.
impl From<Vec<usize>> for Layout {
    fn from(shape: Vec<usize>) -> Self {
        // Go backwards through the shape to calculate the strides. The last stride is always 1.
        let mut strides = vec![1; shape.len()];
        for (i, s) in shape.iter().skip(1).enumerate().rev() {
            strides[i] = strides[i + 1] * s;
        }
        Self { shape, strides }
    }
}

impl<const N: usize> From<[usize; N]> for Layout {
    fn from(shape: [usize; N]) -> Self {
        Layout::from(shape.to_vec())
    }
}

impl<const N: usize> From<&[usize; N]> for Layout {
    fn from(shape: &[usize; N]) -> Self {
        Layout::from(shape.to_vec())
    }
}

impl From<&[usize]> for Layout {
    fn from(shape: &[usize]) -> Self {
        Layout::from(shape.to_vec())
    }
}

impl Default for Layout {
    fn default() -> Self {
        Self {
            shape: Vec::new(),
            strides: Vec::new(),
        }
    }
}

impl Layout {
    /// Returns the layout for a scalar, which has no shape and strides.
    pub fn scalar() -> Self {
        Self::default()
    }

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

    /// Returns 2 layouts where the first is reduced layout and the second is the reducer layout.
    /// The reducer layout is used to map an index in the input tensor to a memory position in the
    /// reduced tensor.
    pub fn reduce(&self, dims: &[usize]) -> Result<(Self, Self), Error> {
        let mut reduced_shape = self.shape.clone();
        for d in dims.iter().copied() {
            if d >= reduced_shape.len() {
                // Reduced along an out-of-bounds dimension.
                return Err(Error::UnknownDimension(d));
            }
            reduced_shape[d] = 1;
        }
        let reduced_layout = Self::from(reduced_shape);
        let mut reducer_layout = reduced_layout.clone();
        for &d in dims {
            // The reducer layout is similar to the reduced layout, except that the strides of the reduced
            // dimensions are set to 0. By setting the stride of the reduced dimension to 0, we can map
            // multiple elements along the reduced dimension within the input tensor to the same element
            // in the reduced tensor.
            reducer_layout.strides[d] = 0;
        }
        Ok((reduced_layout, reducer_layout))
    }

    /// Returns a new layout where all singleton dimensions are removed.
    pub fn squeeze(&self) -> Self {
        let mut shape = Vec::new();
        let mut strides = Vec::new();
        let shape_iter = self.shape.iter().copied();
        let strides_iter = self.strides.iter().copied();
        for (size, stride) in shape_iter.zip(strides_iter) {
            if size != 1 {
                shape.push(size);
                strides.push(stride);
            }
        }
        Self { shape, strides }
    }

    /// Returns a new layout where the 2 dimensions are transposed.
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Self, Error> {
        if dim0 >= self.shape.len() {
            return Err(Error::UnknownDimension(dim0));
        }
        if dim1 >= self.shape.len() {
            return Err(Error::UnknownDimension(dim1));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Ok(Self { shape, strides })
    }

    /// Returns a new layout where the dimensions are permuted.
    pub fn permute(&self, permutation: &[usize]) -> Result<Self, Error> {
        let mut sum_dim = 0;
        let mut shape = Vec::with_capacity(self.shape.len());
        let mut strides = Vec::with_capacity(self.strides.len());
        for &d in permutation {
            if d >= self.shape.len() {
                return Err(Error::UnknownDimension(d));
            }
            sum_dim += d;
            shape.push(self.shape[d]);
            strides.push(self.strides[d]);
        }
        let num_dims = permutation.len();
        if num_dims * (num_dims - 1) / 2 != sum_dim {
            return Err(Error::Custom(
                "Each dimension must be specified exactly once.".to_string(),
            ));
        }
        Ok(Self { shape, strides })
    }

    /// Returns a new layout for a tensor with singleton dimensions expanded to a larger size.
    ///
    /// Tensor can also be expanded to a larger number of dimensions, and the new ones will be
    /// appended at the front. For the new dimensions, the size cannot be set to -1.
    ///
    /// Expanding a tensor does not allocate new memory, but only creates a new view on the
    /// existing tensor where a dimension of size one is expanded to a larger size by setting
    /// the stride to 0. Any dimension of size 1 can be expanded to an arbitrary value without
    /// allocating new memory.
    pub fn expand(&self, new_shape: &[usize]) -> Result<Self, Error> {
        let mut new_strides = vec![0; new_shape.len()];
        let new_shape_iter = new_shape.iter().copied();
        let new_strides_iter = new_strides.iter_mut();
        let old_shape_iter = self.shape.iter().copied();
        let old_strides_iter = self.strides.iter().copied();
        let new_dim = new_shape_iter.zip(new_strides_iter).rev();
        let old_dim = old_shape_iter.zip(old_strides_iter).rev();
        for ((new_size, new_stride), (old_size, old_stride)) in new_dim.zip(old_dim) {
            if old_size == new_size {
                *new_stride = old_stride;
            } else if old_size == 1 {
                *new_stride = 0;
            } else {
                return Err(Error::IncompatibleShapes(
                    self.shape.clone(),
                    new_shape.to_vec(),
                ));
            }
        }
        Ok(Self {
            shape: new_shape.to_vec(),
            strides: new_strides,
        })
    }

    /// Performs broadcasting on the 2 layouts and returns their broadcasted versions.
    /// See [broadcasting rule] for more details.
    ///
    /// [broadcasting rule]: https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules
    pub fn broadcast(&self, other: &Self) -> Result<(Self, Self), Error> {
        // Determine which shape has more dimensions.
        let (small, large) = if self.shape.len() < other.shape.len() {
            (self.shape(), other.shape())
        } else {
            (other.shape(), self.shape())
        };
        // Zipping the 2 shapes in reverse order while filling in 1 for the missing dimensions.
        let mut broadcasted_shape = Vec::with_capacity(large.len());
        let small_iter = small.iter().copied().rev().chain(iter::once(1).cycle());
        let large_iter = large.iter().copied().rev();
        for sizes in small_iter.zip(large_iter) {
            match sizes {
                // Broadcasted shape is the same as the larger shape
                (1, d) => broadcasted_shape.push(d),
                // Broadcasted shape is the same as the smaller shape
                (d, 1) => broadcasted_shape.push(d),
                // The dimensions are equals
                (d, dd) if d == dd => broadcasted_shape.push(d),
                _ => {
                    return Err(Error::IncompatibleShapes(
                        small.to_vec(),
                        large.to_vec(),
                    ))
                }
            }
        }
        broadcasted_shape.reverse();
        let lhs = self.expand(broadcasted_shape.as_slice())?;
        let rhs = other.expand(broadcasted_shape.as_slice())?;
        Ok((lhs, rhs))
    }

    /// Returns a new layout for a tensor having the same number of elements
    /// but with a different shape. This function returns an error if the new
    /// layout can't be accommodated without copying data.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Option<Self>, Error> {
        if self.elems() != new_shape.iter().product() {
            return Err(Error::IncompatibleShapes(
                self.shape.clone(),
                new_shape.to_vec(),
            ));
        }
        let old_layout = self.squeeze();
        let mut new_strides = vec![1; new_shape.len()];
        let mut old_dim = 0;
        let mut new_dim = 0;
        while old_dim < old_layout.shape.len() && new_dim < new_shape.len() {
            // Find the combination of dimensions from the old and new shapes that have the same
            // number of elements.
            let old_dim_prev = old_dim;
            let new_dim_prev = new_dim;
            let mut old_size = old_layout.shape[old_dim];
            let mut new_size = new_shape[new_dim];
            while old_size != new_size {
                if old_size < new_size {
                    old_dim += 1;
                    old_size *= old_layout.shape[old_dim];
                } else {
                    new_dim += 1;
                    new_size *= new_shape[new_dim];
                }
            }
            // Check if the reshaped dimensions are non-contiguous in memory.
            for (d1, d2) in (old_dim_prev..old_dim).map(|d| (d, d + 1)) {
                let expected_stride = old_layout.strides[d2] * old_layout.shape[d2];
                if old_layout.strides[d1] != expected_stride {
                    return Ok(None);
                }
            }
            // Build a strides backward.
            new_strides[new_dim] = old_layout.strides[old_dim];
            for (d1, d2) in (new_dim_prev..new_dim).map(|d| (d, d + 1)).rev() {
                new_strides[d1] = new_strides[d2] * new_shape[d2];
            }
            old_dim += 1;
            new_dim += 1;
        }
        if new_dim > 0 {
            // Fill in the remaining strides.
            let last_stride = new_strides[new_dim - 1];
            for stride in new_strides.iter_mut().skip(new_dim) {
                *stride = last_stride;
            }
        }
        Ok(Some(Self {
            shape: new_shape.to_vec(),
            strides: new_strides,
        }))
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

/// An iterator over a tensor's indices.
pub struct IndexIterator<'a> {
    layout: &'a Layout,
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

impl<'a> From<&'a Layout> for IndexIterator<'a> {
    fn from(layout: &'a Layout) -> Self {
        Self {
            layout,
            index: vec![0; layout.shape.len()],
            exhausted: false,
        }
    }
}

/// An iterator over a tensor's internal buffer positions.
pub struct PositionIterator<'a> {
    layout: &'a Layout,
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

impl<'a> From<&'a Layout> for PositionIterator<'a> {
    fn from(layout: &'a Layout) -> Self {
        Self {
            layout,
            index_iterator: IndexIterator::from(layout),
        }
    }
}
