//! Defines structs and functions to determine how N-dimension arrays are laid out in memory.

use std::iter;

use super::error::TensorError;

/// A description of how to translate between a contiguous memory array and an N-dimention array.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorLayout {
    /// The number of elements in each dimension.
    shape: Vec<usize>,
    /// The number of elements in the memory array that need to be skipped to move to the next
    /// element in each dimension.
    strides: Vec<usize>,
}

/// Creates a contiguous layout based on the given shape.
impl From<Vec<usize>> for TensorLayout {
    fn from(shape: Vec<usize>) -> Self {
        // Go backwards through the shape to calculate the strides. The last stride is always 1.
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

    /// Returns 2 layouts where the first is the layout of the reduced tensor and the second is the
    /// reducer layout. The reducer layout is used to map an index in the input to a memory
    /// position in the reduced tensor.
    pub fn reduce(&self, axis: &[usize]) -> Result<(Self, Self), TensorError> {
        let mut reduced_shape = self.shape.clone();
        for &d in axis {
            if d >= reduced_shape.len() {
                return Err(TensorError::UnknownAxis(d));
            }
            reduced_shape[d] = 1;
        }
        let reduced_layout = Self::from(reduced_shape);
        let mut reducer_layout = reduced_layout.clone();
        for &d in axis {
            // The reducer layout is the same as the reduced layout, except that the strides of the reduced
            // dimensions are set to 0. By setting the stride of the reduced dimension to 0, we can map
            // multiple elements within the input dimension to the same element in the reduced dimension.
            reducer_layout.strides[d] = 0;
        }
        Ok((reduced_layout, reducer_layout))
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

    /// Returns a new layout where the 2 dimensions are transposed.
    pub fn transpose(&self, axis0: usize, axis1: usize) -> Result<Self, TensorError> {
        if axis0 >= self.shape.len() {
            return Err(TensorError::UnknownAxis(axis0));
        }
        if axis1 >= self.shape.len() {
            return Err(TensorError::UnknownAxis(axis1));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(axis0, axis1);
        strides.swap(axis0, axis1);
        Ok(Self { shape, strides })
    }

    /// Returns a new layout where the dimensions are permuted.
    pub fn permute(&self, permutation: &[usize]) -> Result<Self, TensorError> {
        let mut cumsum = 0;
        let mut shape = Vec::with_capacity(self.shape.len());
        let mut strides = Vec::with_capacity(self.strides.len());
        for &d in permutation {
            if d >= self.shape.len() {
                return Err(TensorError::UnknownAxis(d));
            }
            cumsum += d;
            shape.push(self.shape[d]);
            strides.push(self.strides[d]);
        }
        let num_axis = permutation.len();
        if num_axis * (num_axis - 1) / 2 != cumsum {
            return Err(TensorError::Custom(
                "Each axis must be specified exactly once.".to_string(),
            ));
        }
        Ok(Self { shape, strides })
    }

    /// Returns a new layout for a tensor with singleton dimensions expanded to a larger size.
    ///
    /// Tensor can be also expanded to a larger number of dimensions, and the new ones will be
    /// appended at the front. For the new dimensions, the size cannot be set to -1.
    ///
    /// Expanding a tensor does not allocate new memory, but only creates a new view on the
    /// existing existing tensor where a dimension of size one is expanded to a larger size by
    /// setting the stride to 0. Any dimension of size 1 can be expanded to an arbitrary value
    /// without allocating new memory.
    pub fn expand(&self, shape: &[usize]) -> Result<Self, TensorError> {
        let new_shape = shape.to_vec();
        let mut new_strides = vec![0; shape.len()];
        let new_axis = new_shape.iter().zip(new_strides.iter_mut()).rev();
        let old_axis = self.shape.iter().zip(self.strides.iter()).rev();
        for ((new_size, new_stride), (old_size, old_stride)) in new_axis.zip(old_axis) {
            if old_size == new_size {
                *new_stride = *old_stride;
            } else if *old_size == 1 {
                *new_stride = 0;
            } else {
                return Err(TensorError::IncompatibleShapes(
                    self.shape.clone(),
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
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Option<Self>, TensorError> {
        if self.elems() != new_shape.iter().product() {
            return Err(TensorError::IncompatibleShapes(
                self.shape.to_vec(),
                new_shape.to_vec(),
            ));
        }
        let squeezed = self.squeeze();
        let old_dims = squeezed.shape.len();
        let old_shape = &squeezed.shape;
        let old_strides = &squeezed.strides;
        let new_dims = new_shape.len();
        let mut new_strides = vec![0; new_dims];
        let mut old_axis = 0;
        let mut new_axis = 0;
        while old_axis < old_dims && new_axis < new_dims {
            let old_axis_prev = old_axis;
            let new_axis_prev = new_axis;
            let mut old_size = old_shape[old_axis];
            let mut new_size = new_shape[new_axis];
            while old_size != new_size {
                if old_size < new_size {
                    old_axis += 1;
                    old_size *= old_shape[old_axis];
                } else {
                    new_axis += 1;
                    new_size *= new_shape[new_axis];
                }
            }
            if (old_axis_prev..old_axis)
                .any(|axis| old_strides[axis] != old_strides[axis + 1] * old_shape[axis + 1])
            {
                return Ok(None);
            }
            new_strides[new_axis] = old_strides[old_axis];
            for axis in (new_axis_prev + 1..=new_axis).rev() {
                new_strides[axis - 1] = new_strides[axis] * new_shape[axis];
            }
            old_axis += 1;
            new_axis += 1;
        }
        let last_stride = if new_axis > 0 {
            new_strides[new_axis - 1]
        } else {
            1
        };
        for stride in new_strides.iter_mut().skip(new_axis) {
            *stride = last_stride;
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
