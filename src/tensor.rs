//! An N-dimension tensor.

use std::{error, fmt, iter};

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

#[derive(Debug)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Shape,
    strides: Strides,
}

impl From<&[f32]> for Tensor {
    fn from(data: &[f32]) -> Self {
        Self {
            data: Vec::from(data),
            shape: Shape::from([data.len()]),
            strides: Strides::from([1]),
        }
    }
}

impl<const N: usize> From<[f32; N]> for Tensor {
    fn from(data: [f32; N]) -> Self {
        Self {
            data: Vec::from(data),
            shape: Shape::from([data.len()]),
            strides: Strides::from([1]),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Shape(pub Vec<usize>);

impl From<&[usize]> for Shape {
    fn from(data: &[usize]) -> Self {
        Self(Vec::from(data))
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(data: [usize; N]) -> Self {
        Self(Vec::from(data))
    }
}

impl Shape {
    fn broadcast(&self, other: &Self) -> Result<Self, TensorError> {
        let (short, long) = if self.0.len() < other.0.len() {
            (self, other)
        } else {
            (other, self)
        };
        let mut data = Vec::with_capacity(long.0.len());
        for dims in short
            .0
            .iter()
            .rev()
            .chain(iter::once(&1usize).cycle())
            .zip(long.0.iter().rev())
        {
            match dims {
                (1, d) => data.push(*d),
                (d, 1) => data.push(*d),
                (dx, dy) if dx == dy => data.push(*dx),
                _ => return Err(TensorError::ShapeMismatch(short.0.clone(), long.0.clone())),
            }
        }
        data.reverse();
        Ok(Shape(data))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Strides(pub Vec<usize>);

impl From<&[usize]> for Strides {
    fn from(data: &[usize]) -> Self {
        Self(Vec::from(data))
    }
}

impl<const N: usize> From<[usize; N]> for Strides {
    fn from(data: [usize; N]) -> Self {
        Self(Vec::from(data))
    }
}

impl From<&Shape> for Strides {
    fn from(value: &Shape) -> Self {
        let mut strides = Vec::with_capacity(value.0.len());
        let mut stride = 1;
        for s in value.0.iter().rev() {
            strides.push(stride);
            stride *= s;
        }
        strides.reverse();
        Strides(strides)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Index(pub Vec<usize>);

impl From<&[usize]> for Index {
    fn from(data: &[usize]) -> Self {
        Self(Vec::from(data))
    }
}

impl<const N: usize> From<[usize; N]> for Index {
    fn from(data: [usize; N]) -> Self {
        Self(Vec::from(data))
    }
}

impl Index {
    pub fn to_pos(&self, stride: &Strides) -> usize {
        self.0.iter().zip(stride.0.iter()).map(|(x, s)| x * s).sum()
    }

    pub fn from_pos(pos: usize, shape: &Shape) -> Index {
        let mut index = Index(Vec::with_capacity(shape.0.len()));
        let mut remainder = pos;
        for s in Strides::from(shape).0 {
            index.0.push(remainder / s);
            remainder %= s;
        }
        index
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::{Index, Shape, Strides};

    use super::Tensor;

    #[test]
    fn test_tensor_from_array() {
        let t = Tensor::from([1.0f32, 2.0f32, 3.0f32]);
        let expected_len = t
            .shape
            .0
            .iter()
            .zip(t.strides.0.iter())
            .map(|(s, t)| s * t)
            .sum();
        assert_eq!(t.data.len(), expected_len);
    }

    #[test]
    fn test_shapes_broadcast() {
        let x = Shape(vec![1]).broadcast(&Shape(vec![3])).unwrap();
        assert_eq!(x, Shape(vec![3]));

        let x = Shape(vec![3]).broadcast(&Shape(vec![1])).unwrap();
        assert_eq!(x, Shape(vec![3]));

        let x = Shape(vec![1]).broadcast(&Shape(vec![2, 3])).unwrap();
        assert_eq!(x, Shape(vec![2, 3]));

        let x = Shape(vec![3, 2]).broadcast(&Shape(vec![1])).unwrap();
        assert_eq!(x, Shape(vec![3, 2]));

        let x = Shape(vec![3, 1, 2, 1])
            .broadcast(&Shape(vec![1, 5]))
            .unwrap();
        assert_eq!(x, Shape(vec![3, 1, 2, 5]));

        let x = Shape(vec![2, 3, 1])
            .broadcast(&Shape(vec![7, 2, 1, 5]))
            .unwrap();
        assert_eq!(x, Shape(vec![7, 2, 3, 5]));
    }

    #[test]
    fn test_index_to_position() {
        let strides = Strides::from([2, 1]);

        let idx = Index::from([0, 0]);
        assert_eq!(idx.to_pos(&strides), 0);

        let idx = Index::from([0, 1]);
        assert_eq!(idx.to_pos(&strides), 1);

        let idx = Index::from([1, 0]);
        assert_eq!(idx.to_pos(&strides), 2);

        let idx = Index::from([1, 1]);
        assert_eq!(idx.to_pos(&strides), 3);
    }

    #[test]
    fn test_position_to_index() {
        let shape = Shape::from([2, 2, 2]);

        let x = Index::from_pos(0, &shape);
        assert_eq!(x, Index::from([0, 0, 0]));

        let x = Index::from_pos(1, &shape);
        assert_eq!(x, Index::from([0, 0, 1]));

        let x = Index::from_pos(2, &shape);
        assert_eq!(x, Index::from([0, 1, 0]));

        let x = Index::from_pos(3, &shape);
        assert_eq!(x, Index::from([0, 1, 1]));

        let x = Index::from_pos(4, &shape);
        assert_eq!(x, Index::from([1, 0, 0]));

        let x = Index::from_pos(5, &shape);
        assert_eq!(x, Index::from([1, 0, 1]));

        let x = Index::from_pos(6, &shape);
        assert_eq!(x, Index::from([1, 1, 0]));

        let x = Index::from_pos(7, &shape);
        assert_eq!(x, Index::from([1, 1, 1]));
    }
}
