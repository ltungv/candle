//! Traits defining operations across different abstraction levels of an automatically
//! differentiable tensor.

use std::{num::NonZeroUsize, ops};

use num::traits::bounds::LowerBounded;

use crate::tensor::cpu;

/// Low-level tensor operations that must be suppported by the hardware.
///
/// Functions in this trait are considered low-level operations that must be supported by the
/// hardware on which a tensor is used. Higher-level operations are translated to these operations
/// in order to be used across all hardwares for which [`LL`] are implemented.
pub trait LL {
    /// Representation of a tensor parameterized by its element type `E`.
    type Repr<E>: Clone;

    /// Create a new tensor with the given shape and data.
    fn new<E>(shape: &[NonZeroUsize], data: &[E]) -> Option<Self::Repr<E>>
    where
        E: Clone;

    /// Return the shape of the tensor.
    fn shape<E>(t: &Self::Repr<E>) -> &[NonZeroUsize];

    /// Apply exp to each element.
    fn exp<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: num::Float;

    /// Apply the natural logarithm to each element.
    fn ln<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: num::Float;

    /// Add `rhs` to `lhs`, element-wise.
    fn add<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Add<Output = E>;

    /// Subtract `rhs` from `lhs`, element-wise.
    fn sub<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Sub<Output = E>;

    /// Multiply `lhs` by `rhs`, element-wise.
    fn mul<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Mul<Output = E>;

    /// Divide `lhs` by `rhs`, element-wise.
    fn div<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Div<Output = E>;

    /// Raise `lhs` to the power of `rhs`, element-wise.
    fn pow<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: num::Float;

    /// Compare 'lhs' with 'rhs', element-wise, returning 1s where the elements are equal.
    fn eq<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<bool>>
    where
        E: PartialEq;

    /// Reduce along the given axes by summing all elements.
    fn sum<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Clone + num::Zero + ops::Add<Output = E>;

    /// Reduce along the given axes by getting the maximum of all elements.
    fn max<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Clone + LowerBounded + PartialOrd;

    /// Reshape the tensor to the given shape, keeping the number of elements unchanged.
    fn reshape<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>
    where
        E: Clone;

    /// Permute the tensor axes according to the given permutation.
    fn permute<E>(t: &Self::Repr<E>, permutation: &[usize]) -> Option<Self::Repr<E>>;

    /// Expand singleton axes in a tensor to a larger size.
    fn expand<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>;
}

/// Mid-level tensor operations that must be differentiable.
///
/// Functions in this trait are considered mid-level operations on tensors that must be
/// differentiable. All operations are translated to those defined in [`LL`] to support a wide range
/// of hardwares.
pub trait ML {
    /// Representation of a tensor parameterized by its element type `E`.
    type Repr<E>: Clone;

    /// Create a new tensor with the given shape and data.
    fn new<E>(shape: &[NonZeroUsize], data: &[E]) -> Option<Self::Repr<E>>
    where
        E: Clone;

    /// Return the shape of the tensor.
    fn shape<E>(t: &Self::Repr<E>) -> &[NonZeroUsize];

    /// Apply exp to each element.
    fn exp<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: num::Float;

    /// Apply the natural logarithm to each element.
    fn ln<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: num::Float;

    /// Add `rhs` to `lhs`, element-wise.
    fn add<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Add<Output = E>;

    /// Subtract `rhs` from `lhs`, element-wise.
    fn sub<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Sub<Output = E>;

    /// Multiply `lhs` by `rhs`, element-wise.
    fn mul<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Mul<Output = E>;

    /// Divide `lhs` by `rhs`, element-wise.
    fn div<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Div<Output = E>;

    /// Raise `lhs` to the power of `rhs`, element-wise.
    fn pow<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: num::Float;

    /// Compare 'lhs' with 'rhs', element-wise, returning 1s where the elements are equal.
    fn eq<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<bool>>
    where
        E: PartialEq;

    /// Reduce along the given axes by summing all elements.
    fn sum<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Clone + num::Zero + ops::Add<Output = E>;

    /// Reduce along the given axes by getting the maximum of all elements.
    fn max<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Clone + LowerBounded + PartialOrd;

    /// Reshape the tensor to the given shape, keeping the number of elements unchanged.
    fn reshape<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>
    where
        E: Clone;

    /// Permute the tensor axes according to the given permutation.
    fn permute<E>(t: &Self::Repr<E>, permutation: &[usize]) -> Option<Self::Repr<E>>;

    /// Expand singleton axes in a tensor to a larger size.
    fn expand<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>;
}

// The set of low-level operations is a subset of the set of mid-level operations. This effectively
// makes all low-level operations differentiable.
//
// Both [`LL`] and [`ML`] seems to share that same set of operations. However, this won't be the
// case as more functionalities are added. In general, the set of operations in [`LL`] should be
// kept small so a majority of hardwares can be supported, while the set of operations in [`ML`] can
// grow as much as needed.
impl<I> ML for I
where
    I: LL,
{
    type Repr<E> = I::Repr<E>;

    fn new<E>(shape: &[NonZeroUsize], data: &[E]) -> Option<Self::Repr<E>>
    where
        E: Clone,
    {
        I::new::<E>(shape, data)
    }

    fn shape<E>(t: &Self::Repr<E>) -> &[NonZeroUsize] {
        I::shape::<E>(t)
    }

    fn exp<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: num::Float,
    {
        I::exp::<E>(t)
    }

    fn ln<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: num::Float,
    {
        I::ln::<E>(t)
    }

    fn add<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Add<Output = E>,
    {
        I::add::<E>(lhs, rhs)
    }

    fn sub<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Sub<Output = E>,
    {
        I::sub::<E>(lhs, rhs)
    }

    fn mul<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Mul<Output = E>,
    {
        I::mul::<E>(lhs, rhs)
    }

    fn div<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Clone + ops::Div<Output = E>,
    {
        I::div::<E>(lhs, rhs)
    }

    fn pow<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: num::Float,
    {
        I::pow::<E>(lhs, rhs)
    }

    fn eq<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<bool>>
    where
        E: PartialEq,
    {
        I::eq::<E>(lhs, rhs)
    }

    fn sum<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Clone + num::Zero + ops::Add<Output = E>,
    {
        I::sum::<E>(t, axes)
    }

    fn max<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Clone + LowerBounded + PartialOrd,
    {
        I::max::<E>(t, axes)
    }

    fn reshape<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>
    where
        E: Clone,
    {
        I::reshape::<E>(t, shape)
    }

    fn permute<E>(t: &Self::Repr<E>, permutation: &[usize]) -> Option<Self::Repr<E>> {
        I::permute::<E>(t, permutation)
    }

    fn expand<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>> {
        I::expand::<E>(t, shape)
    }
}

/// An operation for returning a clone of a tensor on the CPU.
pub trait ToCpu: LL {
    /// Return a clone of the tensor on the CPU.
    fn to_cpu<E>(t: &Self::Repr<E>) -> cpu::Tensor<E>;
}
