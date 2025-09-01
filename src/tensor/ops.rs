//! Traits defining operations across different abstraction levels of a tensor implementation.

use std::num::NonZeroUsize;

use crate::tensor::{
    cpu,
    dtype::{Bool, Elem, Float, Num},
};

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
        E: Elem;

    /// Convert a tensor to another, changing its elements' type.
    fn convert<E, EInto>(t: &Self::Repr<E>) -> Self::Repr<EInto>
    where
        E: Elem + Into<EInto>,
        EInto: Elem;

    /// Return the shape of the tensor.
    fn shape<E>(t: &Self::Repr<E>) -> &[NonZeroUsize];

    /// Apply exp to each element.
    fn exp<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float;

    /// Apply the natural logarithm to each element.
    fn ln<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float;

    /// Add `rhs` to `lhs`, element-wise.
    fn add<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Subtract `rhs` from `lhs`, element-wise.
    fn sub<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Multiply `lhs` by `rhs`, element-wise.
    fn mul<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Divide `lhs` by `rhs`, element-wise.
    fn div<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Raise `lhs` to the power of `rhs`, element-wise.
    fn pow<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Float;

    /// Compare 'lhs' with 'rhs', element-wise, returning 1s where the elements are equal.
    fn eq<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<bool>>
    where
        E: Elem + PartialEq;

    /// Reduce along the given axes by summing all elements.
    fn sum<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Reduce along the given axes by getting the maximum of all elements.
    fn max<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Bool;

    /// Reshape the tensor to the given shape, keeping the number of elements unchanged.
    fn reshape<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>
    where
        E: Elem;

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
        E: Elem;

    /// Convert a tensor to another, changing its elements' type.
    fn convert<E, EInto>(t: &Self::Repr<E>) -> Self::Repr<EInto>
    where
        E: Elem + Into<EInto>,
        EInto: Elem;

    /// Return the shape of the tensor.
    fn shape<E>(t: &Self::Repr<E>) -> &[NonZeroUsize];

    /// Apply exp to each element.
    fn exp<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float;

    /// Apply the natural logarithm to each element.
    fn ln<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float;

    /// Add `rhs` to `lhs`, element-wise.
    fn add<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Subtract `rhs` from `lhs`, element-wise.
    fn sub<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Multiply `lhs` by `rhs`, element-wise.
    fn mul<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Divide `lhs` by `rhs`, element-wise.
    fn div<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Raise `lhs` to the power of `rhs`, element-wise.
    fn pow<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Float;

    /// Compare 'lhs' with 'rhs', element-wise, returning 1s where the elements are equal.
    fn eq<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<bool>>
    where
        E: Elem + PartialEq;

    /// Reduce along the given axes by summing all elements.
    fn sum<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Reduce along the given axes by getting the maximum of all elements.
    fn max<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Reshape the tensor to the given shape, keeping the number of elements unchanged.
    fn reshape<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>
    where
        E: Elem;

    /// Permute the tensor axes according to the given permutation.
    fn permute<E>(t: &Self::Repr<E>, permutation: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Elem;

    /// Expand singleton axes in a tensor to a larger size.
    fn expand<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>
    where
        E: Num;

    /// Create a tensor given its shape filled with a single value.
    fn fill<E>(shape: &[NonZeroUsize], value: E) -> Self::Repr<E>
    where
        E: Num,
    {
        Self::new(&vec![NonZeroUsize::MIN; shape.len()], &[value])
            .and_then(|t| Self::expand::<E>(&t, shape))
            .expect("fill is infallible")
    }

    /// Apply negation to each element.
    fn neg<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Num,
    {
        let zeroes = Self::fill::<E>(Self::shape::<E>(t), E::zero());
        Self::sub::<E>(&zeroes, t).expect("neg is infallible")
    }
}

// The set of low-level operations is a subset of the set of mid-level operations.
//
// Both [`LL`] and [`ML`] seems to share that same set of operations. However, this won't be the
// case as more functionalities are added. In general, the set of operations in [`LL`] should be
// kept small so a majority of hardwares can be supported, while the set of operations in [`ML`] can
// grow as much as needed.
impl<Ops> ML for Ops
where
    Ops: LL,
{
    type Repr<E> = Ops::Repr<E>;

    fn new<E>(shape: &[NonZeroUsize], data: &[E]) -> Option<Self::Repr<E>>
    where
        E: Elem,
    {
        Ops::new::<E>(shape, data)
    }

    fn convert<E, EInto>(t: &Self::Repr<E>) -> Self::Repr<EInto>
    where
        E: Elem + Into<EInto>,
        EInto: Elem,
    {
        Ops::convert::<E, EInto>(t)
    }

    fn shape<E>(t: &Self::Repr<E>) -> &[NonZeroUsize] {
        Ops::shape::<E>(t)
    }

    fn exp<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float,
    {
        Ops::exp::<E>(t)
    }

    fn ln<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float,
    {
        Ops::ln::<E>(t)
    }

    fn add<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        Ops::add::<E>(lhs, rhs)
    }

    fn sub<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        Ops::sub::<E>(lhs, rhs)
    }

    fn mul<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        Ops::mul::<E>(lhs, rhs)
    }

    fn div<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        Ops::div::<E>(lhs, rhs)
    }

    fn pow<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<E>>
    where
        E: Float,
    {
        Ops::pow::<E>(lhs, rhs)
    }

    fn eq<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Option<Self::Repr<bool>>
    where
        E: Elem + PartialEq,
    {
        Ops::eq::<E>(lhs, rhs)
    }

    fn sum<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Num,
    {
        Ops::sum::<E>(t, axes)
    }

    fn max<E>(t: &Self::Repr<E>, axes: &[usize]) -> Option<Self::Repr<E>>
    where
        E: Bool,
    {
        Ops::max::<E>(t, axes)
    }

    fn reshape<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>>
    where
        E: Elem,
    {
        Ops::reshape::<E>(t, shape)
    }

    fn permute<E>(t: &Self::Repr<E>, permutation: &[usize]) -> Option<Self::Repr<E>> {
        Ops::permute::<E>(t, permutation)
    }

    fn expand<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Option<Self::Repr<E>> {
        Ops::expand::<E>(t, shape)
    }
}

/// Tensor that supports converting itself into a representation on the CPU.
pub trait ToCpu: LL {
    /// Return a clone of the tensor on the CPU.
    fn to_cpu<E>(t: &Self::Repr<E>) -> cpu::Tensor<E>;
}
