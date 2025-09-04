//! Auto-differentiation.

use crate::{
    autodiff::reverse::{Reverse, ReverseOps},
    tensor::{ops::ML, Tensor},
};

mod ops;
pub mod reverse;

/// Trait implemented by autodiff system to lift a tensor into the autodiff context.
///
/// Lifted tensors are treated as constants and their derivatives are not calculated.
pub trait Lift<T>
where
    Self: Sized,
    T: Clone,
{
    /// Tensor operations in the autodiff context.
    type Ops<Ops>: ML
    where
        Ops: 'static + ML;

    /// Lift a tensor into the autodiff context.
    fn lift<E, Ops>(grounded: &Tensor<T, E, Ops>) -> Tensor<Self, E, Self::Ops<Ops>>
    where
        Ops: 'static + ML;
}

impl<T> Lift<T> for Reverse<T>
where
    T: Clone,
{
    type Ops<Ops>
        = ReverseOps<Ops>
    where
        Ops: 'static + ML;

    fn lift<E, Ops>(grounded: &Tensor<T, E, Ops>) -> Tensor<Self, E, Self::Ops<Ops>>
    where
        Ops: 'static + ML,
    {
        Tensor::from_raw(Self::Lifted(grounded.as_raw().clone()))
    }
}
