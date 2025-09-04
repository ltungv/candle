//! Reverse-mode auto-differentiation.

use std::{
    cell::{self, RefCell},
    fmt, iter,
    marker::PhantomData,
    num::NonZeroUsize,
    rc::Rc,
};

use crate::{
    autodiff::ops::{
        Add, Binary, BinaryDiff, Div, Exp, Expand, Ln, Max, Mul, Permute, Pow, Reshape, Sub, Sum,
        Unary, UnaryDiff,
    },
    tensor::{
        ops::ML,
        typ::{Elem, Float, Num},
        Tensor,
    },
};

type AdTensor<T, E, Ops> = Tensor<Reverse<T>, E, ReverseOps<Ops>>;

/// Compute a reverse-mode vector-Jacobian product of `f`.
///
/// Return a tuple of the function's result and a [`PullBack`] object. [`PullBack::vjp`] can be
/// used to compute the vector-Jacobian product of the function at any cotangent.
pub fn vjp<F, T, E, Ops>(
    primal: &Tensor<T, E, Ops>,
    f: F,
) -> (Tensor<T, E, Ops>, PullBack<T, E, Ops>)
where
    F: FnOnce(AdTensor<T, E, Ops>) -> AdTensor<T, E, Ops>,
    T: 'static + Clone,
    E: Num,
    Ops: 'static + ML<Repr<E> = T>,
{
    PullBack::on(&[primal], |mut ts| f(ts.swap_remove(0)))
}

/// Compute a reverse-mode vector-Jacobian product of `f`.
///
/// Return a tuple of the function's result and a [`PullBack`] object. [`PullBack::vjp`] can be
/// used to compute the vector-Jacobian product of the function at any cotangent.
pub fn vjpn<F, T, E, Ops>(
    primals: &[&Tensor<T, E, Ops>],
    f: F,
) -> (Tensor<T, E, Ops>, PullBack<T, E, Ops>)
where
    F: FnOnce(Vec<AdTensor<T, E, Ops>>) -> AdTensor<T, E, Ops>,
    T: 'static + Clone,
    E: Num,
    Ops: 'static + ML<Repr<E> = T>,
{
    PullBack::on(primals, f)
}

/// Evaluate a function at the given primal, returning its result and the gradient.
pub fn grad<F, T, E, Ops>(primal: &Tensor<T, E, Ops>, f: F) -> Tensor<T, E, Ops>
where
    F: FnOnce(AdTensor<T, E, Ops>) -> AdTensor<T, E, Ops>,
    T: 'static + Clone,
    E: Num,
    Ops: 'static + ML<Repr<E> = T>,
{
    let (out, pb) = PullBack::on(&[primal], |mut ts| f(ts.swap_remove(0)));
    let cotangent = Tensor::full(out.shape(), E::one());
    let mut cotangents = pb.vjp(&cotangent);
    cotangents.swap_remove(0)
}

/// Evaluate a function at the given primals, returning its result and the gradient.
#[allow(clippy::type_complexity)]
pub fn gradn<F, T, E, Ops>(primals: &[&Tensor<T, E, Ops>], f: F) -> Vec<Tensor<T, E, Ops>>
where
    F: FnOnce(Vec<AdTensor<T, E, Ops>>) -> AdTensor<T, E, Ops>,
    T: 'static + Clone,
    E: Num,
    Ops: 'static + ML<Repr<E> = T>,
{
    let (out, pb) = PullBack::on(primals, f);
    let cotangent = Tensor::full(out.shape(), E::one());
    pb.vjp(&cotangent)
}

/// The back-propagation function computing a [`Vec`] of cotangents given a cotangent.
#[derive(Debug)]
pub struct PullBack<T, E, Ops> {
    tape: Rc<Tape<T>>,
    zero_primals: Vec<T>,
    result_shape: Vec<NonZeroUsize>,
    result_adjoint_index: Option<usize>,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> PullBack<T, E, Ops>
where
    Ops: 'static + ML<Repr<E> = T>,
{
    /// Take a cotangent with the same shape as the result of the originating VJP function, and
    /// return a [`Vec`] of cotangents of the same length and shapes as the VJP's primals,
    pub fn vjp(&self, cotangent: &Tensor<T, E, Ops>) -> Vec<Tensor<T, E, Ops>>
    where
        T: 'static + Clone,
        E: Num,
    {
        let ts = self.result_adjoint_index.map_or_else(
            || self.zero_primals.clone(),
            |idx| self.reverse(idx, cotangent.as_raw()),
        );
        ts.into_iter().map(|raw| Tensor::from_raw(raw)).collect()
    }

    fn on<F>(primals: &[&Tensor<T, E, Ops>], f: F) -> (Tensor<T, E, Ops>, Self)
    where
        T: Clone,
        E: Num,
        F: FnOnce(Vec<AdTensor<T, E, Ops>>) -> AdTensor<T, E, Ops>,
    {
        // Construct a cache of zero primals to be used where the VJP is not computed.
        let zero_primals: Vec<_> = primals
            .iter()
            .map(|t| Ops::full::<E>(t.shape(), E::zero()))
            .collect();

        // Construct the variables from the given primals and add them to the tape.
        let tape = Rc::new(Tape::new());
        let vars: Vec<_> = primals
            .iter()
            .map(|&t| {
                Tensor::from_raw(Reverse::Traced(Trace::new(
                    &tape,
                    TapeNode::Var,
                    t.as_raw().clone(),
                )))
            })
            .collect();

        // Execute the function. Because [`Reverse`] is used to construct the variable, a dependency
        // graph will be created on the tape once the function returns.
        let result = f(vars);
        let result_shape = result.shape().to_vec();
        let (primal, result_adjoint_index) = match result.into_raw() {
            Reverse::Lifted(primal) => (primal, None),
            Reverse::Traced(trace) => (trace.primal, Some(trace.index)),
        };

        (
            Tensor::from_raw(primal),
            Self {
                tape: Rc::clone(&tape),
                zero_primals,
                result_shape,
                result_adjoint_index,
                _marker: PhantomData,
            },
        )
    }

    // Back-propagate from the given adjoint index.
    #[allow(clippy::similar_names)]
    fn reverse(&self, adjoint_index: usize, adjoint: &T) -> Vec<T>
    where
        T: 'static + Clone,
        E: Num,
    {
        assert_eq!(Ops::shape::<E>(adjoint), self.result_shape);
        let mut adjoints: Vec<_> = iter::repeat_with(Adjoint::<T, E, Ops>::empty)
            .take(adjoint_index + 1)
            .collect();

        adjoints[adjoint_index] = Adjoint::new(adjoint.clone());
        for i in (0..=adjoint_index).rev() {
            if let Some(adjoint) = adjoints[i].as_ref() {
                let nodes = self.tape.nodes();
                match &nodes[i] {
                    TapeNode::Var => break,
                    TapeNode::Unary(op, a) => {
                        let df = op.df(adjoint);
                        adjoints[*a].accumulate(df);
                    }
                    TapeNode::BinaryDA(op, a) => {
                        let dfda = op.dfda(adjoint);
                        adjoints[*a].accumulate(dfda);
                    }
                    TapeNode::BinaryDB(op, b) => {
                        let dfdb = op.dfdb(adjoint);
                        adjoints[*b].accumulate(dfdb);
                    }
                    TapeNode::Binary(op, a, b) => {
                        let dfda = op.dfda(adjoint);
                        let dfdb = op.dfdb(adjoint);
                        adjoints[*a].accumulate(dfda);
                        adjoints[*b].accumulate(dfdb);
                    }
                }
            }
            adjoints.pop();
        }
        assert_eq!(adjoints.len(), self.zero_primals.len());
        adjoints
            .into_iter()
            .zip(self.zero_primals.iter())
            .map(|(mut x, z)| x.take().unwrap_or_else(|| z.clone()))
            .collect()
    }
}

/// Helper type for an uninitialized adjoint.
struct Adjoint<T, E, Ops> {
    df: Option<T>,
    _marker: PhantomData<(E, Ops)>,
}

impl<T, E, Ops> Adjoint<T, E, Ops> {
    const fn empty() -> Self {
        Self {
            df: None,
            _marker: PhantomData,
        }
    }

    const fn new(df: T) -> Self {
        Self {
            df: Some(df),
            _marker: PhantomData,
        }
    }

    const fn take(&mut self) -> Option<T> {
        self.df.take()
    }

    const fn as_ref(&self) -> Option<&T> {
        self.df.as_ref()
    }

    fn accumulate(&mut self, df: T)
    where
        E: Num,
        Ops: ML<Repr<E> = T>,
    {
        match self.df.as_mut() {
            None => self.df = Some(df),
            Some(c) => *c = Ops::add::<E>(c, &df),
        }
    }
}

/// Mid-level tensor operations supporting reverse-mode auto-differentiation.
#[derive(Debug)]
pub struct ReverseOps<Ops>(PhantomData<Ops>);

impl<Ops> ML for ReverseOps<Ops>
where
    Ops: 'static + ML,
{
    type Repr<E> = Reverse<Ops::Repr<E>>;

    fn new<E>(shape: &[NonZeroUsize], data: &[E]) -> Self::Repr<E>
    where
        E: Elem,
    {
        Reverse::Lifted(Ops::new::<E>(shape, data))
    }

    fn convert<E, EInto>(t: &Self::Repr<E>) -> Self::Repr<EInto>
    where
        E: Elem + Into<EInto>,
        EInto: Elem,
    {
        Reverse::Lifted(Ops::convert::<E, EInto>(t.primal()))
    }

    fn shape<E>(t: &Self::Repr<E>) -> &[NonZeroUsize] {
        Ops::shape::<E>(t.primal())
    }

    fn exp<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float,
    {
        t.unary::<Exp<Ops::Repr<E>, E, Ops>>(&())
    }

    fn ln<E>(t: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float,
    {
        t.unary::<Ln<Ops::Repr<E>, E, Ops>>(&())
    }

    fn add<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Num,
    {
        lhs.binary::<Add<E, Ops>>(rhs)
    }

    fn sub<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Num,
    {
        lhs.binary::<Sub<E, Ops>>(rhs)
    }

    fn mul<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Num,
    {
        lhs.binary::<Mul<Ops::Repr<E>, E, Ops>>(rhs)
    }

    fn div<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Num,
    {
        lhs.binary::<Div<Ops::Repr<E>, E, Ops>>(rhs)
    }

    fn pow<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<E>
    where
        E: Float,
    {
        lhs.binary::<Pow<Ops::Repr<E>, E, Ops>>(rhs)
    }

    fn eq<E>(lhs: &Self::Repr<E>, rhs: &Self::Repr<E>) -> Self::Repr<bool>
    where
        E: Elem + PartialEq,
    {
        Reverse::Lifted(Ops::eq::<E>(lhs.primal(), rhs.primal()))
    }

    fn sum<E>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E>
    where
        E: Num,
    {
        t.unary::<Sum<E, Ops>>(axes)
    }

    fn max<E>(t: &Self::Repr<E>, axes: &[usize]) -> Self::Repr<E>
    where
        E: Num,
    {
        t.unary::<Max<Ops::Repr<E>, E, Ops>>(axes)
    }

    fn reshape<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Self::Repr<E>
    where
        E: Elem,
    {
        t.unary::<Reshape<E, Ops>>(shape)
    }

    fn permute<E>(t: &Self::Repr<E>, permutation: &[usize]) -> Self::Repr<E>
    where
        E: Elem,
    {
        t.unary::<Permute<E, Ops>>(permutation)
    }

    fn expand<E>(t: &Self::Repr<E>, shape: &[NonZeroUsize]) -> Self::Repr<E>
    where
        E: Num,
    {
        t.unary::<Expand<E, Ops>>(shape)
    }
}

/// Reverse auto-differentation support.
#[derive(Clone, Debug)]
pub enum Reverse<T> {
    /// A lifted value whose gradient won't be calculated.
    Lifted(T),

    /// A traced value tracking the dependency graph of operations.
    Traced(Trace<T>),
}

impl<T> Reverse<T> {
    const fn primal(&self) -> &T {
        match self {
            Self::Lifted(primal) => primal,
            Self::Traced(trace) => &trace.primal,
        }
    }

    fn unary<Op>(&self, args: &Op::Args) -> Self
    where
        Op: 'static + Unary<T>,
    {
        match self {
            Self::Lifted(primal) => {
                let (primal, _) = Op::call(primal, args);
                Self::Lifted(primal)
            }

            Self::Traced(trace) => {
                let (primal, grad) = Op::call(&trace.primal, args);
                Self::Traced(Trace::new(
                    &trace.tape,
                    TapeNode::Unary(Box::new(grad), trace.index),
                    primal,
                ))
            }
        }
    }

    fn binary<Op>(&self, other: &Self) -> Self
    where
        Op: 'static + Binary<T>,
    {
        match (self, other) {
            (Self::Lifted(lhs), Self::Lifted(rhs)) => {
                let (primal, _) = Op::call(lhs, rhs);
                Self::Lifted(primal)
            }

            (Self::Lifted(lhs), Self::Traced(rhs)) => {
                let (primal, grad) = Op::call(lhs, &rhs.primal);
                Self::Traced(Trace::new(
                    &rhs.tape,
                    TapeNode::BinaryDB(Box::new(grad), rhs.index),
                    primal,
                ))
            }

            (Self::Traced(lhs), Self::Lifted(rhs)) => {
                let (primal, grad) = Op::call(&lhs.primal, rhs);
                Self::Traced(Trace::new(
                    &lhs.tape,
                    TapeNode::BinaryDA(Box::new(grad), lhs.index),
                    primal,
                ))
            }

            (Self::Traced(lhs), Self::Traced(rhs)) => {
                assert!(Rc::ptr_eq(&lhs.tape, &rhs.tape));
                let (primal, grad) = Op::call(&lhs.primal, &rhs.primal);
                Self::Traced(Trace::new(
                    &lhs.tape,
                    TapeNode::Binary(Box::new(grad), lhs.index, rhs.index),
                    primal,
                ))
            }
        }
    }
}

/// Container for a primal, an index to its tangent, and a reference to the tape.
#[derive(Clone, Debug)]
pub struct Trace<T> {
    tape: Rc<Tape<T>>,
    index: usize,
    primal: T,
}

impl<T> Trace<T> {
    fn new(tape: &Rc<Tape<T>>, node: TapeNode<T>, primal: T) -> Self {
        let index = tape.push(node);
        Self {
            tape: Rc::clone(tape),
            index,
            primal,
        }
    }
}

#[derive(Debug)]
struct Tape<T> {
    nodes: RefCell<Vec<TapeNode<T>>>,
}

impl<T> Default for Tape<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Tape<T> {
    #[must_use]
    const fn new() -> Self {
        Self {
            nodes: RefCell::new(vec![]),
        }
    }

    fn push(&self, node: TapeNode<T>) -> usize {
        let mut nodes = self.nodes.borrow_mut();
        let index = nodes.len();
        nodes.push(node);
        index
    }

    fn nodes(&self) -> cell::Ref<'_, Vec<TapeNode<T>>> {
        self.nodes.borrow()
    }
}

enum TapeNode<T> {
    Var,
    Unary(Box<dyn UnaryDiff<T>>, usize),
    BinaryDA(Box<dyn BinaryDiff<T>>, usize),
    BinaryDB(Box<dyn BinaryDiff<T>>, usize),
    Binary(Box<dyn BinaryDiff<T>>, usize, usize),
}

impl<T> fmt::Debug for TapeNode<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Var => f.debug_tuple("TapeNode::Var").finish(),

            Self::Unary(op, idx) => f
                .debug_tuple("TapeNode::Unary")
                .field(&format_args!("{op:#p}"))
                .field(idx)
                .finish(),

            Self::BinaryDA(op, idx) => f
                .debug_tuple("TapeNode::BinaryDA")
                .field(&format_args!("{op:#p}"))
                .field(idx)
                .finish(),

            Self::BinaryDB(op, idx) => f
                .debug_tuple("TapeNode::BinaryDB")
                .field(&format_args!("{op:#p}"))
                .field(idx)
                .finish(),

            Self::Binary(op, idx0, idx1) => f
                .debug_tuple("TapeNode::Binary")
                .field(&format_args!("{op:#p}"))
                .field(idx0)
                .field(idx1)
                .finish(),
        }
    }
}

#[allow(clippy::similar_names)]
#[cfg(test)]
mod tests {
    use std::f32;

    use rand::{rngs::StdRng, Rng, SeedableRng};

    use crate::{
        autodiff::reverse::{grad, gradn},
        tensor::{cpu, ops::ML, shape, typ::Float, Tensor},
    };

    type T32 = Tensor<cpu::Tensor<f32>, f32, cpu::TensorOps>;

    fn float_eq<T: num::Float>(lhs: T, rhs: T, tolerance: T) -> bool {
        (lhs - rhs).abs() <= tolerance
    }

    #[test]
    fn higher_order_grad() {
        let two = T32::scalar(2.0);
        let df = grad(&two, |t| t.tanh());
        let ddf = grad(&two, |t| grad(&t, |t| t.tanh()));
        let dddf = grad(&two, |t| grad(&t, |t| grad(&t, |t| t.tanh())));

        assert!(df.shape().is_empty());
        assert!(ddf.shape().is_empty());
        assert!(dddf.shape().is_empty());

        assert!(float_eq(df.ravel()[0], 0.070_650_82, f32::EPSILON));
        assert!(float_eq(ddf.ravel()[0], -0.136_218_68, f32::EPSILON));
        assert!(float_eq(dddf.ravel()[0], 0.252_654_08, f32::EPSILON));
    }

    #[test]
    fn jax_autodiff_cookbook() {
        fn predict<T, E, Ops>(
            w: &Tensor<T, E, Ops>,
            b: &Tensor<T, E, Ops>,
            inputs: &Tensor<T, E, Ops>,
        ) -> Tensor<T, E, Ops>
        where
            E: Float,
            Ops: ML<Repr<E> = T>,
        {
            (&inputs.matmul(w) + b).sigmoid()
        }

        // Training loss is the negative log-likelihood of the training examples.
        fn loss<T, E, Ops>(
            w: &Tensor<T, E, Ops>,
            b: &Tensor<T, E, Ops>,
            inputs: &Tensor<T, E, Ops>,
            targets: &Tensor<T, E, Ops>,
        ) -> Tensor<T, E, Ops>
        where
            E: Float,
            Ops: ML<Repr<E> = T>,
        {
            let one = Tensor::<T, E, Ops>::scalar(E::one());
            let prediction = predict(w, b, inputs);
            let label_probs = &prediction * targets + (&one - &prediction) * &(&one - targets);
            label_probs.ln().sum(&[0]).neg()
        }

        // Build a toy dataset.
        let targets = T32::new(&shape([4]), &[1.0, 1.0, 0.0, 1.0]);
        let inputs = T32::new(
            &shape([4, 3]),
            &[
                0.52, 1.12, 0.77, //
                0.88, -1.08, 0.15, //
                0.52, 0.06, -1.30, //
                0.74, -2.49, 1.39,
            ],
        );

        let mut rng = StdRng::seed_from_u64(0);
        let ws: Vec<f32> = (&mut rng).random_iter().take(3).collect();
        let bs: Vec<f32> = (&mut rng).random_iter().take(1).collect();

        let w = T32::new(&shape([3]), &ws);
        let b = T32::new(&shape([1]), &bs);
        let l = loss(&w, &b, &inputs, &targets);

        // Differentiate loss wrt weights.
        let w_grad = grad(&w, |w| {
            loss(&w, &b.lift_rev(), &inputs.lift_rev(), &targets.lift_rev())
        });

        // Differentiate loss wrt biases.
        let b_grad = grad(&b, |b| {
            loss(&w.lift_rev(), &b, &inputs.lift_rev(), &targets.lift_rev())
        });

        // Differentiate loss wrt weights and biases - should give the same answer.
        let wb_grads = gradn(&[&w, &b], |ts| {
            let w = &ts[0];
            let b = &ts[1];
            loss(w, b, &inputs.lift_rev(), &targets.lift_rev())
        });
        assert!(w_grad.eq(&wb_grads[0]).ravel().iter().all(|x| *x));
        assert!(b_grad.eq(&wb_grads[1]).ravel().iter().all(|x| *x));

        let new_w = &w - &w_grad;
        let new_b = &b - &b_grad;
        let new_l = loss(&new_w, &new_b, &inputs, &targets);
        assert!(new_l
            .ravel()
            .iter()
            .zip(l.ravel().iter())
            .all(|(new, old)| new < old));

        let eps = T32::scalar(1e-4);
        let half_eps = &eps / T32::scalar(2.0);
        let loss_l = loss(&w, &(&b - &half_eps), &inputs, &targets);
        let loss_r = loss(&w, &(&b + &half_eps), &inputs, &targets);
        let b_grad_numerical = (loss_r - loss_l) / &eps;

        assert!(b_grad
            .ravel()
            .iter()
            .zip(b_grad_numerical.ravel().iter())
            .all(|(ad, num)| float_eq(*ad, *num, 0.005)));
    }
}
