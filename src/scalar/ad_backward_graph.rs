//! Reverse mode automatic differentiation on scalars. This implementation uses
//! a heap-allocated linked list to store the computation graph.

use std::{
    cell::RefCell,
    collections::{HashSet, VecDeque},
    hash::Hash,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
    time,
};

use rand::{seq::SliceRandom, Rng};
use rand_distr::Distribution;

use crate::dataset::VectorMapping;

#[derive(Debug, Clone)]
struct Call {
    from: [Option<Var>; 2],
    grad: [f64; 2],
}

#[derive(Debug)]
struct VarInner {
    value: f64,
    grad: f64,
    call: Call,
}

/// A scalar variable.
#[derive(Debug, Clone)]
pub struct Var {
    inner: Rc<RefCell<VarInner>>,
}

impl Eq for Var {}
impl PartialEq for Var {
    fn eq(&self, other: &Self) -> bool {
        self.inner.as_ptr() == other.inner.as_ptr()
    }
}

impl Hash for Var {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize(self.inner.as_ptr() as usize);
    }
}

impl Add for Var {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let value = self.inner.borrow().value + rhs.inner.borrow().value;
        let call = Call {
            from: [Some(self), Some(rhs)],
            grad: [1.0, 1.0],
        };
        Self::from_call(value, call)
    }
}

impl Add for &Var {
    type Output = Var;

    fn add(self, rhs: Self) -> Self::Output {
        let value = self.inner.borrow().value + rhs.inner.borrow().value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [1.0, 1.0],
        };
        Var::from_call(value, call)
    }
}

impl Sub for Var {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        let value = self.inner.borrow().value - rhs.inner.borrow().value;
        let call = Call {
            from: [Some(self), Some(rhs)],
            grad: [1.0, -1.0],
        };
        Self::from_call(value, call)
    }
}

impl Sub for &Var {
    type Output = Var;

    fn sub(self, rhs: Self) -> Self::Output {
        let value = self.inner.borrow().value - rhs.inner.borrow().value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [1.0, -1.0],
        };
        Var::from_call(value, call)
    }
}

impl Mul for Var {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_value = self.inner.borrow().value;
        let rhs_value = rhs.inner.borrow().value;
        let value = lhs_value * rhs_value;
        let call = Call {
            from: [Some(self), Some(rhs)],
            grad: [rhs_value, lhs_value],
        };
        Self::from_call(value, call)
    }
}

impl Mul for &Var {
    type Output = Var;

    fn mul(self, rhs: Self) -> Self::Output {
        let lhs_value = self.inner.borrow().value;
        let rhs_value = rhs.inner.borrow().value;
        let value = lhs_value * rhs_value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [rhs_value, lhs_value],
        };
        Var::from_call(value, call)
    }
}

impl Div for Var {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let lhs_value = self.inner.borrow().value;
        let rhs_value = rhs.inner.borrow().value;
        let value = lhs_value / rhs_value;
        let call = Call {
            from: [Some(self), Some(rhs)],
            grad: [1.0 / rhs_value, -lhs_value / (rhs_value * rhs_value)],
        };
        Self::from_call(value, call)
    }
}

impl Div for &Var {
    type Output = Var;

    fn div(self, rhs: Self) -> Self::Output {
        let lhs_value = self.inner.borrow().value;
        let rhs_value = rhs.inner.borrow().value;
        let value = lhs_value / rhs_value;
        let call = Call {
            from: [Some(self.clone()), Some(rhs.clone())],
            grad: [1.0 / rhs_value, -lhs_value / (rhs_value * rhs_value)],
        };
        Var::from_call(value, call)
    }
}

impl IntoIterator for Var {
    type Item = Self;
    type IntoIter = VarIterator;

    fn into_iter(self) -> Self::IntoIter {
        VarIterator::new(self)
    }
}

impl IntoIterator for &Var {
    type Item = Var;
    type IntoIter = VarIterator;

    fn into_iter(self) -> Self::IntoIter {
        VarIterator::new(self.clone())
    }
}

impl Var {
    /// Creates a new variable with the given value.
    #[must_use]
    pub fn new(value: f64) -> Self {
        let call = Call {
            from: [None, None],
            grad: [0.0, 0.0],
        };
        Self::from_call(value, call)
    }

    /// Gets the value of the variable.
    #[must_use]
    pub fn value(&self) -> f64 {
        self.inner.borrow().value
    }

    /// Gets the gradient of the variable.
    #[must_use]
    pub fn grad(&self) -> f64 {
        self.inner.borrow().grad
    }

    /// Computes the gradients of all variables in the graph
    /// rooted at this variable.
    pub fn backward(&self) {
        self.inner.borrow_mut().grad = 1.0;
        for var in self {
            var.accumulate_grad();
        }
    }

    /// Apply gradient descent to the variable.
    pub fn learn(&self, rate: f64) {
        let mut inner = self.inner.borrow_mut();
        inner.value -= rate * inner.grad;
        inner.grad = 0.0;
    }

    /// Returns a new variable with the same value as this variable.
    #[must_use]
    pub fn identity(&self) -> Self {
        let value = self.inner.borrow().value;
        let call = Call {
            from: [Some(self.clone()), None],
            grad: [1.0, 0.0],
        };
        Self::from_call(value, call)
    }

    /// Returns a new variable with the sigmoid of this variable.
    #[must_use]
    pub fn sigmoid(&self) -> Self {
        let exp = self.inner.borrow().value.exp();
        let value = exp / (1.0 + exp);
        let call = Call {
            from: [Some(self.clone()), None],
            grad: [value * (1.0 - value), 0.0],
        };
        Self::from_call(value, call)
    }

    /// Returns an iterator over all variables used in the computation graph that produces this variable.
    #[must_use]
    pub fn iter(&self) -> VarIterator {
        self.into_iter()
    }

    fn accumulate_grad(&self) {
        let inner = self.inner.borrow();
        for (parent, grad) in inner.call.from.iter().zip(&inner.call.grad) {
            if let Some(p) = parent {
                let mut parent_inner = p.inner.borrow_mut();
                parent_inner.grad += inner.grad * grad;
            }
        }
    }

    fn from_call(value: f64, call: Call) -> Self {
        let inner = Rc::new(RefCell::new(VarInner {
            value,
            grad: 0.0,
            call,
        }));
        Self { inner }
    }
}

/// An iterator over the variables in the computation graph rooted at some variable.
#[derive(Debug)]
pub struct VarIterator {
    seen: HashSet<Var>,
    deque: VecDeque<Var>,
}

impl VarIterator {
    /// Creates a new iterator over the variables in the computation graph.
    #[must_use]
    pub fn new(var: Var) -> Self {
        let mut seen = HashSet::default();
        seen.insert(var.clone());
        let mut deque = VecDeque::new();
        deque.push_back(var);
        Self { seen, deque }
    }
}

impl Iterator for VarIterator {
    type Item = Var;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.deque.pop_front();
        if let Some(it) = &item {
            let it = it.inner.borrow();
            for parent in it.call.from.iter().flatten() {
                if !self.seen.contains(parent) {
                    self.seen.insert(parent.clone());
                    self.deque.push_back(parent.clone());
                }
            }
        }
        item
    }
}

/// A neuron with a bias and weights.
#[derive(Debug)]
pub struct Neuron {
    bias: Var,
    weights: Vec<Var>,
    activation: fn(&Var) -> Var,
}

impl Neuron {
    /// Creates a new neuron with random weights and bias.
    pub fn rand<R, D>(
        rng: &mut R,
        distribution: &D,
        activation: fn(&Var) -> Var,
        input_size: usize,
    ) -> Self
    where
        R: Rng,
        D: Distribution<f64>,
    {
        let bias = Var::new(rng.gen());
        let weights = (0..input_size)
            .map(|_| Var::new(rng.sample(distribution)))
            .collect();
        Self {
            bias,
            weights,
            activation,
        }
    }

    /// Computes the output of the neuron given an input.
    ///
    /// # Panics
    ///
    /// Panics if the input size mismatches the weights size.
    #[must_use]
    pub fn forward(&self, input: &[Var]) -> Var {
        assert_eq!(input.len(), self.weights.len());
        (self.activation)(
            &self
                .weights
                .iter()
                .zip(input)
                .fold(self.bias.clone(), |acc, (w, x)| acc + w * x),
        )
    }

    fn parameters(&self) -> Vec<Var> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

/// A layer of neurons.
#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Creates a new layer of neurons with random weights and biases.
    pub fn rand<R, D>(
        rng: &mut R,
        distribution: &D,
        activation: fn(&Var) -> Var,
        input_size: usize,
        output_size: usize,
    ) -> Self
    where
        R: Rng,
        D: Distribution<f64>,
    {
        let neurons = (0..output_size)
            .map(|_| Neuron::rand(rng, distribution, activation, input_size))
            .collect();
        Self { neurons }
    }

    /// Computes the output of the layer given an input.
    #[must_use]
    pub fn forward(&self, input: &[Var]) -> Vec<Var> {
        self.neurons.iter().map(|n| n.forward(input)).collect()
    }

    /// Gets the parameters of the layer.
    pub fn parameters(&self) -> Vec<Var> {
        self.neurons.iter().flat_map(Neuron::parameters).collect()
    }
}

/// A multi-layer perceptron.
#[derive(Debug)]
pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    /// Creates a new multi-layer perceptron with the given layers.
    #[must_use]
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    /// Computes the output of the multi-layer perceptron given an input.
    #[must_use]
    pub fn forward(&self, input: &[Var]) -> Vec<Var> {
        match self.layers.split_first() {
            Some((layer, ls)) => ls
                .iter()
                .fold(layer.forward(input), |acc, layer| layer.forward(&acc)),
            None => Vec::new(),
        }
    }

    /// Trains the multi-layer perceptron on the given dataset.
    pub fn train<R, const M: usize, const N: usize>(
        &self,
        rng: &mut R,
        dataset: &[VectorMapping<M, N>],
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        print_interval: usize,
    ) where
        R: Rng,
    {
        let start = time::Instant::now();
        let parameters = self.parameters();
        for epoch in 0..epochs {
            let mut dataset = Vec::from(dataset);
            dataset.shuffle(rng);
            for batch in dataset.chunks(batch_size) {
                let loss = self.loss_dataset(batch, mse);
                loss.backward();
                for param in &parameters {
                    param.learn(learning_rate);
                }
            }
            if print_interval != 0 && (epoch + 1) % print_interval == 0 {
                let loss = self.loss_dataset(&dataset, mse);
                println!("epoch: {}, loss: {}", epoch + 1, loss.value());
            }
        }
        let duration = start.elapsed();
        println!("training took {}ms", duration.as_millis());
    }

    fn parameters(&self) -> Vec<Var> {
        self.layers.iter().flat_map(Layer::parameters).collect()
    }

    fn loss_sample<const M: usize, const N: usize>(
        &self,
        sample: &VectorMapping<M, N>,
        metric: fn(&[Var], &[Var]) -> Var,
    ) -> Var {
        let input: Vec<_> = sample.input.iter().map(|x| Var::new(*x)).collect();
        let output: Vec<_> = sample.output.iter().map(|x| Var::new(*x)).collect();
        let pred = self.forward(&input);
        (metric)(&pred, &output)
    }

    fn loss_dataset<const M: usize, const N: usize>(
        &self,
        samples: &[VectorMapping<M, N>],
        metric: fn(&[Var], &[Var]) -> Var,
    ) -> Var {
        let mut loss = Var::new(0.0);
        for sample in samples {
            loss = &loss + &self.loss_sample(sample, metric);
        }
        let count = u32::try_from(samples.len()).unwrap();
        &loss / &Var::new(f64::from(count))
    }
}

fn mse(pred: &[Var], output: &[Var]) -> Var {
    let mut loss = Var::new(0.0);
    for (p, o) in pred.iter().zip(output) {
        let diff = p - o;
        loss = &loss + &(&diff * &diff);
    }
    let count = u32::try_from(output.len()).unwrap();
    &loss / &Var::new(f64::from(count))
}
