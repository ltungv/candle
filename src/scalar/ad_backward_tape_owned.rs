//! Try out a different design for the backward tape. Mimic the design of [dfdx]
//! which move the tape around the call graph, such that only the final variable
//! holds the gradients tape.

use std::{
    collections::{BTreeMap, VecDeque},
    iter,
    ops::{Add, Div, Mul, Sub},
    time,
};

use rand::{seq::SliceRandom, Rng};
use rand_distr::Distribution;

use crate::dataset::Sample;

/// A node in the computation graph holding the index of the nodes it depends on and the gradients
/// of the output with respect to each of the input.
#[derive(Clone, Debug)]
struct LocalGradient {
    from: [usize; 2],
    grad: [f64; 2],
}

/// A tape recording the computation graph where each element holds the local derivatives
/// of a variable with respect to variables that it directly depends on.
#[derive(Clone, Debug, Default)]
pub struct Tape {
    nodes: BTreeMap<usize, LocalGradient>,
}

impl Tape {
    /// Merge two tapes together.
    pub fn merge(mut self, mut other: Self) -> Self {
        if self.nodes.len() < other.nodes.len() {
            other.nodes.extend(self.nodes);
            other
        } else {
            self.nodes.extend(other.nodes);
            self
        }
    }
}

/// A variable in the computation graph. Operation on variables return new variables and do not
/// mutate the original ones.
#[derive(Clone, Debug)]
pub struct Var {
    /// The value of the variable
    pub value: f64,
    id: usize,
    tape: Option<Tape>,
}

impl Var {
    /// Create a new variable with no tape.
    pub fn new(value: f64) -> Self {
        static ID: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let id = ID.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Var {
            id,
            value,
            tape: None,
        }
    }

    /// Clone the variable with a new tape containing only itself.
    pub fn trace(&self) -> Self {
        let mut tape = Tape::default();
        tape.nodes.insert(
            self.id,
            LocalGradient {
                from: [self.id, self.id],
                grad: [0.0, 0.0],
            },
        );
        Var {
            id: self.id,
            value: self.value,
            tape: Some(tape),
        }
    }

    /// Returns the gradient of the variable with respect to all variables in the tape.
    pub fn gradients(mut self) -> Option<BTreeMap<usize, f64>> {
        self.tape.take().map(|tape| {
            let mut gradients = BTreeMap::new();
            gradients.insert(self.id, 1.0);
            let mut to_visit = VecDeque::new();
            to_visit.push_back(self.id);
            // Traverse the DAG in topological order.
            while let Some(id) = to_visit.pop_front() {
                if let Some(grad_local) = tape.nodes.get(&id) {
                    // The following `unwrap` is safe because we must inserted the gradient of the currently
                    // visited node.
                    let grad_global = gradients.get(&id).copied().unwrap();
                    // Accumulate the gradients.
                    for i in 0..2 {
                        let grad = gradients.entry(grad_local.from[i]).or_insert(0.0);
                        *grad += grad_local.grad[0] * grad_global;
                    }
                    if grad_local.from[0] != id {
                        to_visit.push_back(grad_local.from[0]);
                    }
                    if grad_local.from[1] != id {
                        to_visit.push_back(grad_local.from[1]);
                    }
                }
            }
            gradients
        })
    }

    /// Create a new variable using the same value as the current one.
    pub fn identity(self) -> Self {
        match self.tape {
            Some(mut tape) => {
                let mut var = Self::new(self.value);
                tape.nodes.insert(
                    var.id,
                    LocalGradient {
                        from: [self.id, self.id],
                        grad: [1.0, 0.0],
                    },
                );
                var.tape = Some(tape);
                var
            }
            None => Self::new(self.value),
        }
    }

    /// Create a new variable using the sigmoid of the current value.
    pub fn sigmoid(self) -> Self {
        let exp = self.value.exp();
        let value = exp / (1.0 + exp);
        match self.tape {
            Some(mut tape) => {
                let mut var = Self::new(self.value);
                tape.nodes.insert(
                    var.id,
                    LocalGradient {
                        from: [self.id, self.id],
                        grad: [value * (1.0 - value), 0.0],
                    },
                );
                var.tape = Some(tape);
                var
            }
            None => Self::new(value),
        }
    }

    fn merge_tape(self, other: Self) -> Option<Tape> {
        match (self.tape, other.tape) {
            (Some(lhs), Some(rhs)) => {
                if self.id == other.id {
                    Some(lhs)
                } else {
                    Some(lhs.merge(rhs))
                }
            }
            (Some(lhs), None) => Some(lhs),
            (None, Some(rhs)) => Some(rhs),
            (None, None) => None,
        }
    }
}

impl Add for Var {
    type Output = Var;

    fn add(self, other: Self) -> Self::Output {
        let mut var = Self::new(self.value + other.value);
        let grad = LocalGradient {
            from: [self.id, other.id],
            grad: [1.0, 1.0],
        };
        if let Some(mut tape) = self.merge_tape(other) {
            tape.nodes.insert(var.id, grad);
            var.tape = Some(tape);
        }
        var
    }
}

impl Sub for Var {
    type Output = Var;

    fn sub(self, other: Self) -> Self::Output {
        let mut var = Self::new(self.value - other.value);
        let grad = LocalGradient {
            from: [self.id, other.id],
            grad: [1.0, -1.0],
        };
        if let Some(mut tape) = self.merge_tape(other) {
            tape.nodes.insert(var.id, grad);
            var.tape = Some(tape);
        }
        var
    }
}

impl Mul for Var {
    type Output = Var;

    fn mul(self, other: Self) -> Self::Output {
        let mut var = Self::new(self.value * other.value);
        let grad = LocalGradient {
            from: [self.id, other.id],
            grad: [other.value, self.value],
        };
        if let Some(mut tape) = self.merge_tape(other) {
            tape.nodes.insert(var.id, grad);
            var.tape = Some(tape);
        }
        var
    }
}

impl Div for Var {
    type Output = Var;

    fn div(self, other: Self) -> Self::Output {
        let mut var = Self::new(self.value / other.value);
        let grad = LocalGradient {
            from: [self.id, other.id],
            grad: [1.0 / other.value, -self.value / (other.value * other.value)],
        };
        if let Some(mut tape) = self.merge_tape(other) {
            tape.nodes.insert(var.id, grad);
            var.tape = Some(tape);
        }
        var
    }
}

/// A neuron holding a set of weights and a bias.
#[derive(Debug)]
struct Neuron {
    bias: Var,
    weights: Vec<Var>,
    nonlinearity: fn(Var) -> Var,
}

impl Neuron {
    /// Create a new f32 neuron with randomized weights and bias.
    pub fn rand<R, D>(
        rng: &mut R,
        distribution: &D,
        nonlinearity: fn(Var) -> Var,
        input_size: usize,
    ) -> Self
    where
        R: rand::Rng,
        D: Distribution<f64> + Copy,
    {
        Self {
            bias: Var::new(rng.sample(distribution)),
            weights: (0..input_size)
                .map(|_| Var::new(rng.sample(distribution)))
                .collect(),
            nonlinearity,
        }
    }

    /// Returns a gradient-traced version of the neuron.
    fn trace(&self) -> Self {
        Self {
            bias: self.bias.trace(),
            weights: self.weights.iter().map(|w| w.trace()).collect(),
            nonlinearity: self.nonlinearity,
        }
    }

    /// Applies the neuron to the given input.
    pub fn forward(&self, input: &[Var]) -> Var {
        assert_eq!(input.len(), self.weights.len());
        (self.nonlinearity)(
            self.weights
                .iter()
                .zip(input)
                .fold(self.bias.clone(), |acc, (w, x)| acc + w.clone() * x.clone()),
        )
    }

    fn parameters_mut(&mut self) -> Vec<&mut Var> {
        self.weights
            .iter_mut()
            .chain(iter::once(&mut self.bias))
            .collect()
    }
}

/// A layer of neurons.
#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    /// Create a new f32 layer with randomized neurons.
    pub fn rand<R, D>(
        rng: &mut R,
        distribution: &D,
        nonlinearity: fn(Var) -> Var,
        input_size: usize,
        output_size: usize,
    ) -> Self
    where
        R: rand::Rng,
        D: Distribution<f64> + Copy,
    {
        Self {
            neurons: (0..output_size)
                .map(|_| Neuron::rand(rng, distribution, nonlinearity, input_size))
                .collect(),
        }
    }

    /// Returns a gradient-traced version of the layer.
    pub fn trace(&self) -> Self {
        Self {
            neurons: self.neurons.iter().map(|n| n.trace()).collect(),
        }
    }

    /// Applies the layer to the given input.
    pub fn forward(&self, input: &[Var]) -> Vec<Var> {
        self.neurons
            .iter()
            .map(|neuron| neuron.forward(input))
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Var> {
        self.neurons
            .iter_mut()
            .flat_map(|n| n.parameters_mut())
            .collect()
    }
}

/// A multi-layer perceptron holding a set of layers.
pub struct Mlp {
    layers: Vec<Layer>,
}

impl Mlp {
    /// Create a new f32 MLP with the given list of layers.
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    /// Returns a gradient-traced version of the MLP.
    pub fn trace(&self) -> Self {
        Self {
            layers: self.layers.iter().map(|l| l.trace()).collect(),
        }
    }

    /// Applies the MLP to the given input.
    pub fn forward(&self, input: &[Var]) -> Vec<Var> {
        match self.layers.split_first() {
            Some((layer, ls)) => ls
                .iter()
                .fold(layer.forward(input), |acc, layer| layer.forward(&acc)),
            None => Vec::new(),
        }
    }

    /// Train the MLP on the given dataset.
    #[allow(clippy::too_many_arguments)]
    pub fn train<R, const M: usize, const N: usize>(
        &mut self,
        rng: &mut R,
        dataset: &[Sample<M, N>],
        epochs: usize,
        batch_size: usize,
        learning_rate: f64,
        print_interval: usize,
    ) where
        R: Rng,
    {
        let start = time::Instant::now();
        for epoch in 0..epochs {
            let mut dataset = Vec::from(dataset);
            dataset.shuffle(rng);
            for batch in dataset.chunks(batch_size) {
                let loss = self.loss_dataset(batch, mse);
                let gradients = loss.gradients().unwrap();
                for param in self.parameters_mut() {
                    let grad = gradients.get(&param.id).unwrap();
                    param.value -= grad * learning_rate;
                }
            }
            if print_interval != 0 && (epoch + 1) % print_interval == 0 {
                let loss = self.loss_dataset(&dataset, mse);
                println!("epoch: {}, loss: {}", epoch + 1, loss.value);
            }
        }
        let duration = time::Instant::now() - start;
        println!("training took {}ms", duration.as_millis());
    }

    fn loss_dataset<const M: usize, const N: usize>(
        &self,
        samples: &[Sample<M, N>],
        metric: fn(Vec<Var>, Vec<Var>) -> Var,
    ) -> Var {
        let mut loss = Var::new(0.0);
        for sample in samples {
            loss = loss + self.loss_sample(sample, metric);
        }
        loss / Var::new(samples.len() as f64)
    }

    fn loss_sample<const M: usize, const N: usize>(
        &self,
        sample: &Sample<M, N>,
        metric: fn(Vec<Var>, Vec<Var>) -> Var,
    ) -> Var {
        let input: Vec<_> = sample.input.iter().map(|x| Var::new(*x)).collect();
        let output: Vec<_> = sample.output.iter().map(|x| Var::new(*x)).collect();
        let pred = self.forward(&input);
        (metric)(pred, output)
    }

    fn parameters_mut(&mut self) -> Vec<&mut Var> {
        self.layers
            .iter_mut()
            .flat_map(|l| l.parameters_mut())
            .collect()
    }
}

fn mse(pred: Vec<Var>, output: Vec<Var>) -> Var {
    let mut loss = Var::new(0.0);
    let n = Var::new(output.len() as f64);
    for (p, o) in pred.into_iter().zip(output.into_iter()) {
        let diff = p - o;
        loss = loss + diff.clone() * diff;
    }
    loss / n
}
