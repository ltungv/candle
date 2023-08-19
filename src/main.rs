use candle::reverse::{Layer, Mlp, Tape, Variable};
use rand::{seq::SliceRandom, Rng};
use rand_distr::{Bernoulli, StandardNormal};

#[derive(Debug)]
struct Sample<const M: usize, const N: usize> {
    input: [f64; M],
    output: [f64; N],
}

fn dataset_add() -> Vec<Sample<2, 1>> {
    let dist = StandardNormal;
    let mut rng = rand::thread_rng();
    let mut samples = Vec::new();
    for _ in 0..10000 {
        let x = rng.sample(dist);
        let y = rng.sample(dist);
        let z = x + y;
        let sample = Sample {
            input: [x, y],
            output: [z],
        };
        samples.push(sample);
    }
    samples
}

fn dataset_xor() -> Vec<Sample<2, 1>> {
    let dist = Bernoulli::new(0.5).unwrap();
    let mut rng = rand::thread_rng();
    let mut samples = Vec::new();
    let convert = |x| if x { 1.0 } else { 0.0 };
    for _ in 0..10000 {
        let x = rng.sample(dist);
        let y = rng.sample(dist);
        let z = x ^ y;
        let sample = Sample {
            input: [convert(x), convert(y)],
            output: [convert(z)],
        };
        samples.push(sample);
    }
    samples
}

fn dataset_and() -> Vec<Sample<2, 1>> {
    let dist = Bernoulli::new(0.5).unwrap();
    let mut rng = rand::thread_rng();
    let mut samples = Vec::new();
    let convert = |x| if x { 1.0 } else { 0.0 };
    for _ in 0..10000 {
        let x = rng.sample(dist);
        let y = rng.sample(dist);
        let z = x & y;
        let sample = Sample {
            input: [convert(x), convert(y)],
            output: [convert(z)],
        };
        samples.push(sample);
    }
    samples
}

fn mse<'a, const M: usize, const N: usize>(
    tape: &'a Tape,
    mlp: &Mlp<'a>,
    dataset: &[Sample<M, N>],
) -> Variable<'a> {
    let mut loss = tape.add_variable(0.0);
    for sample in dataset {
        let input: Vec<_> = sample.input.iter().map(|x| tape.add_variable(*x)).collect();
        let pred = mlp.call(&input);
        let mut loss_sample = tape.add_variable(0.0);
        for (p, o) in pred.iter().zip(sample.output) {
            let diff = p - &tape.add_variable(o);
            loss_sample = loss_sample + diff * diff;
        }
        loss_sample = loss_sample / tape.add_variable(sample.output.len() as f64);
        loss = loss + loss_sample;
    }
    loss / tape.add_variable(dataset.len() as f64)
}

fn main() {
    let mut rng = rand::thread_rng();
    let distribution = StandardNormal;
    {
        let tape = Tape::default();
        let layers = vec![Layer::new(
            &tape,
            2,
            1,
            Variable::identity,
            &mut rng,
            distribution,
        )];
        let mut mlp = Mlp::new(layers);
        println!("{} parameters", mlp.parameters().len());
        train_eval(&tape, &mut mlp, &mut rng, dataset_add(), 100, 0.1);
    }
    {
        let tape = Tape::default();
        let layers = vec![Layer::new(
            &tape,
            2,
            1,
            Variable::sigmoid,
            &mut rng,
            distribution,
        )];
        let mut mlp = Mlp::new(layers);
        println!("{} parameters", mlp.parameters().len());
        train_eval(&tape, &mut mlp, &mut rng, dataset_and(), 2000, 0.05);
    }
    {
        let tape = Tape::default();
        let layers = vec![
            Layer::new(&tape, 2, 3, Variable::relu, &mut rng, distribution),
            Layer::new(&tape, 3, 1, Variable::sigmoid, &mut rng, distribution),
        ];
        let mut mlp = Mlp::new(layers);
        println!("{} parameters", mlp.parameters().len());
        train_eval(&tape, &mut mlp, &mut rng, dataset_xor(), 2000, 0.05);
    }
}

fn train_eval<'a, R, const M: usize, const N: usize>(
    tape: &'a Tape,
    mlp: &mut Mlp<'a>,
    rng: &mut R,
    mut dataset: Vec<Sample<M, N>>,
    epochs: usize,
    learning_rate: f64,
) where
    R: Rng,
{
    // Mark the last index of the tape. At this point in time, nodes in the tape
    // includes the variables in the MLP and in the dataset.
    tape.mark();
    println!("{} tape nodes", tape.len());

    for epoch in 0..epochs {
        dataset.shuffle(rng);
        for batch in dataset.chunks(1000) {
            let loss = mse(tape, mlp, batch);
            let gradients = loss.gradients();
            for param in mlp.parameters() {
                param.learn(&gradients, learning_rate);
                param.zerograd();
            }
            // Clear all intermediate nodes in the tape (nodes whose index is greater
            // than the marked index).
            tape.clean();
        }
        if epoch % 5 == 0 {
            // Compute the loss on the entire dataset.
            let loss = mse(tape, mlp, &dataset);
            println!("Epoch: {}, Loss: {}", epoch, loss.value);
            println!("\tTape size: {}", tape.len());
            tape.clean();
            println!("\tTape size: {}", tape.len());
        }
    }

    for sample in dataset {
        let input: Vec<_> = sample.input.iter().map(|x| tape.add_variable(*x)).collect();
        let pred: Vec<_> = mlp.call(&input).iter().map(|x| x.value).collect();
        tape.clean();
        println!("Input: {:?}", sample.input);
        println!("Real: {:?}", sample.output);
        println!("Pred: {:?}", pred);
        println!("====================")
    }
}
