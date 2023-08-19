use candle::{
    reverse::{train_eval, Layer, Mlp, Tape, Variable},
    Sample,
};
use rand::Rng;
use rand_distr::StandardNormal;

fn main() {
    let mut rng = rand::thread_rng();
    let distribution = StandardNormal;
    let tape = Tape::default();
    let mut mlp = Mlp::new(vec![Layer::new(
        2,
        1,
        Variable::identity,
        &mut rng,
        distribution,
    )]);
    let dataset = dataset_add(1000);
    train_eval(&tape, &mut mlp, &mut rng, &dataset, 100, 100, 0.1, 20);
}

fn dataset_add(count: usize) -> Vec<Sample<2, 1>> {
    let dist = StandardNormal;
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
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
