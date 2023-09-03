use std::time;

use candle::{
    dataset::Sample,
    scalar::ad_backward_tape_owned::{Layer, Mlp, Var},
};
use rand::Rng;
use rand_distr::StandardNormal;

fn main() {
    let dataset = dataset_add(100000);
    let mut rng = rand::thread_rng();
    let distribution = StandardNormal;
    {
        let mlp = Mlp::new(vec![Layer::rand(
            &mut rng,
            &distribution,
            Var::sigmoid,
            2,
            1,
        )]);
        let start = time::Instant::now();
        let mlp_traced = mlp.trace();
        for sample in &dataset {
            let input: Vec<_> = sample.input.iter().map(|x| Var::new(*x)).collect();
            let _ = mlp_traced.forward(&input);
        }
        let duration = time::Instant::now() - start;
        println!("inference took {}ms", duration.as_millis());
    }
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
