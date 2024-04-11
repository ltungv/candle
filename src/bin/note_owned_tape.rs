use std::time;

use candle::experimentals::{
    dataset::VectorMapping,
    scalar::ad_backward_tape_owned::{Layer, Mlp, Var},
};
use rand::Rng;
use rand_distr::StandardNormal;

const NUM_SAMPLES: usize = 100000;

fn main() {
    let dataset = dataset_add(NUM_SAMPLES);
    let mut rng = rand::thread_rng();
    let distribution = StandardNormal;

    let mut mlp = Mlp::new(vec![Layer::rand(
        &mut rng,
        &distribution,
        Var::sigmoid,
        2,
        1,
    )]);

    let start = time::Instant::now();
    for sample in &dataset {
        let input: Vec<_> = sample.input.iter().map(|x| Var::new(*x)).collect();
        let _ = mlp.forward(&input);
    }
    let duration = time::Instant::now() - start;
    println!("--------------------------------");
    println!("Inference without gradients accumulation");
    println!("runtime {}ms", duration.as_millis());

    mlp.trace();

    let start = time::Instant::now();
    for sample in &dataset {
        let input: Vec<_> = sample.input.iter().map(|x| Var::new(*x)).collect();
        let _ = mlp.forward(&input);
    }
    let duration = time::Instant::now() - start;
    println!("--------------------------------");
    println!("Inference with gradients accumulation");
    println!("runtime {}ms", duration.as_millis());

    let start = time::Instant::now();
    for sample in &dataset {
        let input: Vec<_> = sample.input.iter().map(|x| Var::new(*x)).collect();
        let output = mlp.forward(&input);
        for x in output {
            let _ = x.gradients();
        }
    }
    let duration = time::Instant::now() - start;
    println!("--------------------------------");
    println!("Inference with gradients");
    println!("runtime {}ms", duration.as_millis());
}

fn dataset_add(count: usize) -> Vec<VectorMapping<2, 1>> {
    let dist = StandardNormal;
    let mut rng = rand::thread_rng();
    let mut samples = Vec::with_capacity(count);
    for _ in 0..count {
        let x = rng.sample(dist);
        let y = rng.sample(dist);
        let z = x + y;
        let sample = VectorMapping {
            input: [x, y],
            output: [z],
        };
        samples.push(sample);
    }
    samples
}
