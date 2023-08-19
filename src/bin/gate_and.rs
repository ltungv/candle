use candle::{
    reverse::{train_eval, Layer, Mlp, Tape, Variable},
    Sample,
};
use rand_distr::StandardNormal;

fn main() {
    let mut rng = rand::thread_rng();
    let distribution = StandardNormal;
    let tape = Tape::default();
    let mut mlp = Mlp::new(vec![Layer::new(
        &tape,
        2,
        1,
        Variable::sigmoid,
        &mut rng,
        distribution,
    )]);
    println!("{} parameters", mlp.parameters().len());

    let dataset = vec![
        Sample {
            input: [0.0, 0.0],
            output: [0.0],
        },
        Sample {
            input: [0.0, 1.0],
            output: [0.0],
        },
        Sample {
            input: [1.0, 0.0],
            output: [0.0],
        },
        Sample {
            input: [1.0, 1.0],
            output: [1.0],
        },
    ];

    train_eval(&tape, &mut mlp, &mut rng, &dataset, 100000, 4, 0.1, 20000);
}
