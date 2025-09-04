use candle::{
    autodiff::reverse::{grad, gradn},
    tensor::{cpu, ops::ML, shape, typ::Float, Tensor},
};
use rand::{rngs::StdRng, Rng, SeedableRng};

type T32 = Tensor<cpu::Tensor<f32>, f32, cpu::TensorOps>;

fn main() {
    let two = T32::scalar(2.0);

    let df = grad(&two, |t| t.tanh());
    println!("df={:?}", df);

    let ddf = grad(&two, |t| grad(&t, |t| t.tanh()));
    println!("ddf={:?}", ddf);

    let dddf = grad(&two, |t| grad(&t, |t| grad(&t, |t| t.tanh())));
    println!("dddf={:?}", dddf);

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

    let key = 0;
    let mut rng = StdRng::seed_from_u64(key);

    let ws: Vec<f32> = (&mut rng).random_iter().take(3).collect();
    let w = T32::new(&shape([3]), &ws);

    let bs: Vec<f32> = (&mut rng).random_iter().take(1).collect();
    let b = T32::new(&shape([1]), &bs);

    let prediction = predict(&w, &b, &inputs);
    println!("prediction: {:?}", prediction);

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
    let l = loss(&w, &b, &inputs, &targets);
    println!("loss: {:?}", l);

    // Differentiate loss wrt weights
    let w_grad = grad(&w, |w| loss(&w, &b.lift(), &inputs.lift(), &targets.lift()));
    println!("w_grad: {:?}", w_grad);

    // Differentiate loss wrt biases
    let b_grad = grad(&b, |b| loss(&w.lift(), &b, &inputs.lift(), &targets.lift()));
    println!("b_grad: {:?}", b_grad);

    // Differentiate loss wrt W and b - should give the same answer
    let wb_grads = gradn(&[&w, &b], |ts| {
        let w = &ts[0];
        let b = &ts[1];
        loss(w, b, &inputs.lift(), &targets.lift())
    });
    println!("w_grad: {:?}", wb_grads[0]);
    println!("b_grad: {:?}", wb_grads[1]);

    let new_w = &w - &w_grad;
    let new_b = &b - &b_grad;
    let new_prediction = predict(&new_w, &new_b, &inputs);
    let new_loss = loss(&new_w, &new_b, &inputs, &targets);
    println!("new_prediction: {:?}", new_prediction);
    println!("new_loss: {:?}", new_loss);

    let eps = T32::scalar(1e-4);
    let half_eps = &eps / T32::scalar(2.0);
    let loss_l = loss(&w, &(&b - &half_eps), &inputs, &targets);
    let loss_r = loss(&w, &(&b + &half_eps), &inputs, &targets);
    let b_grad_numerical = (loss_r - loss_l) / &eps;
    println!("b_grad_numerical {:?}", b_grad_numerical);
    println!("b_grad_autodiff {:?}", b_grad);
}
