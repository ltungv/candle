use candle::tensor::Tensor;

fn main() {
    let x = Tensor::shaped(&[2], &[1.0, 2.0]).unwrap();
    let w = Tensor::shaped(&[2, 2], &[3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::shaped(&[2], &[7.0, 8.0]).unwrap();
    let z = x.matmul(&w) + &b;
    println!("{:?}", z.unwrap());

    let z = (&w * &b).and_then(|x| x + &w * &b);
    println!("{:?}", z.unwrap());
}
