use candle::tensor::Tensor;

fn main() {
    let x = Tensor::shaped(&[2], &[1.0, 2.0]).unwrap();
    let w = Tensor::shaped(&[2, 2], &[3.0, 4.0, 5.0, 6.0]).unwrap();
    let b = Tensor::shaped(&[2], &[7.0, 8.0]).unwrap();
    let z = x.matmul(&w).unwrap() + &b;
    println!("{:?}", z);

    let z = &w * &b + &w * &b;
    println!("{:?}", z);
}
