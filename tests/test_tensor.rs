use candle::tensor::{layout::TensorLayout, Tensor};

#[test]
fn test_tensor_iterator() {
    let data = [2f32, 3f32, 4f32];
    let t = Tensor::from(data);
    for (x, y) in t.into_iter().zip(&data) {
        assert_eq!(*x, *y);
    }

    let data = (0..24).map(|x| x as f32).collect::<Vec<_>>();
    let t = Tensor::new(&data, TensorLayout::from(&[2, 3, 4])).unwrap();
    for (x, y) in t.into_iter().zip(&data) {
        assert_eq!(*x, *y);
    }
}
