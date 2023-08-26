use candle::tensor::{layout::Layout, Tensor};

#[test]
fn test_tensor_iterator() {
    let data = [2f32, 3f32, 4f32];
    let t = Tensor::from(data);
    for (x, y) in t.into_iter().zip(&data) {
        assert_eq!(*x, *y);
    }

    let data = [
        0f32, 1f32, 2f32, 3f32, 
        4f32, 5f32, 6f32, 7f32, 
        8f32, 9f32, 10f32, 11f32, 
        12f32, 13f32, 14f32, 15f32, 
        16f32, 17f32, 18f32, 19f32, 
        20f32, 21f32, 22f32, 23f32,
    ];
    let t = Tensor::new(&data, Layout::from(&[2, 3, 4])).unwrap();
    for (x, y) in t.into_iter().zip(&data) {
        assert_eq!(*x, *y);
    }
}
