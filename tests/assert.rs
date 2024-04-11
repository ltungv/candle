use candle::tensor::{Layout, Tensor};

pub fn assert_contiguous_layout(layout: &Layout, expected_shape: &[usize]) {
    let shape = layout.shape();
    let strides = layout.strides();
    assert_eq!(shape, expected_shape);
    assert_eq!(shape.len(), strides.len());
    if !shape.is_empty() {
        for dim in (0..shape.len() - 1).rev() {
            assert_eq!(strides[dim], strides[dim + 1] * shape[dim + 1]);
        }
    }
}

pub fn assert_scalar_layout(layout: &Layout) {
    assert!(layout.shape().is_empty());
    assert!(layout.strides().is_empty());
}

pub fn assert_reduced_layout(reduced: &Layout, reducer: &Layout, dims: &[usize], shape: &[usize]) {
    assert_contiguous_layout(reduced, shape);
    let mut strides = reduced.strides().to_vec();
    for d in dims {
        strides[*d] = 0;
    }
    assert_eq!(reducer.shape(), shape);
    assert_eq!(reducer.strides(), strides);
}

pub fn assert_expanded_layout(expanded: &Layout, original: &Layout, shape: &[usize]) {
    let mut strides = vec![0; shape.len()];
    let original_shape = original.shape();
    let original_strides = original.strides();
    for i in 0..original.shape().len() {
        let d1 = original_shape.len() - i - 1;
        let d2 = shape.len() - i - 1;
        if shape[d2] == original_shape[d1] {
            strides[d2] = original_strides[d1];
        }
    }
    assert_eq!(expanded.shape(), shape);
    assert_eq!(expanded.strides(), strides);
}

pub fn assert_contiguous_tensor(tensor: &Tensor, data: &[f32], shape: &[usize]) {
    assert_contiguous_layout(tensor.layout(), shape);
    let data_collected: Vec<_> = tensor.into_iter().collect();
    assert_eq!(data_collected, data);
}

pub fn assert_scalar_tensor(tensor: &Tensor, data: f32) {
    assert_scalar_layout(tensor.layout());
    let data_collected: Vec<_> = tensor.into_iter().collect();
    assert_eq!(data_collected.len(), 1);
    assert_eq!(data_collected[0], data);
}
