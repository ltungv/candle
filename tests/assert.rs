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
    let mut strides = Vec::with_capacity(shape.len());
    for ((d1, d2), stride) in original
        .shape()
        .iter()
        .rev()
        .zip(shape.iter().rev())
        .zip(original.strides().iter().rev())
    {
        if d1 == d2 {
            strides.push(*stride);
        } else {
            strides.push(0);
        }
    }
    while shape.len() > strides.len() {
        strides.push(0);
    }
    strides.reverse();
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
