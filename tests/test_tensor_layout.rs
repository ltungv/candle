use candle::tensor::layout::TensorLayout;

#[test]
fn test_layout_create() {
    let data = [2, 3, 4];
    let assert_layout = |layout: &TensorLayout| {
        assert_eq!(layout.elems(), 24);
        assert_eq!(layout.shape(), &[2, 3, 4]);
        assert_eq!(layout.strides(), &[12, 4, 1]);
    };

    let layout = TensorLayout::from(data.as_slice());
    assert_layout(&layout);

    let layout = TensorLayout::from(data.to_vec());
    assert_layout(&layout);

    let layout = TensorLayout::from(&data);
    assert_layout(&layout);

    let layout = TensorLayout::from(data);
    assert_layout(&layout);
}

#[test]
fn test_layout_reduce() {
    let layout = TensorLayout::from(&[2, 3, 4]);

    let (reduced, reducer) = layout.reduce(&[0]).unwrap();
    assert_eq!(reduced.shape(), &[1, 3, 4]);
    assert_eq!(reduced.strides(), &[12, 4, 1]);
    assert_eq!(reducer.shape(), &[1, 3, 4]);
    assert_eq!(reducer.strides(), &[0, 4, 1]);

    let (reduced, reducer) = layout.reduce(&[1]).unwrap();
    assert_eq!(reduced.shape(), &[2, 1, 4]);
    assert_eq!(reduced.strides(), &[4, 4, 1]);
    assert_eq!(reducer.shape(), &[2, 1, 4]);
    assert_eq!(reducer.strides(), &[4, 0, 1]);

    let (reduced, reducer) = layout.reduce(&[2]).unwrap();
    assert_eq!(reduced.shape(), &[2, 3, 1]);
    assert_eq!(reduced.strides(), &[3, 1, 1]);
    assert_eq!(reducer.shape(), &[2, 3, 1]);
    assert_eq!(reducer.strides(), &[3, 1, 0]);

    let (reduced, reducer) = layout.reduce(&[0, 1]).unwrap();
    assert_eq!(reduced.shape(), &[1, 1, 4]);
    assert_eq!(reduced.strides(), &[4, 4, 1]);
    assert_eq!(reducer.shape(), &[1, 1, 4]);
    assert_eq!(reducer.strides(), &[0, 0, 1]);

    let (reduced, reducer) = layout.reduce(&[0, 2]).unwrap();
    assert_eq!(reduced.shape(), &[1, 3, 1]);
    assert_eq!(reduced.strides(), &[3, 1, 1]);
    assert_eq!(reducer.shape(), &[1, 3, 1]);
    assert_eq!(reducer.strides(), &[0, 1, 0]);

    let (reduced, reducer) = layout.reduce(&[0, 1, 2]).unwrap();
    assert_eq!(reduced.shape(), &[1, 1, 1]);
    assert_eq!(reduced.strides(), &[1, 1, 1]);
    assert_eq!(reducer.shape(), &[1, 1, 1]);
    assert_eq!(reducer.strides(), &[0, 0, 0]);
}

#[test]
fn test_layout_squeeze() {
    let layout = TensorLayout::from(&[1, 2, 3]);
    let squeezed = layout.squeeze();
    assert_eq!(squeezed.shape(), &[2, 3]);
    assert_eq!(squeezed.strides(), &[3, 1]);

    let layout = TensorLayout::from(&[2, 1, 3]);
    let squeezed = layout.squeeze();
    assert_eq!(squeezed.shape(), &[2, 3]);
    assert_eq!(squeezed.strides(), &[3, 1]);

    let layout = TensorLayout::from(&[2, 3, 1]);
    let squeezed = layout.squeeze();
    assert_eq!(squeezed.shape(), &[2, 3]);
    assert_eq!(squeezed.strides(), &[3, 1]);
}

#[test]
fn test_layout_transpose() {
    let layout = TensorLayout::from(&[2, 3, 4]);

    let transposed = layout.transpose(0, 1).unwrap();
    assert_eq!(transposed.shape(), &[3, 2, 4]);
    assert_eq!(transposed.strides(), &[4, 12, 1]);

    let transposed = layout.transpose(1, 0).unwrap();
    assert_eq!(transposed.shape(), &[3, 2, 4]);
    assert_eq!(transposed.strides(), &[4, 12, 1]);

    let transposed = layout.transpose(0, 2).unwrap();
    assert_eq!(transposed.shape(), &[4, 3, 2]);
    assert_eq!(transposed.strides(), &[1, 4, 12]);

    let transposed = layout.transpose(2, 0).unwrap();
    assert_eq!(transposed.shape(), &[4, 3, 2]);
    assert_eq!(transposed.strides(), &[1, 4, 12]);

    let transposed = layout.transpose(1, 2).unwrap();
    assert_eq!(transposed.shape(), &[2, 4, 3]);
    assert_eq!(transposed.strides(), &[12, 1, 4]);

    let transposed = layout.transpose(2, 1).unwrap();
    assert_eq!(transposed.shape(), &[2, 4, 3]);
    assert_eq!(transposed.strides(), &[12, 1, 4]);
}

#[test]
fn test_layout_permute() {
    let layout = TensorLayout::from(&[2, 3, 4]);

    let permuted = layout.permute(&[0, 1, 2]).unwrap();
    assert_eq!(permuted.shape(), &[2, 3, 4]);
    assert_eq!(permuted.strides(), &[12, 4, 1]);

    let permuted = layout.permute(&[0, 2, 1]).unwrap();
    assert_eq!(permuted.shape(), &[2, 4, 3]);
    assert_eq!(permuted.strides(), &[12, 1, 4]);

    let permuted = layout.permute(&[1, 0, 2]).unwrap();
    assert_eq!(permuted.shape(), &[3, 2, 4]);
    assert_eq!(permuted.strides(), &[4, 12, 1]);

    let permuted = layout.permute(&[1, 2, 0]).unwrap();
    assert_eq!(permuted.shape(), &[3, 4, 2]);
    assert_eq!(permuted.strides(), &[4, 1, 12]);

    let permuted = layout.permute(&[2, 0, 1]).unwrap();
    assert_eq!(permuted.shape(), &[4, 2, 3]);
    assert_eq!(permuted.strides(), &[1, 12, 4]);

    let permuted = layout.permute(&[2, 1, 0]).unwrap();
    assert_eq!(permuted.shape(), &[4, 3, 2]);
    assert_eq!(permuted.strides(), &[1, 4, 12]);
}

#[test]
fn test_layout_expand() {
    let layout = TensorLayout::from(&[1]);
    let expanded = layout.expand(&[3]).unwrap();
    assert_eq!(expanded.shape(), &[3]);
    assert_eq!(expanded.strides(), &[0]);

    let layout = TensorLayout::from(&[1]);
    let expanded = layout.expand(&[3, 2]).unwrap();
    assert_eq!(expanded.shape(), &[3, 2]);
    assert_eq!(expanded.strides(), &[0, 0]);

    let layout = TensorLayout::from(&[2, 1, 1]);
    let expanded = layout.expand(&[7, 2, 4, 5]).unwrap();
    assert_eq!(expanded.shape(), &[7, 2, 4, 5]);
    assert_eq!(expanded.strides(), &[0, 1, 0, 0]);

    let layout = TensorLayout::from(&[1, 1, 2]);
    let expanded = layout.expand(&[4, 3, 2]).unwrap();
    assert_eq!(expanded.shape(), &[4, 3, 2]);
    assert_eq!(expanded.strides(), &[0, 0, 1]);
}

#[test]
fn test_broadcast_shape() {
    let (t1, t2) = TensorLayout::from(&[1])
        .broadcast(&TensorLayout::from(&[3]))
        .unwrap();
    assert_eq!(t1.shape(), &[3]);
    assert_eq!(t1.strides(), &[0]);
    assert_eq!(t2.shape(), &[3]);
    assert_eq!(t2.strides(), &[1]);

    let (t1, t2) = TensorLayout::from(&[3])
        .broadcast(&TensorLayout::from(&[1]))
        .unwrap();
    assert_eq!(t1.shape(), &[3]);
    assert_eq!(t1.strides(), &[1]);
    assert_eq!(t2.shape(), &[3]);
    assert_eq!(t2.strides(), &[0]);

    let (t1, t2) = TensorLayout::from(&[2, 3])
        .broadcast(&TensorLayout::from(&[1]))
        .unwrap();
    assert_eq!(t1.shape(), &[2, 3]);
    assert_eq!(t1.strides(), &[3, 1]);
    assert_eq!(t2.shape(), &[2, 3]);
    assert_eq!(t2.strides(), &[0, 0]);

    let (t1, t2) = TensorLayout::from(&[1])
        .broadcast(&TensorLayout::from(&[3, 2]))
        .unwrap();
    assert_eq!(t1.shape(), &[3, 2]);
    assert_eq!(t1.strides(), &[0, 0]);
    assert_eq!(t2.shape(), &[3, 2]);
    assert_eq!(t2.strides(), &[2, 1]);

    let (t1, t2) = TensorLayout::from(&[2, 1, 4])
        .broadcast(&TensorLayout::from(&[7, 2, 4, 1]))
        .unwrap();
    assert_eq!(t1.shape(), &[7, 2, 4, 4]);
    assert_eq!(t1.strides(), &[0, 4, 0, 1]);
    assert_eq!(t2.shape(), &[7, 2, 4, 4]);
    assert_eq!(t2.strides(), &[8, 4, 1, 0]);

    let (t1, t2) = TensorLayout::from(&[1, 4, 1, 2])
        .broadcast(&TensorLayout::from(&[1, 3, 1]))
        .unwrap();
    assert_eq!(t1.shape(), &[1, 4, 3, 2]);
    assert_eq!(t1.strides(), &[8, 2, 0, 1]);
    assert_eq!(t2.shape(), &[1, 4, 3, 2]);
    assert_eq!(t2.strides(), &[0, 0, 1, 0]);
}

#[test]
fn test_layout_reshape() {
    let layout = TensorLayout::from(&[2, 3, 4]);

    let l = layout.reshape(&[6, 4]).unwrap().unwrap();
    assert_eq!(l.shape(), &[6, 4]);
    assert_eq!(l.strides(), &[4, 1]);

    let l = layout.reshape(&[2, 12]).unwrap().unwrap();
    assert_eq!(l.shape(), &[2, 12]);
    assert_eq!(l.strides(), &[12, 1]);

    let l = layout.reshape(&[1, 6, 4]).unwrap().unwrap();
    assert_eq!(l.shape(), &[1, 6, 4]);
    assert_eq!(l.strides(), &[24, 4, 1]);

    let l = layout.reshape(&[2, 6, 2]).unwrap().unwrap();
    assert_eq!(l.shape(), &[2, 6, 2]);
    assert_eq!(l.strides(), &[12, 2, 1]);

    let l = layout.reshape(&[2, 3, 2, 2]).unwrap().unwrap();
    assert_eq!(l.shape(), &[2, 3, 2, 2]);
    assert_eq!(l.strides(), &[12, 4, 2, 1]);
}

#[test]
fn test_layout_index_to_position() {
    let layout = TensorLayout::from(&[2, 2, 2]);
    let indices: Vec<_> = (0..2)
        .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| vec![x, y, z])))
        .collect();
    for (exp, idx) in indices.iter().enumerate() {
        let pos = layout.index_to_position(idx);
        assert_eq!(pos, exp);
    }
}

#[test]
fn test_layout_position_to_index() {
    let layout = TensorLayout::from(&[2, 2, 2]);
    let indices: Vec<_> = (0..2)
        .flat_map(|x| (0..2).flat_map(move |y| (0..2).map(move |z| vec![x, y, z])))
        .collect();
    for (pos, exp) in indices.iter().enumerate() {
        let idx = layout.position_to_index(pos);
        assert_eq!(idx.as_slice(), exp.as_slice());
    }
}

#[test]
fn test_layout_index_iterator() {
    let layout = TensorLayout::from(&[2, 3, 4]);
    let indices: Vec<_> = (0..2)
        .flat_map(|x| (0..3).flat_map(move |y| (0..4).map(move |z| vec![x, y, z])))
        .collect();
    for (i, idx) in layout.iter_index().enumerate() {
        assert_eq!(idx, indices[i]);
    }
}

#[test]
fn test_layout_position_iterator() {
    let layout = TensorLayout::from(&[2, 3, 4]);
    for (i, out) in layout.iter_position().enumerate() {
        assert_eq!(out, i);
    }
}
