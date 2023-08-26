use candle::tensor::layout::Layout;

#[test]
fn test_layout_create() {
    let l = Layout::new(&[2, 3, 4]);
    assert_eq!(l.shape(), &[2, 3, 4]);
    assert_eq!(l.strides(), &[12, 4, 1]);
    assert_eq!(l.elems(), 24);
}

#[test]
fn test_layout_index_to_position() {
    let l = Layout::new(&[2, 2, 2]);

    let pos = l.index_to_position(&[0, 0, 0]);
    assert_eq!(pos, 0);

    let pos = l.index_to_position(&[0, 0, 1]);
    assert_eq!(pos, 1);

    let pos = l.index_to_position(&[0, 1, 0]);
    assert_eq!(pos, 2);

    let pos = l.index_to_position(&[0, 1, 1]);
    assert_eq!(pos, 3);

    let pos = l.index_to_position(&[1, 0, 0]);
    assert_eq!(pos, 4);

    let pos = l.index_to_position(&[1, 0, 1]);
    assert_eq!(pos, 5);

    let pos = l.index_to_position(&[1, 1, 0]);
    assert_eq!(pos, 6);

    let pos = l.index_to_position(&[1, 1, 1]);
    assert_eq!(pos, 7);
}

#[test]
fn test_layout_position_to_index() {
    let l = Layout::new(&[2, 2, 2]);

    let pos = l.position_to_index(0);
    assert_eq!(pos, &[0, 0, 0]);

    let pos = l.position_to_index(1);
    assert_eq!(pos, &[0, 0, 1]);

    let pos = l.position_to_index(2);
    assert_eq!(pos, &[0, 1, 0]);

    let pos = l.position_to_index(3);
    assert_eq!(pos, &[0, 1, 1]);

    let pos = l.position_to_index(4);
    assert_eq!(pos, &[1, 0, 0]);

    let pos = l.position_to_index(5);
    assert_eq!(pos, &[1, 0, 1]);

    let pos = l.position_to_index(6);
    assert_eq!(pos, &[1, 1, 0]);

    let pos = l.position_to_index(7);
    assert_eq!(pos, &[1, 1, 1]);
}

#[test]
fn test_layout_expand() {
    let l1 = Layout::new(&[1]);
    let l2 = l1.expand(&[3]).unwrap();
    assert_eq!(l2, Layout::new(&[3]));

    let l1 = Layout::new(&[3]);
    let l2 = l1.expand(&[1]).unwrap();
    assert_eq!(l2, Layout::new(&[3]));

    let l1 = Layout::new(&[2, 3]);
    let l2 = l1.expand(&[1]).unwrap();
    assert_eq!(l2, Layout::new(&[2, 3]));

    let l1 = Layout::new(&[1]);
    let l2 = l1.expand(&[3, 2]).unwrap();
    assert_eq!(l2, Layout::new(&[3, 2]));

    let l1 = Layout::new(&[3, 1, 2, 1]);
    let l2 = l1.expand(&[1, 5]).unwrap();
    assert_eq!(l2, Layout::new(&[3, 1, 2, 5]));

    let l1 = Layout::new(&[2, 3, 1]);
    let l2 = l1.expand(&[7, 2, 1, 5]).unwrap();
    assert_eq!(l2, Layout::new(&[7, 2, 3, 5]));
}
