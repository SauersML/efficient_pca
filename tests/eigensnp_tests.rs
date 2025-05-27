// In tests/eigensnp_tests.rs

use efficient_pca::eigensnp::{reorder_array_owned, reorder_columns_owned};

use ndarray::{Array1, Array2, arr2};

#[test]
fn test_reorder_array_basic() {
    let original = Array1::from(vec![10, 20, 30, 40]);
    let order = vec![2, 0, 3, 1];
    let expected = Array1::from(vec![30, 10, 40, 20]);
    let reordered = reorder_array_owned(&original, &order);
    assert_eq!(reordered, expected);
}

#[test]
fn test_reorder_array_empty_array_empty_order() {
    let original = Array1::<i32>::from(vec![]);
    let order_empty = vec![];
    let expected = Array1::<i32>::from(vec![]);
    let reordered_empty_order = reorder_array_owned(&original, &order_empty);
    assert_eq!(reordered_empty_order, expected);
}

#[test]
#[should_panic]
fn test_reorder_array_select_from_zero_elements_panics() {
    let original = Array1::<i32>::from(vec![]);
    let order = vec![0];
    reorder_array_owned(&original, &order);
}

#[test]
fn test_reorder_array_empty_order() {
    let original = Array1::from(vec![10, 20, 30]);
    let order = vec![];
    let expected = Array1::<i32>::from(vec![]);
    let reordered = reorder_array_owned(&original, &order);
    assert_eq!(reordered, expected);
}

#[test]
fn test_reorder_array_repeated_indices() {
    let original = Array1::from(vec![10, 20, 30]);
    let order = vec![0, 1, 0, 2, 1, 1];
    let expected = Array1::from(vec![10, 20, 10, 30, 20, 20]);
    let reordered = reorder_array_owned(&original, &order);
    assert_eq!(reordered, expected);
}

#[test]
fn test_reorder_columns_basic() {
    let original = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let order = vec![2, 0, 1];
    let expected = arr2(&[[3, 1, 2], [6, 4, 5]]);
    let reordered = reorder_columns_owned(&original, &order);
    assert_eq!(reordered, expected);
}

#[test]
fn test_reorder_columns_empty_matrix_variants() {
    let original_0_rows = Array2::<i32>::zeros((0, 3));
    let order = vec![1, 0, 2];
    let expected_0_rows = Array2::<i32>::zeros((0, 3));
    let reordered_0_rows = reorder_columns_owned(&original_0_rows, &order);
    assert_eq!(reordered_0_rows, expected_0_rows);

    let original_0_cols = Array2::<i32>::zeros((2, 0));
    let order_empty = vec![];
    let expected_0_cols_empty_order = Array2::<i32>::zeros((2,0));
    let reordered_empty_order = reorder_columns_owned(&original_0_cols, &order_empty);
    assert_eq!(reordered_empty_order, expected_0_cols_empty_order);
}

#[test]
#[should_panic]
fn test_reorder_columns_select_from_zero_cols_panics() {
    let original_0_cols = Array2::<i32>::zeros((2, 0));
    let order_for_0_cols = vec![0];
    reorder_columns_owned(&original_0_cols, &order_for_0_cols);
}
   
#[test]
fn test_reorder_columns_empty_order() {
    let original = arr2(&[[1, 2, 3], [4, 5, 6]]);
    let order = vec![];
    let expected = Array2::<i32>::zeros((2, 0));
    let reordered = reorder_columns_owned(&original, &order);
    assert_eq!(reordered, expected);
}

#[test]
fn test_reorder_columns_repeated_indices() {
    let original = arr2(&[[1, 2], [3, 4]]);
    let order = vec![0, 1, 0, 0];
    let expected = arr2(&[[1, 2, 1, 1], [3, 4, 3, 3]]);
    let reordered = reorder_columns_owned(&original, &order);
    assert_eq!(reordered, expected);
}
