use ndarray::{Array1, Array2};

pub(crate) type Matrix = Array2<f32>;
pub(crate) type Vector = Array1<f32>;

pub(crate) fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

pub(crate) fn silu(values: &mut Matrix) {
    values.mapv_inplace(|value| value * sigmoid(value));
}

pub(crate) fn softmax(values: &mut [f32]) {
    let maximum = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0;
    for value in values.iter_mut() {
        *value = (*value - maximum).exp();
        sum += *value;
    }
    for value in values {
        *value /= sum;
    }
}

pub(crate) fn argmax(values: impl Iterator<Item = f32>) -> usize {
    let mut best_index = 0;
    let mut best_value = f32::NEG_INFINITY;
    for (index, value) in values.enumerate() {
        if value > best_value {
            best_index = index;
            best_value = value;
        }
    }
    best_index
}
