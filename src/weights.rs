use half::f16;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::collections::HashSet;

use crate::{
    Result,
    error::invalid,
    tensor::{Matrix, Vector},
};

pub(crate) struct Weights<'a> {
    tensors: SafeTensors<'a>,
}

impl<'a> Weights<'a> {
    pub fn new(bytes: &'a [u8]) -> Result<Self> {
        Ok(Self {
            tensors: SafeTensors::deserialize(bytes)?,
        })
    }

    pub fn floats(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let tensor = self.tensors.tensor(name)?;
        if tensor.shape() != shape {
            return Err(invalid(format!(
                "{name}: expected {shape:?}, found {:?}",
                tensor.shape()
            )));
        }
        let values = match tensor.dtype() {
            Dtype::F32 => tensor
                .data()
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            Dtype::F16 => tensor
                .data()
                .chunks_exact(2)
                .map(|b| f16::from_bits(u16::from_le_bytes([b[0], b[1]])).to_f32())
                .collect(),
            dtype => return Err(invalid(format!("{name}: unsupported dtype {dtype:?}"))),
        };
        Ok(values)
    }

    pub fn vector(&self, name: &str, length: usize) -> Result<Vector> {
        Ok(Vector::from_vec(self.floats(name, &[length])?))
    }

    pub fn matrix(&self, name: &str, rows: usize, columns: usize) -> Result<Matrix> {
        Ok(Matrix::from_shape_vec(
            (rows, columns),
            self.floats(name, &[rows, columns])?,
        )?)
    }

    pub fn bytes(&self, name: &str, shape: &[usize]) -> Result<Vec<u8>> {
        let tensor = self.tensors.tensor(name)?;
        if tensor.dtype() != Dtype::U8 || tensor.shape() != shape {
            return Err(invalid(format!("{name}: expected U8 {shape:?}")));
        }
        if tensor.data().iter().any(|&byte| byte >= 243) {
            return Err(invalid(format!("{name}: invalid base-3 packed byte")));
        }
        Ok(tensor.data().to_vec())
    }
}

#[derive(Deserialize)]
struct Manifest {
    format: String,
    names: String,
    quant: Quantization,
    packing: Packing,
    quantized_modules: Vec<QuantizedModule>,
}

#[derive(Deserialize)]
struct Quantization {
    mode: String,
    group_size: usize,
}
#[derive(Deserialize)]
struct Packing {
    base: usize,
    code_offset: usize,
    elements_per_byte: usize,
    row_major: bool,
}
#[derive(Deserialize)]
struct QuantizedModule {
    name: String,
    in_features: usize,
    out_features: usize,
    group_size: usize,
}

pub(crate) fn validate_manifest(
    bytes: &[u8],
    group_size: usize,
    weights: &Weights<'_>,
) -> Result<usize> {
    let manifest: Manifest = serde_json::from_slice(bytes)?;
    if manifest.format != "thrush-ternary-v2"
        || manifest.names != "hf"
        || manifest.quant.mode != "ternary"
        || manifest.quant.group_size != group_size
        || manifest.packing.base != 3
        || manifest.packing.code_offset != 1
        || manifest.packing.elements_per_byte != 5
        || !manifest.packing.row_major
    {
        return Err(invalid("unsupported ternary manifest"));
    }
    let mut names = HashSet::new();
    for module in &manifest.quantized_modules {
        if !names.insert(&module.name) {
            return Err(invalid(format!("duplicate ternary module {}", module.name)));
        }
        if module.group_size != group_size || module.in_features == 0 || module.out_features == 0 {
            return Err(invalid(format!("invalid quantization for {}", module.name)));
        }
        let tensor = weights
            .tensors
            .tensor(&format!("{}.qweight", module.name))?;
        if tensor.shape() != [module.out_features, module.in_features.div_ceil(5)] {
            return Err(invalid(format!(
                "packed shape mismatch for {}",
                module.name
            )));
        }
    }
    Ok(manifest.quantized_modules.len())
}
