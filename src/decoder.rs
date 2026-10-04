use ndarray::Axis;

use crate::{
    ModelConfig, Result,
    error::invalid,
    layers::Linear,
    tensor::{Matrix, argmax, sigmoid},
    weights::Weights,
};

struct LstmLayer {
    input_projection: Linear,
    recurrent_projection: Linear,
}

struct LstmState {
    hidden: Matrix,
    cell: Matrix,
}

impl LstmLayer {
    fn load(weights: &Weights<'_>, index: usize, width: usize) -> Result<Self> {
        let input_weight = weights.matrix(
            &format!("decoder.lstm.weight_ih_l{index}"),
            4 * width,
            width,
        )?;
        let recurrent_weight = weights.matrix(
            &format!("decoder.lstm.weight_hh_l{index}"),
            4 * width,
            width,
        )?;
        let input_bias = weights.vector(&format!("decoder.lstm.bias_ih_l{index}"), 4 * width)?;
        let recurrent_bias =
            weights.vector(&format!("decoder.lstm.bias_hh_l{index}"), 4 * width)?;
        Ok(Self {
            input_projection: Linear::from_parts(input_weight, input_bias),
            recurrent_projection: Linear::from_parts(recurrent_weight, recurrent_bias),
        })
    }

    fn step(&self, input: &Matrix, state: &mut LstmState) -> Matrix {
        let gates =
            self.input_projection.forward(input) + self.recurrent_projection.forward(&state.hidden);
        let width = state.hidden.ncols();
        for index in 0..width {
            let input_gate = sigmoid(gates[(0, index)]);
            let forget_gate = sigmoid(gates[(0, width + index)]);
            let candidate = gates[(0, 2 * width + index)].tanh();
            let output_gate = sigmoid(gates[(0, 3 * width + index)]);
            state.cell[(0, index)] = forget_gate * state.cell[(0, index)] + input_gate * candidate;
            state.hidden[(0, index)] = output_gate * state.cell[(0, index)].tanh();
        }
        state.hidden.clone()
    }
}

pub(crate) struct Emission {
    pub token_id: usize,
    pub frame: usize,
    pub duration: usize,
}

pub(crate) struct Decoder {
    embedding: Matrix,
    layers: Vec<LstmLayer>,
    projection: Linear,
    encoder_projection: Linear,
    joint: Linear,
    blank: usize,
    vocabulary_size: usize,
    durations: Vec<usize>,
    max_symbols_per_step: usize,
}

impl Decoder {
    pub fn load(weights: &Weights<'_>, config: &ModelConfig) -> Result<Self> {
        let width = config.decoder_hidden_size;
        let layers = (0..config.num_decoder_layers)
            .map(|index| LstmLayer::load(weights, index, width))
            .collect::<Result<_>>()?;
        Ok(Self {
            embedding: weights.matrix("decoder.embedding.weight", config.vocab_size, width)?,
            layers,
            projection: Linear::load(weights, "decoder.decoder_projector", width, width, true)?,
            encoder_projection: Linear::load(
                weights,
                "encoder_projector",
                config.encoder_config.hidden_size,
                width,
                true,
            )?,
            joint: Linear::load(
                weights,
                "joint.head",
                width,
                config.vocab_size + config.durations.len(),
                true,
            )?,
            blank: config.blank_token_id,
            vocabulary_size: config.vocab_size,
            durations: config.durations.clone(),
            max_symbols_per_step: config.max_symbols_per_step,
        })
    }

    fn predict(&self, token: usize, states: &mut [LstmState]) -> Matrix {
        let mut hidden = self.embedding.row(token).insert_axis(Axis(0)).to_owned();
        for (layer, state) in self.layers.iter().zip(states) {
            hidden = layer.step(&hidden, state);
        }
        self.projection.forward(&hidden)
    }

    pub fn decode(&self, encoded: &Matrix) -> Result<Vec<Emission>> {
        let width = self.embedding.ncols();
        let mut states: Vec<_> = self
            .layers
            .iter()
            .map(|_| LstmState {
                hidden: Matrix::zeros((1, width)),
                cell: Matrix::zeros((1, width)),
            })
            .collect();
        let mut prediction = self.predict(self.blank, &mut states);
        let projected = self.encoder_projection.forward(encoded);
        let mut frame = 0;
        let mut emissions = Vec::new();
        let step_limit = encoded.nrows().saturating_mul(self.max_symbols_per_step);
        for _ in 0..step_limit {
            if frame >= encoded.nrows() {
                return Ok(emissions);
            }
            let joint_input = (&prediction + &projected.row(frame).insert_axis(Axis(0)))
                .mapv(|value| value.max(0.0));
            let logits = self.joint.forward(&joint_input);
            let token_id = argmax(logits.row(0).iter().take(self.vocabulary_size).copied());
            let duration_index = argmax(logits.row(0).iter().skip(self.vocabulary_size).copied());
            let duration = frame_advance(token_id == self.blank, self.durations[duration_index]);
            if token_id != self.blank {
                emissions.push(Emission {
                    token_id,
                    frame,
                    duration,
                });
                prediction = self.predict(token_id, &mut states);
            }
            frame = frame.saturating_add(duration);
        }
        if frame < encoded.nrows() {
            return Err(invalid("TDT decoding exceeded its step budget"));
        }
        Ok(emissions)
    }
}

fn frame_advance(blank: bool, duration: usize) -> usize {
    if blank { duration.max(1) } else { duration }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_must_advance_but_zero_duration_tokens_stay_at_the_frame() {
        assert_eq!(frame_advance(true, 0), 1);
        assert_eq!(frame_advance(false, 0), 0);
        assert_eq!(frame_advance(false, 4), 4);
    }
}
