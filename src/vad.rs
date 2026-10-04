use std::ops::Range;

use crate::{
    Result,
    layers::TemporalConvolution,
    tensor::{Matrix, sigmoid, silu},
    weights::Weights,
};

pub(crate) struct VadHead {
    projection: TemporalConvolution,
    context: TemporalConvolution,
    output: TemporalConvolution,
}

impl VadHead {
    pub fn load(weights: &Weights<'_>, width: usize) -> Result<Self> {
        Ok(Self {
            projection: TemporalConvolution::load(
                weights,
                "vad_head.proj",
                width,
                128,
                1,
                false,
                true,
            )?,
            context: TemporalConvolution::load(weights, "vad_head.ctx", 128, 128, 5, false, true)?,
            output: TemporalConvolution::load(weights, "vad_head.out", 128, 1, 1, false, true)?,
        })
    }

    pub fn probabilities(&self, hidden: &Matrix) -> Vec<f32> {
        let mut hidden = self.projection.forward(hidden);
        silu(&mut hidden);
        hidden = self.context.forward(&hidden);
        silu(&mut hidden);
        self.output
            .forward(&hidden)
            .column(0)
            .iter()
            .copied()
            .map(sigmoid)
            .collect()
    }
}

pub(crate) fn speech_regions(probabilities: &[f32]) -> Vec<Range<usize>> {
    let mut regions: Vec<Range<usize>> = Vec::new();
    let mut start = None;
    for (frame, probability) in probabilities.iter().copied().chain([0.0]).enumerate() {
        if probability >= 0.5 {
            start.get_or_insert(frame);
            continue;
        }
        let Some(begin) = start.take() else { continue };
        // One 80 ms frame is a gap shorter than the upstream 100 ms threshold.
        if let Some(previous) = regions
            .last_mut()
            .filter(|previous| begin - previous.end <= 1)
        {
            previous.end = frame;
        } else {
            regions.push(begin..frame);
        }
    }
    regions.retain(|region| region.len() >= 2);
    regions
}

/// Select the last complete >=200 ms pause before the cap, then a pause
/// midpoint inside the cap, then the cap itself. Positions are PCM samples.
pub(crate) fn next_cut(
    regions: &[Range<usize>],
    origin: usize,
    available: usize,
    cap: usize,
) -> usize {
    let minimum = origin + 16_000;
    let limit = origin + cap;
    let mut previous = origin;
    let mut complete = None;
    let mut midpoint = None;
    for region in regions
        .iter()
        .filter(|region| region.end > origin)
        .cloned()
        .chain(std::iter::once(available..available))
    {
        let start = region.start.max(origin);
        if start.saturating_sub(previous) >= 3200 {
            let middle = previous + (start - previous) / 2;
            if previous >= minimum && start <= limit {
                complete = Some(middle);
            }
            if (minimum..=limit).contains(&middle) {
                midpoint = Some(middle);
            }
        }
        previous = previous.max(region.end);
    }
    complete.or(midpoint).unwrap_or(limit)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bridges_short_gaps_and_drops_short_speech() {
        assert_eq!(
            speech_regions(&[0., 1., 0., 1., 0., 0., 1., 0.]),
            vec![1..4]
        );
    }

    #[test]
    fn pause_cut_and_fallback_are_bounded() {
        assert_eq!(
            next_cut(&[0..160_000, 176_000..640_000], 0, 640_000, 480_000),
            168_000
        );
        assert_eq!(
            next_cut(std::slice::from_ref(&(0..640_000)), 0, 640_000, 480_000),
            480_000
        );
    }
}
