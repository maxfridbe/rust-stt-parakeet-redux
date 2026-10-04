use parakeet_redux::features::FeatureExtractor;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    samples: Vec<f32>,
    features: Vec<Vec<f32>>,
}

#[test]
fn frontend_matches_independent_pytorch_reference() {
    let fixture: Fixture = serde_json::from_str(include_str!("fixtures/reference.json")).unwrap();
    let actual = FeatureExtractor::default()
        .extract(&fixture.samples)
        .unwrap();
    assert_eq!(
        actual.dim(),
        (fixture.features.len(), fixture.features[0].len())
    );
    let maximum_error = actual
        .iter()
        .zip(fixture.features.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        maximum_error < 1e-4,
        "frontend maximum error: {maximum_error}"
    );
}
