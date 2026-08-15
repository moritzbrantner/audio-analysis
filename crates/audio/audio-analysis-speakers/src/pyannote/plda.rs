//! Typed PLDA transforms for pyannote community diarization.

use std::{fs, path::Path};

use audio_contracts::{DetectError, Result};
use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct PldaTransform {
    schema_version: u32,
    input_dimension: usize,
    output_dimension: usize,
    mean1: Vec<f64>,
    mean2: Vec<f64>,
    lda: Vec<Vec<f64>>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(super) struct PldaModel {
    schema_version: u32,
    dimension: usize,
    mean: Vec<f64>,
    transform: Vec<Vec<f64>>,
    psi: Vec<f64>,
}

/// Fully validated transform used by the VBx implementation.
#[derive(Debug, Clone)]
pub(super) struct Plda {
    xvector: PldaTransform,
    model: PldaModel,
    order: Vec<usize>,
}

impl Plda {
    pub(super) fn load(transform_path: &Path, model_path: &Path) -> Result<Self> {
        let xvector: PldaTransform = read_json(transform_path, "PLDA transform")?;
        let model: PldaModel = read_json(model_path, "PLDA model")?;
        validate_transform(&xvector)?;
        validate_model(&model)?;
        if xvector.output_dimension != model.dimension {
            return Err(setup_error(
                "PLDA transform outputDimension must match PLDA model dimension",
            ));
        }
        let mut order = (0..model.dimension).collect::<Vec<_>>();
        // scipy.linalg.eigh(B, W), as used by the pinned pyannote PLDA
        // implementation, returns the same already-diagonalized rows sorted by
        // ascending generalized eigenvalue. VBx reverses that order.
        order.sort_by(|left, right| {
            model.psi[*right]
                .partial_cmp(&model.psi[*left])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| left.cmp(right))
        });
        Ok(Self {
            xvector,
            model,
            order,
        })
    }

    pub(super) fn input_dimension(&self) -> usize {
        self.xvector.input_dimension
    }

    pub(super) fn phi(&self) -> Vec<f64> {
        self.order
            .iter()
            .map(|index| self.model.psi[*index])
            .collect()
    }

    /// Applies the exact centering, length normalization, LDA, and PLDA
    /// projection used by `pyannote.audio.core.plda.PLDA`.
    pub(super) fn project(&self, embedding: &[f32]) -> Result<Vec<f64>> {
        if embedding.len() != self.xvector.input_dimension {
            return Err(model_error(format!(
                "embedding dimension {} does not match PLDA input dimension {}",
                embedding.len(),
                self.xvector.input_dimension
            )));
        }
        let centered = embedding
            .iter()
            .zip(&self.xvector.mean1)
            .map(|(value, mean)| f64::from(*value) - mean)
            .collect::<Vec<_>>();
        let centered = length_normalize(&centered, "centered x-vector")?;
        let input_scale = (self.xvector.input_dimension as f64).sqrt();
        let mut lda = vec![0.0; self.xvector.output_dimension];
        for (row, value) in centered.iter().enumerate() {
            for (column, output) in lda.iter_mut().enumerate() {
                *output += input_scale * value * self.xvector.lda[row][column];
            }
        }
        for (value, mean) in lda.iter_mut().zip(&self.xvector.mean2) {
            *value -= mean;
        }
        let lda = length_normalize(&lda, "LDA x-vector")?;
        let output_scale = (self.xvector.output_dimension as f64).sqrt();
        let centered = lda
            .iter()
            .zip(&self.model.mean)
            .map(|(value, mean)| output_scale * value - mean)
            .collect::<Vec<_>>();
        let mut projected: Vec<f64> = Vec::with_capacity(self.model.dimension);
        for &row in &self.order {
            projected.push(
                self.model.transform[row]
                    .iter()
                    .zip(&centered)
                    .map(|(weight, value)| weight * value)
                    .sum(),
            );
        }
        if projected.iter().any(|value| !value.is_finite()) {
            return Err(model_error("PLDA projection produced non-finite values"));
        }
        Ok(projected)
    }
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &Path, role: &str) -> Result<T> {
    let bytes =
        fs::read(path).map_err(|error| setup_error(format!("failed to read {role}: {error}")))?;
    serde_json::from_slice(&bytes)
        .map_err(|error| setup_error(format!("failed to parse {role}: {error}")))
}

fn validate_transform(value: &PldaTransform) -> Result<()> {
    if value.schema_version != 1
        || value.input_dimension == 0
        || value.output_dimension == 0
        || value.mean1.len() != value.input_dimension
        || value.mean2.len() != value.output_dimension
        || value.lda.len() != value.input_dimension
        || value
            .lda
            .iter()
            .any(|row| row.len() != value.output_dimension)
        || !all_finite(&value.mean1)
        || !all_finite(&value.mean2)
        || value.lda.iter().any(|row| !all_finite(row))
    {
        return Err(setup_error(
            "PLDA transform dimensions or values are incompatible",
        ));
    }
    Ok(())
}

fn validate_model(value: &PldaModel) -> Result<()> {
    if value.schema_version != 1
        || value.dimension == 0
        || value.mean.len() != value.dimension
        || value.psi.len() != value.dimension
        || value.transform.len() != value.dimension
        || value
            .transform
            .iter()
            .any(|row| row.len() != value.dimension)
        || !all_finite(&value.mean)
        || !all_finite(&value.psi)
        || value.psi.iter().any(|value| *value <= 0.0)
        || value.transform.iter().any(|row| !all_finite(row))
    {
        return Err(setup_error(
            "PLDA model dimensions or values are incompatible",
        ));
    }
    Ok(())
}

fn length_normalize(values: &[f64], role: &str) -> Result<Vec<f64>> {
    let norm = values.iter().map(|value| value * value).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(model_error(format!(
            "{role} norm must be finite and non-zero"
        )));
    }
    Ok(values.iter().map(|value| value / norm).collect())
}

fn all_finite(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite())
}

fn setup_error(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("setup_error: {}", message.into()))
}

fn model_error(message: impl Into<String>) -> DetectError {
    DetectError::InvalidArgument(format!("model_output_mismatch: {}", message.into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transform() -> PldaTransform {
        PldaTransform {
            schema_version: 1,
            input_dimension: 2,
            output_dimension: 2,
            mean1: vec![0.0, 0.0],
            mean2: vec![0.0, 0.0],
            lda: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        }
    }

    fn model() -> PldaModel {
        PldaModel {
            schema_version: 1,
            dimension: 2,
            mean: vec![0.0, 0.0],
            transform: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            psi: vec![1.0, 2.0],
        }
    }

    #[test]
    fn every_plda_value_is_validated() {
        let mut transform = transform();
        transform.lda[1][1] = f64::NAN;
        assert!(validate_transform(&transform).is_err());

        let mut model = model();
        model.psi[0] = 0.0;
        assert!(validate_model(&model).is_err());
    }

    #[test]
    fn projection_applies_both_typed_artifacts() {
        let plda = Plda {
            xvector: transform(),
            model: model(),
            order: vec![1, 0],
        };
        let projected = plda.project(&[1.0, 0.0]).unwrap();
        assert_eq!(projected, vec![0.0, 2.0_f64.sqrt()]);

        let mut changed = model();
        changed.transform[1][0] = 0.5;
        let changed = Plda {
            xvector: transform(),
            model: changed,
            order: vec![1, 0],
        }
        .project(&[1.0, 0.0])
        .unwrap();
        assert_ne!(projected, changed);
    }
}
