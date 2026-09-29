use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{DType, Device, Module, Result, Storage, Tensor};
use candle_nn::Linear;
use candle_transformers::quantized_var_builder::VarBuilder;
use half::f16;
use rayon::prelude::*;

const Q8_BLOCK_SIZE: usize = 32;
const Q8_BLOCK_BYTES: usize = 34;
const TILE_SIZE: usize = 4;
const COLUMN_TILE_SIZE: usize = 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Q8CpuKernelRoute {
    DequantizedFp32,
    ScalarQ8,
    Avx2FmaQ8,
}

impl Q8CpuKernelRoute {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::DequantizedFp32 => "dequantized-fp32",
            Self::ScalarQ8 => "scalar",
            Self::Avx2FmaQ8 => "avx2-fma",
        }
    }

    pub(crate) fn detected() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            return Self::for_x86_features(
                std::arch::is_x86_feature_detected!("avx2"),
                std::arch::is_x86_feature_detected!("fma"),
            );
        }
        #[cfg(not(target_arch = "x86_64"))]
        Self::ScalarQ8
    }

    #[cfg(any(target_arch = "x86_64", test))]
    fn for_x86_features(avx2: bool, fma: bool) -> Self {
        if avx2 && fma {
            Self::Avx2FmaQ8
        } else {
            Self::ScalarQ8
        }
    }
}

#[derive(Debug, Clone)]
struct Q8Blocks {
    scales: Vec<f32>,
    values: Vec<[i8; Q8_BLOCK_SIZE]>,
}

#[derive(Clone, Copy)]
struct Q8Row<'a> {
    scales: &'a [f32],
    values: &'a [[i8; Q8_BLOCK_SIZE]],
}

#[derive(Debug, Clone)]
struct Q8Matrix {
    blocks: Q8Blocks,
    input_width: usize,
    output_width: usize,
    route: Q8CpuKernelRoute,
}

impl Q8Matrix {
    fn from_qtensor(tensor: &QTensor, input_width: usize, output_width: usize) -> Result<Self> {
        if tensor.dtype() != GgmlDType::Q8_0 {
            candle_core::bail!(
                "Q8 CPU linear requires Q8_0 weights, got {:?}",
                tensor.dtype()
            )
        }
        if !tensor.device().is_cpu() {
            candle_core::bail!("Q8 CPU linear requires weights on the CPU")
        }
        if !input_width.is_multiple_of(Q8_BLOCK_SIZE) {
            candle_core::bail!(
                "Q8 CPU linear input width {input_width} must be divisible by {Q8_BLOCK_SIZE}"
            )
        }
        let expected_blocks = input_width / Q8_BLOCK_SIZE * output_width;
        let data = tensor.data()?;
        if data.len() != expected_blocks * Q8_BLOCK_BYTES {
            candle_core::bail!(
                "Q8 CPU linear weight data has {} bytes, expected {}",
                data.len(),
                expected_blocks * Q8_BLOCK_BYTES
            )
        }
        let mut scales = Vec::with_capacity(expected_blocks);
        let mut values = Vec::with_capacity(expected_blocks);
        for bytes in data.chunks_exact(Q8_BLOCK_BYTES) {
            scales.push(f16::from_bits(u16::from_le_bytes([bytes[0], bytes[1]])).to_f32());
            let mut block = [0_i8; Q8_BLOCK_SIZE];
            for (output, input) in block.iter_mut().zip(&bytes[2..]) {
                *output = *input as i8;
            }
            values.push(block);
        }
        let blocks = Q8Blocks { scales, values };
        Ok(Self {
            blocks,
            input_width,
            output_width,
            route: Q8CpuKernelRoute::detected(),
        })
    }

    fn forward(&self, input: &Tensor, bias: Option<&[f32]>) -> Result<Tensor> {
        if !input.device().is_cpu() || input.dtype() != DType::F32 {
            candle_core::bail!(
                "Q8 CPU linear requires CPU f32 activations, got {:?} on {:?}",
                input.dtype(),
                input.device().location()
            )
        }
        if !input.is_contiguous() {
            return self.forward(&input.contiguous()?, bias);
        }
        if let Some(bias) = bias {
            if bias.len() != self.output_width {
                candle_core::bail!(
                    "Q8 CPU linear expected bias width {}, got {}",
                    self.output_width,
                    bias.len()
                )
            }
        }
        let mut output_shape = input.dims().to_vec();
        let Some(last_dim) = output_shape.last_mut() else {
            candle_core::bail!("Q8 CPU linear requires an activation tensor with rank at least one")
        };
        if *last_dim != self.input_width {
            candle_core::bail!(
                "Q8 CPU linear expected activation width {}, got {}",
                self.input_width,
                *last_dim
            )
        }
        *last_dim = self.output_width;
        let row_count = input.elem_count() / self.input_width;
        let input_blocks = {
            let (storage, layout) = input.storage_and_layout();
            let (start, end) = layout.contiguous_offsets().ok_or_else(|| {
                candle_core::Error::msg("Q8 CPU linear activation layout was not contiguous")
            })?;
            let Storage::Cpu(storage) = &*storage else {
                candle_core::bail!("Q8 CPU linear activation storage was not on the CPU")
            };
            quantize_rows(&storage.as_slice::<f32>()?[start..end], row_count)
        };
        let blocks_per_row = self.input_width / Q8_BLOCK_SIZE;
        let mut output = vec![0_f32; row_count * self.output_width];
        if let Some(bias) = bias {
            output
                .par_chunks_mut(self.output_width)
                .for_each(|row| row.copy_from_slice(bias));
        }

        if row_count < TILE_SIZE {
            for (row_index, row_output) in output.chunks_mut(self.output_width).enumerate() {
                let input_row = input_blocks.row(row_index, blocks_per_row);
                row_output
                    .par_chunks_mut(COLUMN_TILE_SIZE)
                    .enumerate()
                    .for_each(|(group_index, group_output)| {
                        let first_column = group_index * COLUMN_TILE_SIZE;
                        if group_output.len() == COLUMN_TILE_SIZE {
                            let weights = std::array::from_fn(|offset| {
                                self.blocks.row(first_column + offset, blocks_per_row)
                            });
                            let values = dot_eight_columns(weights, input_row, self.route);
                            for (output, value) in group_output.iter_mut().zip(values) {
                                *output += value;
                            }
                        } else {
                            for (offset, value) in group_output.iter_mut().enumerate() {
                                let weight = self.blocks.row(first_column + offset, blocks_per_row);
                                *value += dot(weight, input_row, self.route);
                            }
                        }
                    });
            }
        } else {
            output
                .par_chunks_mut(self.output_width * TILE_SIZE)
                .enumerate()
                .for_each(|(group_index, group_output)| {
                    let first_row = group_index * TILE_SIZE;
                    if group_output.len() == self.output_width * TILE_SIZE {
                        let inputs = std::array::from_fn(|offset| {
                            input_blocks.row(first_row + offset, blocks_per_row)
                        });
                        for column in 0..self.output_width {
                            let weight = self.blocks.row(column, blocks_per_row);
                            let values = dot_four_rows(weight, inputs, self.route);
                            for (row, value) in values.into_iter().enumerate() {
                                group_output[row * self.output_width + column] += value;
                            }
                        }
                    } else {
                        for (row_offset, row_output) in
                            group_output.chunks_mut(self.output_width).enumerate()
                        {
                            let input_row =
                                input_blocks.row(first_row + row_offset, blocks_per_row);
                            for (column, value) in row_output.iter_mut().enumerate() {
                                let weight = self.blocks.row(column, blocks_per_row);
                                *value += dot(weight, input_row, self.route);
                            }
                        }
                    }
                });
        }

        Tensor::from_vec(output, output_shape, &Device::Cpu)
    }
}

impl Q8Blocks {
    fn row(&self, row: usize, blocks_per_row: usize) -> Q8Row<'_> {
        let range = row * blocks_per_row..(row + 1) * blocks_per_row;
        Q8Row {
            scales: &self.scales[range.clone()],
            values: &self.values[range],
        }
    }
}

fn quantize_rows(values: &[f32], row_count: usize) -> Q8Blocks {
    let quantize = |values: &[f32]| {
        let maximum = values
            .iter()
            .fold(0_f32, |maximum, value| maximum.max(value.abs()));
        let scale = maximum / 127_f32;
        let inverse = if scale == 0_f32 { 0_f32 } else { scale.recip() };
        let mut quantized = [0_i8; Q8_BLOCK_SIZE];
        for (output, value) in quantized.iter_mut().zip(values) {
            *output = (value * inverse).round() as i8;
        }
        (f16::from_f32(scale).to_f32(), quantized)
    };
    let blocks = if row_count < TILE_SIZE {
        values
            .chunks_exact(Q8_BLOCK_SIZE)
            .map(quantize)
            .collect::<Vec<_>>()
    } else {
        values
            .par_chunks_exact(Q8_BLOCK_SIZE)
            .map(quantize)
            .collect::<Vec<_>>()
    };
    let (scales, values) = blocks.into_iter().unzip();
    Q8Blocks { scales, values }
}

fn dot(left: Q8Row<'_>, right: Q8Row<'_>, route: Q8CpuKernelRoute) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if route == Q8CpuKernelRoute::Avx2FmaQ8 {
        // SAFETY: the route is selected only after runtime AVX2 and FMA detection.
        return unsafe { dot_avx2(left, right) };
    }
    dot_scalar(left, right)
}

fn dot_eight_columns(
    weights: [Q8Row<'_>; COLUMN_TILE_SIZE],
    input: Q8Row<'_>,
    route: Q8CpuKernelRoute,
) -> [f32; COLUMN_TILE_SIZE] {
    #[cfg(target_arch = "x86_64")]
    if route == Q8CpuKernelRoute::Avx2FmaQ8 {
        // SAFETY: the route is selected only after runtime AVX2 and FMA detection.
        return unsafe { dot_eight_columns_avx2(weights, input) };
    }
    weights.map(|weight| dot_scalar(weight, input))
}

fn dot_four_rows(
    weight: Q8Row<'_>,
    inputs: [Q8Row<'_>; TILE_SIZE],
    route: Q8CpuKernelRoute,
) -> [f32; TILE_SIZE] {
    #[cfg(target_arch = "x86_64")]
    if route == Q8CpuKernelRoute::Avx2FmaQ8 {
        // SAFETY: the route is selected only after runtime AVX2 and FMA detection.
        return unsafe { dot_four_rows_avx2(weight, inputs) };
    }
    inputs.map(|input| dot_scalar(weight, input))
}

fn dot_scalar(left: Q8Row<'_>, right: Q8Row<'_>) -> f32 {
    left.values
        .iter()
        .zip(right.values)
        .zip(left.scales.iter().zip(right.scales))
        .map(|((left, right), (left_scale, right_scale))| {
            let integer_dot = left
                .iter()
                .zip(right)
                .map(|(left, right)| i32::from(*left) * i32::from(*right))
                .sum::<i32>();
            integer_dot as f32 * left_scale * right_scale
        })
        .sum()
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_avx2(left: Q8Row<'_>, right: Q8Row<'_>) -> f32 {
    use std::arch::x86_64::*;

    let mut accumulator = _mm256_setzero_ps();
    for ((left_values, left_scale), (right_values, right_scale)) in left
        .values
        .iter()
        .zip(left.scales)
        .zip(right.values.iter().zip(right.scales))
    {
        let integer_dot = unsafe { block_dot_avx2(left_values, right_values) };
        accumulator = _mm256_fmadd_ps(
            _mm256_set1_ps(left_scale * right_scale),
            _mm256_cvtepi32_ps(integer_dot),
            accumulator,
        );
    }
    horizontal_sum_avx2(accumulator)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_eight_columns_avx2(
    weights: [Q8Row<'_>; COLUMN_TILE_SIZE],
    input: Q8Row<'_>,
) -> [f32; COLUMN_TILE_SIZE] {
    use std::arch::x86_64::*;

    let mut accumulators = [_mm256_setzero_ps(); COLUMN_TILE_SIZE];
    for block_index in 0..input.values.len() {
        // SAFETY: scales and values are appended together for every block.
        let input_scale = unsafe { input.scales.get_unchecked(block_index) };
        let input_values =
            unsafe { _mm256_loadu_si256(input.values[block_index].as_ptr().cast::<__m256i>()) };
        for (accumulator, weight) in accumulators.iter_mut().zip(weights) {
            // SAFETY: every row has the validated input-width block count.
            let weight_block = unsafe { weight.values.get_unchecked(block_index) };
            // SAFETY: scales and values are appended together for every block.
            let weight_scale = unsafe { weight.scales.get_unchecked(block_index) };
            let weight_values =
                unsafe { _mm256_loadu_si256(weight_block.as_ptr().cast::<__m256i>()) };
            let integer_dot = block_dot_values_avx2(weight_values, input_values);
            *accumulator = _mm256_fmadd_ps(
                _mm256_set1_ps(weight_scale * input_scale),
                _mm256_cvtepi32_ps(integer_dot),
                *accumulator,
            );
        }
    }
    accumulators.map(|accumulator| horizontal_sum_avx2(accumulator))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_four_rows_avx2(
    weight: Q8Row<'_>,
    inputs: [Q8Row<'_>; TILE_SIZE],
) -> [f32; TILE_SIZE] {
    use std::arch::x86_64::*;

    let mut accumulators = [_mm256_setzero_ps(); TILE_SIZE];
    for block_index in 0..weight.values.len() {
        // SAFETY: scales and values are appended together for every block.
        let weight_scale = unsafe { weight.scales.get_unchecked(block_index) };
        let weight_values =
            unsafe { _mm256_loadu_si256(weight.values[block_index].as_ptr().cast::<__m256i>()) };
        for (accumulator, input) in accumulators.iter_mut().zip(inputs) {
            // SAFETY: every row has the validated input-width block count.
            let input_block = unsafe { input.values.get_unchecked(block_index) };
            // SAFETY: scales and values are appended together for every block.
            let input_scale = unsafe { input.scales.get_unchecked(block_index) };
            let input_values =
                unsafe { _mm256_loadu_si256(input_block.as_ptr().cast::<__m256i>()) };
            let integer_dot = block_dot_values_avx2(weight_values, input_values);
            *accumulator = _mm256_fmadd_ps(
                _mm256_set1_ps(weight_scale * input_scale),
                _mm256_cvtepi32_ps(integer_dot),
                *accumulator,
            );
        }
    }
    accumulators.map(|accumulator| horizontal_sum_avx2(accumulator))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn block_dot_avx2(
    left: &[i8; Q8_BLOCK_SIZE],
    right: &[i8; Q8_BLOCK_SIZE],
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    let left = unsafe { _mm256_loadu_si256(left.as_ptr().cast::<__m256i>()) };
    let right = unsafe { _mm256_loadu_si256(right.as_ptr().cast::<__m256i>()) };
    block_dot_values_avx2(left, right)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn block_dot_values_avx2(
    left: std::arch::x86_64::__m256i,
    right: std::arch::x86_64::__m256i,
) -> std::arch::x86_64::__m256i {
    use std::arch::x86_64::*;

    let absolute_left = _mm256_sign_epi8(left, left);
    let signed_right = _mm256_sign_epi8(right, left);
    let pairs = _mm256_maddubs_epi16(absolute_left, signed_right);
    _mm256_madd_epi16(_mm256_set1_epi16(1), pairs)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
fn horizontal_sum_avx2(values: std::arch::x86_64::__m256) -> f32 {
    use std::arch::x86_64::*;

    let high = _mm256_extractf128_ps::<1>(values);
    let low = _mm256_castps256_ps128(values);
    let sum = _mm_add_ps(low, high);
    let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let sum = _mm_add_ss(sum, _mm_movehdup_ps(sum));
    _mm_cvtss_f32(sum)
}

#[derive(Debug, Clone)]
enum Q8CpuLinearInner {
    Dequantized(Linear),
    Quantized(Q8Matrix),
}

#[derive(Debug, Clone)]
pub(crate) struct Q8CpuLinear {
    inner: Q8CpuLinearInner,
    bias: Option<Vec<f32>>,
}

impl Q8CpuLinear {
    pub(crate) fn load_quantized(
        input_width: usize,
        output_width: usize,
        bias: bool,
        builder: VarBuilder,
    ) -> Result<Self> {
        let weight = builder.get((output_width, input_width), "weight")?;
        let matrix = Q8Matrix::from_qtensor(&weight, input_width, output_width)?;
        let bias = load_bias(output_width, bias, &builder)?
            .map(|bias| bias.to_vec1::<f32>())
            .transpose()?;
        Ok(Self {
            inner: Q8CpuLinearInner::Quantized(matrix),
            bias,
        })
    }

    pub(crate) fn load_dequantized(
        input_width: usize,
        output_width: usize,
        bias: bool,
        builder: VarBuilder,
    ) -> Result<Self> {
        let weight = builder
            .get((output_width, input_width), "weight")?
            .dequantize(builder.device())?;
        let bias = load_bias(output_width, bias, &builder)?;
        Ok(Self {
            inner: Q8CpuLinearInner::Dequantized(Linear::new(weight, bias)),
            bias: None,
        })
    }

    pub(crate) fn route(&self) -> Q8CpuKernelRoute {
        match &self.inner {
            Q8CpuLinearInner::Dequantized(_) => Q8CpuKernelRoute::DequantizedFp32,
            Q8CpuLinearInner::Quantized(matrix) => matrix.route,
        }
    }

    #[cfg(test)]
    fn from_qtensor_for_test(
        tensor: std::sync::Arc<QTensor>,
        bias: Option<Tensor>,
    ) -> Result<Self> {
        let [output_width, input_width] = tensor.shape().dims() else {
            candle_core::bail!("test Q8 linear requires a rank-two weight")
        };
        Ok(Self {
            inner: Q8CpuLinearInner::Quantized(Q8Matrix::from_qtensor(
                &tensor,
                *input_width,
                *output_width,
            )?),
            bias: bias.map(|bias| bias.to_vec1::<f32>()).transpose()?,
        })
    }
}

fn load_bias(output_width: usize, load: bool, builder: &VarBuilder) -> Result<Option<Tensor>> {
    load.then(|| {
        builder
            .get(output_width, "bias")?
            .dequantize(builder.device())
    })
    .transpose()
}

impl Module for Q8CpuLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        match &self.inner {
            Q8CpuLinearInner::Dequantized(linear) => linear.forward(input),
            Q8CpuLinearInner::Quantized(matrix) => matrix.forward(input, self.bias.as_deref()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::quantized::QMatMul;
    use std::time::{Duration, Instant};

    fn weight_fixture(output_width: usize, input_width: usize) -> Result<Tensor> {
        let values = (0..output_width * input_width)
            .map(|index| ((index % 251) as f32 - 125_f32) / 125_f32)
            .collect::<Vec<_>>();
        Tensor::from_vec(values, (output_width, input_width), &Device::Cpu)
    }

    fn measure(iterations: usize, mut forward: impl FnMut() -> Result<Tensor>) -> Result<Duration> {
        let _ = std::hint::black_box(forward()?);
        let started = Instant::now();
        for _ in 0..iterations {
            let _ = std::hint::black_box(forward()?);
        }
        Ok(started.elapsed())
    }

    #[test]
    fn standard_build_selects_runtime_avx2_when_available() -> Result<()> {
        let weight = weight_fixture(64, 64)?;
        let weight = std::sync::Arc::new(QTensor::quantize(&weight, GgmlDType::Q8_0)?);
        let linear = Q8CpuLinear::from_qtensor_for_test(weight, None)?;
        if cfg!(target_arch = "x86_64")
            && std::arch::is_x86_feature_detected!("avx2")
            && std::arch::is_x86_feature_detected!("fma")
        {
            assert_eq!(linear.route(), Q8CpuKernelRoute::Avx2FmaQ8);
        } else {
            assert_eq!(linear.route(), Q8CpuKernelRoute::ScalarQ8);
        }
        Ok(())
    }

    #[test]
    fn incomplete_x86_feature_sets_select_the_scalar_fallback() {
        assert_eq!(
            Q8CpuKernelRoute::for_x86_features(false, false),
            Q8CpuKernelRoute::ScalarQ8
        );
        assert_eq!(
            Q8CpuKernelRoute::for_x86_features(true, false),
            Q8CpuKernelRoute::ScalarQ8
        );
        assert_eq!(
            Q8CpuKernelRoute::for_x86_features(false, true),
            Q8CpuKernelRoute::ScalarQ8
        );
    }

    #[test]
    fn runtime_q8_linear_matches_candle_q8_linear() -> Result<()> {
        let weight = weight_fixture(12, 64)?;
        let weight = std::sync::Arc::new(QTensor::quantize(&weight, GgmlDType::Q8_0)?);
        let bias_values = (0..12)
            .map(|index| index as f32 / 10_f32)
            .collect::<Vec<_>>();
        let bias = Tensor::from_vec(bias_values.clone(), 12, &Device::Cpu)?;
        let input = (0..6 * 64)
            .map(|index| ((index % 127) as f32 - 63_f32) / 63_f32)
            .collect::<Vec<_>>();
        let input = Tensor::from_vec(input, (2, 3, 64), &Device::Cpu)?;
        let expected = QMatMul::from_arc(weight.clone())?.forward(&input)?;
        let actual =
            Q8CpuLinear::from_qtensor_for_test(weight, Some(bias.clone()))?.forward(&input)?;
        let expected = expected.flatten_all()?.to_vec1::<f32>()?;
        let actual = actual.flatten_all()?.to_vec1::<f32>()?;
        let maximum_difference = expected
            .iter()
            .zip(actual)
            .enumerate()
            .map(|(index, (expected, actual))| (expected + bias_values[index % 12] - actual).abs())
            .fold(0_f32, f32::max);
        assert!(
            maximum_difference <= 1e-4,
            "runtime Q8 output differed from Candle Q8 by {maximum_difference}"
        );
        Ok(())
    }

    #[test]
    fn scalar_and_runtime_q8_routes_match() -> Result<()> {
        let weight = weight_fixture(12, 64)?;
        let weight = QTensor::quantize(&weight, GgmlDType::Q8_0)?;
        let mut runtime = Q8Matrix::from_qtensor(&weight, 64, 12)?;
        let input = Tensor::from_vec(
            (0..3 * 64)
                .map(|index| ((index % 97) as f32 - 48_f32) / 48_f32)
                .collect::<Vec<_>>(),
            (3, 64),
            &Device::Cpu,
        )?;
        runtime.route = Q8CpuKernelRoute::ScalarQ8;
        let scalar = runtime.forward(&input, None)?.to_vec2::<f32>()?;
        runtime.route = Q8CpuKernelRoute::detected();
        let detected = runtime.forward(&input, None)?.to_vec2::<f32>()?;
        for (scalar, detected) in scalar
            .into_iter()
            .flatten()
            .zip(detected.into_iter().flatten())
        {
            assert!((scalar - detected).abs() <= 1e-4);
        }
        Ok(())
    }

    #[test]
    fn q8_linear_rejects_a_mismatched_bias_width() -> Result<()> {
        let weight = QTensor::quantize(&weight_fixture(12, 64)?, GgmlDType::Q8_0)?;
        let matrix = Q8Matrix::from_qtensor(&weight, 64, 12)?;
        let input = Tensor::zeros((1, 64), DType::F32, &Device::Cpu)?;
        let error = matrix.forward(&input, Some(&[0_f32])).unwrap_err();
        assert!(error.to_string().contains("expected bias width 12, got 1"));
        Ok(())
    }

    #[test]
    fn q8_linear_is_deterministic_in_request_scoped_rayon_pools() -> Result<()> {
        let weight = weight_fixture(64, 64)?;
        let weight = QTensor::quantize(&weight, GgmlDType::Q8_0)?;
        let matrix = Q8Matrix::from_qtensor(&weight, 64, 64)?;
        let input = Tensor::from_vec(
            (0..8 * 64)
                .map(|index| ((index % 89) as f32 - 44_f32) / 44_f32)
                .collect::<Vec<_>>(),
            (8, 64),
            &Device::Cpu,
        )?;
        let run = |thread_count| -> Result<Vec<f32>> {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(thread_count)
                .build()
                .map_err(candle_core::Error::wrap)?;
            pool.install(|| {
                assert_eq!(rayon::current_num_threads(), thread_count);
                matrix
                    .forward(&input, None)?
                    .flatten_all()?
                    .to_vec1::<f32>()
            })
        };
        assert_eq!(run(1)?, run(4)?);
        Ok(())
    }

    #[test]
    #[ignore = "manual encoder/decoder Q8 CPU microbenchmark; wall time is not a CI contract"]
    fn q8_linear_microbenchmark() -> Result<()> {
        benchmark_shape("encoder", 768, 768, 1_500, 3)?;
        benchmark_shape("decoder", 768, 3_072, 2, 10)?;
        Ok(())
    }

    fn benchmark_shape(
        label: &str,
        input_width: usize,
        output_width: usize,
        row_count: usize,
        iterations: usize,
    ) -> Result<()> {
        let weight = weight_fixture(output_width, input_width)?;
        let quantized = QTensor::quantize(&weight, GgmlDType::Q8_0)?;
        let mut scalar = Q8Matrix::from_qtensor(&quantized, input_width, output_width)?;
        scalar.route = Q8CpuKernelRoute::ScalarQ8;
        let detected = Q8Matrix::from_qtensor(&quantized, input_width, output_width)?;
        let fp32 = Linear::new(weight, None);
        let input = Tensor::from_vec(
            (0..row_count * input_width)
                .map(|index| ((index % 127) as f32 - 63_f32) / 63_f32)
                .collect::<Vec<_>>(),
            (row_count, input_width),
            &Device::Cpu,
        )?;

        let scalar_duration = measure(iterations, || scalar.forward(&input, None))?;
        let detected_duration = measure(iterations, || detected.forward(&input, None))?;
        let fp32_duration = measure(iterations, || fp32.forward(&input))?;
        eprintln!(
            "{label}-shaped linear: scalar-q8={scalar_duration:?}, {}={detected_duration:?}, fp32={fp32_duration:?}",
            detected.route.as_str()
        );
        Ok(())
    }
}
