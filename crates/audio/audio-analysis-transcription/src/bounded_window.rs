//! Reusable bounded-memory PCM window scheduling.
//!
//! This module deliberately owns only sample validation, buffering, window
//! scheduling, and cancellation boundaries. Product concerns such as live
//! transcript stabilization and wall-clock event timestamps belong to callers.

use media_core::{DetectError, Result};

/// Configuration for a [`BoundedPcmWindowSession`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundedPcmWindowConfig {
    /// PCM sample rate represented by the session.
    pub sample_rate: u32,
    /// Number of samples in each complete processing window.
    pub window_samples: usize,
    /// Number of samples between the starts of successive windows.
    pub hop_samples: usize,
    /// Maximum number of unprocessed samples retained by the session.
    pub max_buffered_samples: usize,
}

impl BoundedPcmWindowConfig {
    /// Creates validated bounded-window scheduling configuration.
    pub fn new(
        sample_rate: u32,
        window_samples: usize,
        hop_samples: usize,
        max_buffered_samples: usize,
    ) -> Result<Self> {
        let config = Self {
            sample_rate,
            window_samples,
            hop_samples,
            max_buffered_samples,
        };
        config.validate()?;
        Ok(config)
    }

    /// Validates the scheduling and memory bounds.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(invalid_window_config(
                "sample_rate must be greater than zero",
            ));
        }
        if self.window_samples == 0 {
            return Err(invalid_window_config(
                "window_samples must be greater than zero",
            ));
        }
        if self.hop_samples == 0 {
            return Err(invalid_window_config(
                "hop_samples must be greater than zero",
            ));
        }
        if self.hop_samples > self.window_samples {
            return Err(invalid_window_config(
                "hop_samples must not exceed window_samples",
            ));
        }
        if self.max_buffered_samples < self.window_samples {
            return Err(invalid_window_config(
                "max_buffered_samples must be at least window_samples",
            ));
        }
        Ok(())
    }
}

/// A ready PCM window delivered to a caller-owned processor.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundedPcmWindow {
    /// Zero-based position in the original continuous input stream.
    pub start_sample: u64,
    /// PCM samples for this window. A final tail can be shorter than the configured window.
    pub samples: Vec<f32>,
    /// Whether this is the final, potentially short tail emitted by [`BoundedPcmWindowSession::flush`].
    pub is_final_tail: bool,
}

impl BoundedPcmWindow {
    /// Duration represented by this window at the supplied sample rate.
    pub fn duration_seconds(&self, sample_rate: u32) -> f64 {
        self.samples.len() as f64 / sample_rate as f64
    }
}

/// Observable bounded-window session statistics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundedPcmWindowStats {
    /// Number of validated samples accepted from callers.
    pub samples_ingested: u64,
    /// Number of windows passed to the processor.
    pub windows_processed: u64,
    /// Number of samples contained in passed windows, including overlap.
    pub window_samples_processed: u64,
    /// Number of samples retained for a future complete window or final flush.
    pub buffered_samples: usize,
    /// Duration of all accepted input samples.
    pub input_duration_seconds: f64,
}

/// Reusable synchronous session for continuously arriving f32 PCM samples.
#[derive(Debug)]
pub struct BoundedPcmWindowSession {
    config: BoundedPcmWindowConfig,
    buffered_samples: Vec<f32>,
    next_window_start_sample: u64,
    samples_ingested: u64,
    windows_processed: u64,
    window_samples_processed: u64,
}

impl BoundedPcmWindowSession {
    /// Creates a session with a bounded amount of retained PCM.
    pub fn new(config: BoundedPcmWindowConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            buffered_samples: Vec::with_capacity(config.max_buffered_samples),
            config,
            next_window_start_sample: 0,
            samples_ingested: 0,
            windows_processed: 0,
            window_samples_processed: 0,
        })
    }

    /// Returns the fixed scheduling configuration.
    pub fn config(&self) -> BoundedPcmWindowConfig {
        self.config
    }

    /// Returns current session statistics.
    pub fn stats(&self) -> BoundedPcmWindowStats {
        BoundedPcmWindowStats {
            samples_ingested: self.samples_ingested,
            windows_processed: self.windows_processed,
            window_samples_processed: self.window_samples_processed,
            buffered_samples: self.buffered_samples.len(),
            input_duration_seconds: self.samples_ingested as f64 / self.config.sample_rate as f64,
        }
    }

    /// Ingests validated PCM and processes every newly ready complete window.
    ///
    /// Cancellation is checked immediately before each processor invocation,
    /// making each window boundary safe and deterministic. The caller can pass
    /// an existing pipeline observer with `|| observer.cancellation_requested()`.
    pub fn ingest(
        &mut self,
        samples: &[f32],
        processor: &mut dyn FnMut(BoundedPcmWindow) -> Result<()>,
        cancellation_requested: &dyn Fn() -> bool,
    ) -> Result<()> {
        validate_samples(samples)?;
        let mut remaining = samples;
        while !remaining.is_empty() {
            self.process_ready_windows(processor, cancellation_requested)?;
            let capacity = self
                .config
                .max_buffered_samples
                .checked_sub(self.buffered_samples.len())
                .ok_or_else(|| invalid_window_config("buffer exceeded configured capacity"))?;
            if capacity == 0 {
                return Err(invalid_window_config(
                    "buffer cannot make progress with the configured window and hop sizes",
                ));
            }
            let accepted = remaining.len().min(capacity);
            self.buffered_samples
                .extend_from_slice(&remaining[..accepted]);
            self.samples_ingested += accepted as u64;
            remaining = &remaining[accepted..];
        }
        self.process_ready_windows(processor, cancellation_requested)
    }

    /// Processes the remaining samples as exactly one final tail, if any.
    pub fn flush(
        &mut self,
        processor: &mut dyn FnMut(BoundedPcmWindow) -> Result<()>,
        cancellation_requested: &dyn Fn() -> bool,
    ) -> Result<()> {
        self.process_ready_windows(processor, cancellation_requested)?;
        if self.buffered_samples.is_empty() {
            return Ok(());
        }
        if cancellation_requested() {
            return Err(cancelled());
        }
        let samples = std::mem::take(&mut self.buffered_samples);
        self.process_window(
            BoundedPcmWindow {
                start_sample: self.next_window_start_sample,
                samples,
                is_final_tail: true,
            },
            processor,
        )?;
        self.next_window_start_sample += self.config.hop_samples as u64;
        Ok(())
    }

    fn process_ready_windows(
        &mut self,
        processor: &mut dyn FnMut(BoundedPcmWindow) -> Result<()>,
        cancellation_requested: &dyn Fn() -> bool,
    ) -> Result<()> {
        while self.buffered_samples.len() >= self.config.window_samples {
            if cancellation_requested() {
                return Err(cancelled());
            }
            let samples = self.buffered_samples[..self.config.window_samples].to_vec();
            self.process_window(
                BoundedPcmWindow {
                    start_sample: self.next_window_start_sample,
                    samples,
                    is_final_tail: false,
                },
                processor,
            )?;
            let advance = self.config.hop_samples.min(self.buffered_samples.len());
            self.buffered_samples.drain(..advance);
            self.next_window_start_sample += self.config.hop_samples as u64;
        }
        Ok(())
    }

    fn process_window(
        &mut self,
        window: BoundedPcmWindow,
        processor: &mut dyn FnMut(BoundedPcmWindow) -> Result<()>,
    ) -> Result<()> {
        self.windows_processed += 1;
        self.window_samples_processed += window.samples.len() as u64;
        processor(window)
    }
}

fn validate_samples(samples: &[f32]) -> Result<()> {
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(DetectError::InvalidArgument(
            "PCM samples must be finite".to_string(),
        ));
    }
    Ok(())
}

fn invalid_window_config(message: &str) -> DetectError {
    DetectError::InvalidArgument(format!(
        "invalid bounded PCM window configuration: {message}"
    ))
}

fn cancelled() -> DetectError {
    DetectError::InvalidArgument(
        "bounded PCM window session cancelled at a safe window boundary".to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> BoundedPcmWindowConfig {
        BoundedPcmWindowConfig::new(10, 4, 2, 5).expect("valid configuration")
    }

    #[test]
    fn schedules_exact_and_overlapping_windows_with_a_final_tail() {
        let mut session = BoundedPcmWindowSession::new(config()).expect("session");
        let mut windows = Vec::new();
        let mut processor = |window| {
            windows.push(window);
            Ok(())
        };
        let active = || false;

        session
            .ingest(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], &mut processor, &active)
            .expect("ingestion");
        session.flush(&mut processor, &active).expect("flush");

        assert_eq!(
            windows,
            vec![
                BoundedPcmWindow {
                    start_sample: 0,
                    samples: vec![0.0, 1.0, 2.0, 3.0],
                    is_final_tail: false,
                },
                BoundedPcmWindow {
                    start_sample: 2,
                    samples: vec![2.0, 3.0, 4.0, 5.0],
                    is_final_tail: false,
                },
                BoundedPcmWindow {
                    start_sample: 4,
                    samples: vec![4.0, 5.0],
                    is_final_tail: true,
                },
            ]
        );
        assert_eq!(session.stats().buffered_samples, 0);
        assert_eq!(session.stats().windows_processed, 3);
        assert_eq!(session.stats().input_duration_seconds, 0.6);
    }

    #[test]
    fn never_retains_more_than_its_configured_bound() {
        let mut session = BoundedPcmWindowSession::new(config()).expect("session");
        let mut windows = 0;
        let mut processor = |_| {
            windows += 1;
            Ok(())
        };
        let active = || false;

        session
            .ingest(&vec![0.0; 100], &mut processor, &active)
            .expect("ingestion");

        assert!(windows > 0);
        assert!(session.stats().buffered_samples < config().window_samples);
    }

    #[test]
    fn rejects_invalid_configuration_and_non_finite_pcm() {
        assert!(BoundedPcmWindowConfig::new(0, 4, 2, 4).is_err());
        assert!(BoundedPcmWindowConfig::new(16_000, 4, 0, 4).is_err());
        assert!(BoundedPcmWindowConfig::new(16_000, 4, 5, 5).is_err());
        assert!(BoundedPcmWindowConfig::new(16_000, 4, 2, 3).is_err());

        let mut session = BoundedPcmWindowSession::new(config()).expect("session");
        let mut processor = |_| Ok(());
        let active = || false;
        assert!(session
            .ingest(&[0.0, f32::NAN], &mut processor, &active)
            .is_err());
        assert_eq!(session.stats().samples_ingested, 0);
    }

    #[test]
    fn cancellation_stops_before_the_next_window() {
        let mut session = BoundedPcmWindowSession::new(config()).expect("session");
        let mut windows = 0;
        let mut processor = |_| {
            windows += 1;
            Ok(())
        };
        let cancelled = std::cell::Cell::new(false);
        let cancellation_requested = || cancelled.get();

        session
            .ingest(
                &[0.0, 1.0, 2.0, 3.0],
                &mut processor,
                &cancellation_requested,
            )
            .expect("first window");
        cancelled.set(true);
        let error = session
            .ingest(&[4.0, 5.0], &mut processor, &cancellation_requested)
            .expect_err("second window must not run after cancellation");

        assert_eq!(windows, 1);
        assert!(error.to_string().contains("cancelled"));
    }
}
