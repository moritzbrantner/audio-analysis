//! Internal module support for FFI.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

use std::ffi::{c_char, c_float, c_int, c_void};

/// Variants describing whisper context.
pub enum whisper_context {}
/// Variants describing whisper state.
pub enum whisper_state {}

/// Type alias for whisper token.
pub type whisper_token = c_int;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Variants describing whisper alignment heads preset.
pub enum whisper_alignment_heads_preset {
    /// The whisper aheads none variant.
    WHISPER_AHEADS_NONE = -1,
    /// The whisper aheads n top most variant.
    WHISPER_AHEADS_N_TOP_MOST = 0,
    /// The whisper aheads custom variant.
    WHISPER_AHEADS_CUSTOM = 1,
    /// The whisper aheads tiny en variant.
    WHISPER_AHEADS_TINY_EN = 2,
    /// The whisper aheads tiny variant.
    WHISPER_AHEADS_TINY = 3,
    /// The whisper aheads base en variant.
    WHISPER_AHEADS_BASE_EN = 4,
    /// The whisper aheads base variant.
    WHISPER_AHEADS_BASE = 5,
    /// The whisper aheads small en variant.
    WHISPER_AHEADS_SMALL_EN = 6,
    /// The whisper aheads small variant.
    WHISPER_AHEADS_SMALL = 7,
    /// The whisper aheads medium en variant.
    WHISPER_AHEADS_MEDIUM_EN = 8,
    /// The whisper aheads medium variant.
    WHISPER_AHEADS_MEDIUM = 9,
    /// The whisper aheads large v1 variant.
    WHISPER_AHEADS_LARGE_V1 = 10,
    /// The whisper aheads large v2 variant.
    WHISPER_AHEADS_LARGE_V2 = 11,
    /// The whisper aheads large v3 variant.
    WHISPER_AHEADS_LARGE_V3 = 12,
    /// The whisper aheads large v3 turbo variant.
    WHISPER_AHEADS_LARGE_V3_TURBO = 13,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Data type for whisper ahead.
pub struct whisper_ahead {
    /// The n text layer value.
    pub n_text_layer: c_int,
    /// The n head value.
    pub n_head: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Data type for whisper aheads.
pub struct whisper_aheads {
    /// The n heads value.
    pub n_heads: usize,
    /// The heads value.
    pub heads: *const whisper_ahead,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Data type for whisper context params.
pub struct whisper_context_params {
    /// The use GPU value.
    pub use_gpu: bool,
    /// The flash attn value.
    pub flash_attn: bool,
    /// The GPU device value.
    pub gpu_device: c_int,
    /// The dtw token timestamps value.
    pub dtw_token_timestamps: bool,
    /// The dtw aheads preset value.
    pub dtw_aheads_preset: whisper_alignment_heads_preset,
    /// The dtw n top value.
    pub dtw_n_top: c_int,
    /// The dtw aheads value.
    pub dtw_aheads: whisper_aheads,
    /// The dtw mem size value.
    pub dtw_mem_size: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Data type for whisper token data.
pub struct whisper_token_data {
    /// Identifier for this value.
    pub id: whisper_token,
    /// The tid value.
    pub tid: whisper_token,
    /// The p value.
    pub p: c_float,
    /// The plog value.
    pub plog: c_float,
    /// The pt value.
    pub pt: c_float,
    /// The ptsum value.
    pub ptsum: c_float,
    /// The t0 value.
    pub t0: i64,
    /// The t1 value.
    pub t1: i64,
    /// The t dtw value.
    pub t_dtw: i64,
    /// The vlen value.
    pub vlen: c_float,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Variants describing whisper gretype.
pub enum whisper_gretype {
    /// The whisper gretype end variant.
    WHISPER_GRETYPE_END = 0,
    /// The whisper gretype alt variant.
    WHISPER_GRETYPE_ALT = 1,
    /// The whisper gretype rule ref variant.
    WHISPER_GRETYPE_RULE_REF = 2,
    /// The whisper gretype char variant.
    WHISPER_GRETYPE_CHAR = 3,
    /// The whisper gretype char not variant.
    WHISPER_GRETYPE_CHAR_NOT = 4,
    /// The whisper gretype char rng upper variant.
    WHISPER_GRETYPE_CHAR_RNG_UPPER = 5,
    /// The whisper gretype char alt variant.
    WHISPER_GRETYPE_CHAR_ALT = 6,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Data type for whisper grammar element.
pub struct whisper_grammar_element {
    /// The type value.
    pub type_: whisper_gretype,
    /// The value value.
    pub value: u32,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Data type for whisper vad params.
pub struct whisper_vad_params {
    /// The threshold value.
    pub threshold: c_float,
    /// The min speech duration ms value.
    pub min_speech_duration_ms: c_int,
    /// The min silence duration ms value.
    pub min_silence_duration_ms: c_int,
    /// The max speech duration s value.
    pub max_speech_duration_s: c_float,
    /// The speech pad ms value.
    pub speech_pad_ms: c_int,
    /// The samples overlap value.
    pub samples_overlap: c_float,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
/// Variants describing whisper sampling strategy.
pub enum whisper_sampling_strategy {
    /// The whisper sampling greedy variant.
    WHISPER_SAMPLING_GREEDY = 0,
    /// The whisper sampling beam search variant.
    WHISPER_SAMPLING_BEAM_SEARCH = 1,
}

/// Type alias for whisper new segment callback.
pub type whisper_new_segment_callback =
    Option<unsafe extern "C" fn(*mut whisper_context, *mut whisper_state, c_int, *mut c_void)>;
/// Type alias for whisper progress callback.
pub type whisper_progress_callback =
    Option<unsafe extern "C" fn(*mut whisper_context, *mut whisper_state, c_int, *mut c_void)>;
/// Type alias for whisper encoder begin callback.
pub type whisper_encoder_begin_callback =
    Option<unsafe extern "C" fn(*mut whisper_context, *mut whisper_state, *mut c_void) -> bool>;
/// Type alias for ggml abort callback.
pub type ggml_abort_callback = Option<unsafe extern "C" fn(*mut c_void) -> bool>;
/// Type alias for whisper logits filter callback.
pub type whisper_logits_filter_callback = Option<
    unsafe extern "C" fn(
        *mut whisper_context,
        *mut whisper_state,
        *const whisper_token_data,
        c_int,
        *mut c_float,
        *mut c_void,
    ),
>;

#[repr(C)]
#[derive(Clone, Copy)]
/// Data type for whisper full params greedy.
pub struct whisper_full_params_greedy {
    /// The best of value.
    pub best_of: c_int,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Data type for whisper full params beam search.
pub struct whisper_full_params_beam_search {
    /// The beam size value.
    pub beam_size: c_int,
    /// The patience value.
    pub patience: c_float,
}

#[repr(C)]
#[derive(Clone, Copy)]
/// Data type for whisper full params.
pub struct whisper_full_params {
    /// The strategy value.
    pub strategy: whisper_sampling_strategy,
    /// The n threads value.
    pub n_threads: c_int,
    /// The n max text ctx value.
    pub n_max_text_ctx: c_int,
    /// The offset ms value.
    pub offset_ms: c_int,
    /// The duration ms value.
    pub duration_ms: c_int,
    /// The translate value.
    pub translate: bool,
    /// The no context value.
    pub no_context: bool,
    /// The no timestamps value.
    pub no_timestamps: bool,
    /// The single segment value.
    pub single_segment: bool,
    /// The print special value.
    pub print_special: bool,
    /// The print progress value.
    pub print_progress: bool,
    /// The print realtime value.
    pub print_realtime: bool,
    /// The print timestamps value.
    pub print_timestamps: bool,
    /// The token timestamps value.
    pub token_timestamps: bool,
    /// The thold pt value.
    pub thold_pt: c_float,
    /// The thold ptsum value.
    pub thold_ptsum: c_float,
    /// The max len value.
    pub max_len: c_int,
    /// The split on word value.
    pub split_on_word: bool,
    /// The max tokens value.
    pub max_tokens: c_int,
    /// The debug mode value.
    pub debug_mode: bool,
    /// The audio ctx value.
    pub audio_ctx: c_int,
    /// The tdrz enable value.
    pub tdrz_enable: bool,
    /// The suppress regex value.
    pub suppress_regex: *const c_char,
    /// The initial prompt value.
    pub initial_prompt: *const c_char,
    /// The carry initial prompt value.
    pub carry_initial_prompt: bool,
    /// The prompt tokens value.
    pub prompt_tokens: *const whisper_token,
    /// The prompt n tokens value.
    pub prompt_n_tokens: c_int,
    /// Language tag for this value.
    pub language: *const c_char,
    /// The detect language value.
    pub detect_language: bool,
    /// The suppress blank value.
    pub suppress_blank: bool,
    /// The suppress nst value.
    pub suppress_nst: bool,
    /// The temperature value.
    pub temperature: c_float,
    /// The max initial ts value.
    pub max_initial_ts: c_float,
    /// The length penalty value.
    pub length_penalty: c_float,
    /// The temperature inc value.
    pub temperature_inc: c_float,
    /// The entropy thold value.
    pub entropy_thold: c_float,
    /// The logprob thold value.
    pub logprob_thold: c_float,
    /// The no speech thold value.
    pub no_speech_thold: c_float,
    /// The greedy value.
    pub greedy: whisper_full_params_greedy,
    /// The beam search value.
    pub beam_search: whisper_full_params_beam_search,
    /// The new segment callback value.
    pub new_segment_callback: whisper_new_segment_callback,
    /// The new segment callback user data value.
    pub new_segment_callback_user_data: *mut c_void,
    /// The progress callback value.
    pub progress_callback: whisper_progress_callback,
    /// The progress callback user data value.
    pub progress_callback_user_data: *mut c_void,
    /// The encoder begin callback value.
    pub encoder_begin_callback: whisper_encoder_begin_callback,
    /// The encoder begin callback user data value.
    pub encoder_begin_callback_user_data: *mut c_void,
    /// The abort callback value.
    pub abort_callback: ggml_abort_callback,
    /// The abort callback user data value.
    pub abort_callback_user_data: *mut c_void,
    /// The logits filter callback value.
    pub logits_filter_callback: whisper_logits_filter_callback,
    /// The logits filter callback user data value.
    pub logits_filter_callback_user_data: *mut c_void,
    /// The grammar rules value.
    pub grammar_rules: *const *const whisper_grammar_element,
    /// The n grammar rules value.
    pub n_grammar_rules: usize,
    /// The i start rule value.
    pub i_start_rule: usize,
    /// The grammar penalty value.
    pub grammar_penalty: c_float,
    /// The vad value.
    pub vad: bool,
    /// The vad model path value.
    pub vad_model_path: *const c_char,
    /// The vad params value.
    pub vad_params: whisper_vad_params,
}

unsafe extern "C" {
    /// Returns whisper context default params.
    pub fn whisper_context_default_params() -> whisper_context_params;
    /// Returns whisper full default params.
    pub fn whisper_full_default_params(strategy: whisper_sampling_strategy) -> whisper_full_params;
    /// Returns whisper init from file with params.
    pub fn whisper_init_from_file_with_params(
        path_model: *const c_char,
        params: whisper_context_params,
    ) -> *mut whisper_context;
    /// Returns whisper free.
    pub fn whisper_free(ctx: *mut whisper_context);
    /// Returns whisper full.
    pub fn whisper_full(
        ctx: *mut whisper_context,
        params: whisper_full_params,
        samples: *const c_float,
        n_samples: c_int,
    ) -> c_int;
    /// Returns whisper full n segments.
    pub fn whisper_full_n_segments(ctx: *mut whisper_context) -> c_int;
    /// Returns whisper full lang identifier.
    pub fn whisper_full_lang_id(ctx: *mut whisper_context) -> c_int;
    /// Returns whisper full get segment t0.
    pub fn whisper_full_get_segment_t0(ctx: *mut whisper_context, i_segment: c_int) -> i64;
    /// Returns whisper full get segment t1.
    pub fn whisper_full_get_segment_t1(ctx: *mut whisper_context, i_segment: c_int) -> i64;
    /// Returns whisper full get segment text.
    pub fn whisper_full_get_segment_text(
        ctx: *mut whisper_context,
        i_segment: c_int,
    ) -> *const c_char;
    /// Returns whisper full n tokens.
    pub fn whisper_full_n_tokens(ctx: *mut whisper_context, i_segment: c_int) -> c_int;
    /// Returns whisper full get token p.
    pub fn whisper_full_get_token_p(
        ctx: *mut whisper_context,
        i_segment: c_int,
        i_token: c_int,
    ) -> c_float;
    /// Returns whisper lang identifier.
    pub fn whisper_lang_id(lang: *const c_char) -> c_int;
    /// Returns whisper lang str.
    pub fn whisper_lang_str(id: c_int) -> *const c_char;
    /// Returns whisper print system info.
    pub fn whisper_print_system_info() -> *const c_char;
}
