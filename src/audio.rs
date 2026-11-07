use std::f64::consts::{FRAC_1_SQRT_2, PI, TAU};
use std::fs;
use std::io;
use std::path::Path;

use crate::fluid::FluidTelemetryFrame;
use crate::rng::Rng;

pub const SAMPLE_RATE: u32 = 48_000;
pub const DURATION_SECONDS: usize = 15;
pub const TOTAL_FRAMES: usize = (SAMPLE_RATE as usize) * DURATION_SECONDS;
pub const NUM_CHANNELS: u16 = 2;
pub const BITS_PER_SAMPLE: u16 = 24;

// ============================================================================
// 4-Pole Topology-Preserving Transform State-Variable Filter (Zavalishin ZDF)
// ============================================================================

#[derive(Clone, Copy, Debug, Default)]
pub struct TptSvf2Pole {
    ic1eq: f64,
    ic2eq: f64,
}

impl TptSvf2Pole {
    /// Process one sample and return `(lowpass, bandpass, highpass)` simultaneously.
    #[inline]
    pub fn process(&mut self, input: f64, cutoff_hz: f64, resonance: f64, fs: f64) -> (f64, f64, f64) {
        let fc = cutoff_hz.clamp(20.0, 0.47 * fs);
        let g = (PI * fc / fs).tan();
        // `resonance` in [0.0, 0.97] -> damping factor k in [2.0, 0.06]
        let k = 2.0 * (1.0 - resonance.clamp(0.0, 0.97));
        let a1 = 1.0 / (1.0 + g * (g + k));
        let a2 = g * a1;
        let a3 = g * a2;

        let v3 = input - self.ic2eq;
        let v1 = a1 * self.ic1eq + a2 * v3;
        let v2 = self.ic2eq + a2 * self.ic1eq + a3 * v3;

        self.ic1eq = 2.0 * v1 - self.ic1eq;
        self.ic2eq = 2.0 * v2 - self.ic2eq;

        let lp = v2;
        let bp = v1;
        let hp = input - k * v1 - v2;
        (lp, bp, hp)
    }
}

/// True 24 dB/oct 4-Pole TPT State-Variable Filter (two cascaded ZDF 2-pole stages with subtle analog saturation).
#[derive(Clone, Copy, Debug, Default)]
pub struct TptSvf4Pole {
    stage1: TptSvf2Pole,
    stage2: TptSvf2Pole,
}

impl TptSvf4Pole {
    #[inline]
    pub fn process_lowpass(&mut self, input: f64, cutoff_hz: f64, resonance: f64, fs: f64) -> f64 {
        let (lp1, _, _) = self.stage1.process(input, cutoff_hz, resonance * 0.88, fs);
        let driven = (lp1 * 1.08).tanh() * (1.0 / 1.08);
        let (lp2, _, _) = self.stage2.process(driven, cutoff_hz, resonance, fs);
        lp2
    }

    #[inline]
    pub fn process_bandpass(&mut self, input: f64, cutoff_hz: f64, resonance: f64, fs: f64) -> f64 {
        let (_, bp1, _) = self.stage1.process(input, cutoff_hz, resonance * 0.85, fs);
        let (_, bp2, _) = self.stage2.process(bp1, cutoff_hz, resonance, fs);
        bp2
    }
}

// ============================================================================
// 8-Line Orthogonal Hadamard Feedback Delay Network (FDN) Reverb
// ============================================================================

/// Fast in-place normalized 8x8 Sylvester-Hadamard transform (`y = (H_8 / sqrt(8)) * x`).
/// Strictly orthogonal (`||y||_2 == ||x||_2`).
#[inline]
pub fn hadamard8_normalized(x: &mut [f64; 8]) {
    // Stage 1 (stride 1)
    for i in (0..8).step_by(2) {
        let a = x[i];
        let b = x[i + 1];
        x[i] = a + b;
        x[i + 1] = a - b;
    }
    // Stage 2 (stride 2)
    for base in (0..8).step_by(4) {
        for j in 0..2 {
            let a = x[base + j];
            let b = x[base + j + 2];
            x[base + j] = a + b;
            x[base + j + 2] = a - b;
        }
    }
    // Stage 3 (stride 4)
    for j in 0..4 {
        let a = x[j];
        let b = x[j + 4];
        x[j] = a + b;
        x[j + 4] = a - b;
    }

    const INV_SQRT8: f64 = 0.5 * FRAC_1_SQRT_2; // 1 / sqrt(8) = 1 / (2 * sqrt(2))
    for v in x.iter_mut() {
        *v *= INV_SQRT8;
    }
}

pub struct HadamardFdn8 {
    buffers: [Vec<f64>; 8],
    base_delays: [usize; 8],
    write_pos: [usize; 8],
    damp_state: [f64; 8],
    gain_dc: [f64; 8],
    damp_coeff: [f64; 8],
    lfo_phase: [f64; 8],
    lfo_inc: [f64; 8],
}

impl HadamardFdn8 {
    pub fn new(fs: f64, rt60_low: f64, rt60_high: f64) -> Self {
        // 8 mutually prime delay lengths spanning ~31ms to ~97ms at 48kHz
        let base_delays: [usize; 8] = [1499, 1877, 2267, 2683, 3119, 3583, 4079, 4637];
        let extra_margin = 64usize;

        let mut gain_dc = [0.0; 8];
        let mut damp_coeff = [0.0; 8];
        let mut lfo_phase = [0.0; 8];
        let mut lfo_inc = [0.0; 8];

        for i in 0..8 {
            let d_sec = (base_delays[i] as f64) / fs;
            let g_low = 10.0_f64.powf(-3.0 * d_sec / rt60_low.max(0.2));
            let g_high = 10.0_f64.powf(-3.0 * d_sec / rt60_high.max(0.1));
            gain_dc[i] = g_low.min(0.992);
            // One-pole absorption coefficient smoothing high frequencies toward g_high
            damp_coeff[i] = (1.0 - (g_high / g_low.max(1e-6))).clamp(0.08, 0.72);
            lfo_phase[i] = (i as f64) * (TAU / 8.0);
            lfo_inc[i] = TAU * (0.21 + 0.07 * (i as f64)) / fs;
        }

        let buffers = std::array::from_fn(|i| vec![0.0; base_delays[i] + extra_margin]);
        Self {
            buffers,
            base_delays,
            write_pos: [0; 8],
            damp_state: [0.0; 8],
            gain_dc,
            damp_coeff,
            lfo_phase,
            lfo_inc,
        }
    }

    #[inline]
    pub fn process_stereo(&mut self, in_l: f64, in_r: f64) -> (f64, f64) {
        let mut delayed = [0.0; 8];

        for i in 0..8 {
            let buf = &self.buffers[i];
            let len = buf.len();
            // Subtle chorus-like delay modulation (+/- 6.5 samples) breaks up modal metallic ringing
            let mod_samples = 6.5 * self.lfo_phase[i].sin();
            self.lfo_phase[i] += self.lfo_inc[i];
            if self.lfo_phase[i] >= TAU {
                self.lfo_phase[i] -= TAU;
            }

            let read_offset = (self.base_delays[i] as f64) + mod_samples;
            let read_f = (self.write_pos[i] as f64) + (len as f64) - read_offset;
            let r0 = (read_f.floor() as usize) % len;
            let r1 = (r0 + 1) % len;
            let frac = read_f - read_f.floor();
            let raw = buf[r0] * (1.0 - frac) + buf[r1] * frac;

            // Per-line one-pole high-frequency absorption + RT60 decay gain
            let lp = raw * (1.0 - self.damp_coeff[i]) + self.damp_state[i] * self.damp_coeff[i];
            self.damp_state[i] = lp;
            delayed[i] = lp * self.gain_dc[i];
        }

        // Orthogonal 8x8 Hadamard mixing matrix
        let mut mixed = delayed;
        hadamard8_normalized(&mut mixed);

        // Decorrelated stereo input injection into all 8 delay lines
        let inputs = [
            in_l,
            in_r,
            0.7 * (in_l + in_r),
            0.7 * (in_l - in_r),
            -in_l * 0.8,
            in_r * 0.8,
            -0.6 * (in_l - in_r),
            0.6 * (in_l + in_r),
        ];

        for i in 0..8 {
            let wp = self.write_pos[i];
            self.buffers[i][wp] = mixed[i] + inputs[i] * 0.35;
            self.write_pos[i] = (wp + 1) % self.buffers[i].len();
        }

        // Decorrelated stereo output tap combination
        let wet_l = 0.38 * (delayed[0] - delayed[2] + delayed[4] - delayed[6]);
        let wet_r = 0.38 * (delayed[1] - delayed[3] + delayed[5] - delayed[7]);
        (wet_l, wet_r)
    }
}

// ============================================================================
// Polyphonic 4-Operator FM Bell Voice + Binaural ITD Spatializer
// ============================================================================

#[derive(Clone, Copy, Debug)]
struct FmBellVoice {
    active: bool,
    freq_hz: f64,
    age_sec: f64,
    duration_sec: f64,
    velocity: f64,
    pan: f64, // [-1.0 Left .. +1.0 Right]
    phase_c1: f64,
    phase_m1: f64,
    phase_c2: f64,
    phase_m2: f64,
    prev_m1: f64,
}

impl Default for FmBellVoice {
    fn default() -> Self {
        Self {
            active: false,
            freq_hz: 440.0,
            age_sec: 0.0,
            duration_sec: 3.5,
            velocity: 0.0,
            pan: 0.0,
            phase_c1: 0.0,
            phase_m1: 0.0,
            phase_c2: 0.0,
            phase_m2: 0.0,
            prev_m1: 0.0,
        }
    }
}

impl FmBellVoice {
    fn trigger(&mut self, freq_hz: f64, velocity: f64, pan: f64, duration_sec: f64) {
        self.active = true;
        self.freq_hz = freq_hz;
        self.age_sec = 0.0;
        self.duration_sec = duration_sec;
        self.velocity = velocity;
        self.pan = pan.clamp(-0.95, 0.95);
        self.phase_c1 = 0.0;
        self.phase_m1 = 0.0;
        self.phase_c2 = 0.0;
        self.phase_m2 = 0.0;
        self.prev_m1 = 0.0;
    }

    #[inline]
    fn render_sample(&mut self, dt: f64, fluid_ke_norm: f64) -> (f64, f64) {
        if !self.active {
            return (0.0, 0.0);
        }
        let t = self.age_sec;
        if t >= self.duration_sec {
            self.active = false;
            return (0.0, 0.0);
        }

        // Fast crystalline 2ms percussive attack + multi-rate exponential bell decay
        let attack = (t / 0.0025).min(1.0);
        let env_body = attack * (-t * 1.45).exp();
        let env_mod1 = (-t * 4.8).exp();
        let env_chime = attack * (-t * 3.1).exp();
        let env_mod2 = (-t * 11.5).exp();
        let tail_gate = ((self.duration_sec - t) / 0.06).clamp(0.0, 1.0);

        // Fluid kinetic energy expands FM modulation index for metallic harmonic brilliance
        let idx1 = (1.35 + 2.45 * fluid_ke_norm) * env_mod1;
        let idx2 = (0.85 + 2.10 * fluid_ke_norm) * env_mod2;

        // Branch 1: Tubular harmonic bell body (1 : 2.0007 + self-feedback)
        let m1 = (self.phase_m1 + 0.22 * self.prev_m1).sin();
        self.prev_m1 = m1;
        let c1 = (self.phase_c1 + idx1 * m1).sin() * env_body;

        // Branch 2: Inharmonic glassy strike partial (2.001 : 5.4142)
        let m2 = self.phase_m2.sin();
        let c2 = (self.phase_c2 + idx2 * m2).sin() * env_chime * 0.36;

        self.phase_c1 = (self.phase_c1 + TAU * self.freq_hz * dt) % TAU;
        self.phase_m1 = (self.phase_m1 + TAU * (self.freq_hz * 2.0007) * dt) % TAU;
        self.phase_c2 = (self.phase_c2 + TAU * (self.freq_hz * 2.0015) * dt) % TAU;
        self.phase_m2 = (self.phase_m2 + TAU * (self.freq_hz * 5.4142) * dt) % TAU;
        self.age_sec += dt;

        let mono = (c1 + c2) * self.velocity * tail_gate;
        // Equal-power constant-intensity stereo pan law
        let angle = (self.pan + 1.0) * (PI * 0.25);
        (mono * angle.cos(), mono * angle.sin())
    }
}

/// Short fractional delay line for binaural Interaural Time Difference (ITD) spatialization.
struct MicroDelay {
    buf: [f64; 128],
    wp: usize,
}

impl MicroDelay {
    fn new() -> Self {
        Self {
            buf: [0.0; 128],
            wp: 0,
        }
    }

    #[inline]
    fn process(&mut self, x: f64, delay_samples: f64) -> f64 {
        self.buf[self.wp] = x;
        let d = delay_samples.clamp(0.0, 64.0);
        let rf = (self.wp as f64) + 128.0 - d;
        let r0 = (rf.floor() as usize) & 127;
        let r1 = (r0 + 1) & 127;
        let frac = rf - rf.floor();
        self.wp = (self.wp + 1) & 127;
        self.buf[r0] * (1.0 - frac) + self.buf[r1] * frac
    }
}

#[inline]
fn midi_to_hz(midi: f64) -> f64 {
    440.0 * 2.0_f64.powf((midi - 69.0) / 12.0)
}

#[inline]
fn poly_blep(t: f64, dt: f64) -> f64 {
    if t < dt {
        let x = t / dt;
        x + x - x * x - 1.0
    } else if t > 1.0 - dt {
        let x = (t - 1.0) / dt;
        x * x + x + x + 1.0
    } else {
        0.0
    }
}

/// Interpolate normalized `(ke_norm, vort_norm, pan_x, height_y)` from fluid simulation telemetry at `t_sec`.
fn sample_fluid_curves(
    frames: &[FluidTelemetryFrame],
    t_sec: f64,
    max_ke: f64,
    max_vort: f64,
) -> (f64, f64, f64, f64) {
    if frames.is_empty() {
        return (0.5, 0.5, 0.0, 0.0);
    }
    if frames.len() == 1 {
        let f = &frames[0];
        return (
            (f.kinetic_energy / max_ke.max(1e-9)).clamp(0.0, 1.0),
            (f.peak_vorticity / max_vort.max(1e-9)).clamp(0.0, 1.0),
            f.center_of_mass.x.clamp(-1.0, 1.0),
            (0.5 * (f.center_of_mass.y + 1.0)).clamp(0.0, 1.0),
        );
    }
    let u = (t_sec / (DURATION_SECONDS as f64)).clamp(0.0, 1.0) * ((frames.len() - 1) as f64);
    let i0 = (u.floor() as usize).min(frames.len() - 1);
    let i1 = (i0 + 1).min(frames.len() - 1);
    let frac = u - (i0 as f64);

    let a = &frames[i0];
    let b = &frames[i1];
    let ke = a.kinetic_energy * (1.0 - frac) + b.kinetic_energy * frac;
    let vort = a.peak_vorticity * (1.0 - frac) + b.peak_vorticity * frac;
    let cx = a.center_of_mass.x * (1.0 - frac) + b.center_of_mass.x * frac;
    let cy = a.center_of_mass.y * (1.0 - frac) + b.center_of_mass.y * frac;

    (
        (ke / max_ke.max(1e-9)).clamp(0.0, 1.0),
        (vort / max_vort.max(1e-9)).clamp(0.0, 1.0),
        cx.clamp(-1.0, 1.0),
        (0.5 * (cy + 1.0)).clamp(0.0, 1.0),
    )
}

/// Synthesize a complete 15-second 48kHz stereo soundtrack coupled to the 3D fluid kinetic energy & vorticity telemetry.
pub fn synthesize_fluid_soundtrack(telemetry: &[FluidTelemetryFrame]) -> Vec<(f64, f64)> {
    let fs = SAMPLE_RATE as f64;
    let dt = 1.0 / fs;

    let mut max_ke = 1e-6_f64;
    let mut max_vort = 1e-6_f64;
    for f in telemetry {
        max_ke = max_ke.max(f.kinetic_energy);
        max_vort = max_vort.max(f.peak_vorticity);
    }

    let mut voices = [FmBellVoice::default(); 16];
    let mut pad_svf_l = TptSvf4Pole::default();
    let mut pad_svf_r = TptSvf4Pole::default();
    let mut breath_svf_l = TptSvf4Pole::default();
    let mut breath_svf_r = TptSvf4Pole::default();
    let mut fdn_reverb = HadamardFdn8::new(fs, 3.85, 1.35);
    let mut itd_left = MicroDelay::new();
    let mut itd_right = MicroDelay::new();
    let mut rng = Rng::from_seed(0xAE74_E9C0_2026);

    // 4 Cyberpunk Blade-Runner-esque harmonic voicings across 15.0 seconds (3.75s per chord):
    //   0..3.75s: D minor add9   (D2, A2, F3, A3, C4, E4, A4, D5)
    //   3.75..7.5s: Bb Maj9#11   (Bb1, F2, D3, A3, C4, E4, F4, A4)
    //   7.5..11.25s: G minor 11  (G1, D2, Bb2, F3, A3, C4, D4, G4)
    //   11.25..15s: A7sus4 -> Dm9 (A1, E2, G3, C4, D4, E4, A4, E5)
    let chord_pads: [[f64; 4]; 4] = [
        [38.0, 57.0, 60.0, 64.0], // D2, A3, C4, E4
        [34.0, 53.0, 57.0, 64.0], // Bb1, F3, A3, E4 (#11)
        [31.0, 50.0, 58.0, 62.0], // G1, D3, Bb3, D4
        [33.0, 52.0, 57.0, 64.0], // A1, E3, A3, E4
    ];
    let bell_scales: [[f64; 8]; 4] = [
        [62.0, 64.0, 65.0, 69.0, 72.0, 76.0, 79.0, 81.0],
        [58.0, 62.0, 64.0, 65.0, 69.0, 72.0, 76.0, 77.0],
        [55.0, 58.0, 62.0, 65.0, 69.0, 70.0, 74.0, 79.0],
        [57.0, 62.0, 64.0, 67.0, 69.0, 74.0, 76.0, 81.0],
    ];

    let mut pad_phases = [[0.0_f64; 3]; 4];
    let detunes = [0.9955, 1.0, 1.0048];
    let mut sub_phase = 0.0_f64;

    let mut smooth_ke = 0.25_f64;
    let mut smooth_vort = 0.25_f64;
    let mut prev_ke = 0.25_f64;
    let mut next_grid_step = 0usize;
    let mut flux_cooldown = 0.0_f64;
    let mut step_counter = 0usize;

    let step_interval_frames = (SAMPLE_RATE as usize) * 3 / 16; // 187.5ms (16th-note pulse at 80 BPM)
    let mut stereo_out = Vec::with_capacity(TOTAL_FRAMES);

    for frame_idx in 0..TOTAL_FRAMES {
        let t_sec = (frame_idx as f64) * dt;
        let chord_idx = ((t_sec / 3.75).floor() as usize).min(3);

        let (raw_ke, raw_vort, com_x, com_y) =
            sample_fluid_curves(telemetry, t_sec, max_ke, max_vort);
        smooth_ke += 0.0015 * (raw_ke - smooth_ke);
        smooth_vort += 0.0015 * (raw_vort - smooth_vort);
        let ke_flux = (smooth_ke - prev_ke) * fs;
        prev_ke = smooth_ke;

        if flux_cooldown > 0.0 {
            flux_cooldown -= dt;
        }

        // Trigger crystalline FM bells on musical grid + kinetic-energy turbulence spikes
        let grid_hit = frame_idx >= next_grid_step && t_sec < 14.15;
        let flux_hit = ke_flux > 0.22 && flux_cooldown <= 0.0 && t_sec > 0.4 && t_sec < 14.0;
        if grid_hit || flux_hit {
            if grid_hit {
                next_grid_step += step_interval_frames;
            }
            if flux_hit {
                flux_cooldown = 0.14;
            }

            // Melodic note selection steered by fluid plume elevation `com_y` and harmonic step pattern
            let scale = &bell_scales[chord_idx];
            let pattern_offset = [0usize, 3, 4, 2, 6, 4, 5, 7][step_counter & 7];
            let fluid_offset = ((com_y * 3.0).round() as usize).min(3);
            let degree = (pattern_offset + fluid_offset) & 7;
            let midi = scale[degree];
            let vel = (0.16 + 0.18 * smooth_ke + if flux_hit { 0.06 } else { 0.0 }).min(0.36);

            // Spatialize bell using fluid plume horizontal center-of-mass + alternating stereo spread
            let spread = if (step_counter & 1) == 0 { -0.45 } else { 0.45 };
            let pan = (0.65 * com_x * 3.0 + spread).clamp(-0.92, 0.92);

            // Allocate free voice or steal oldest voice
            let mut slot = 0usize;
            let mut best_age = -1.0_f64;
            for (vi, v) in voices.iter().enumerate() {
                if !v.active {
                    slot = vi;
                    break;
                }
                if v.age_sec > best_age {
                    best_age = v.age_sec;
                    slot = vi;
                }
            }
            voices[slot].trigger(midi_to_hz(midi), vel, pan, 3.1 + 1.2 * smooth_ke);
            step_counter += 1;
        }

        // 1. Render polyphonic FM Bell voices
        let mut bells_l = 0.0;
        let mut bells_r = 0.0;
        for v in &mut voices {
            let (vl, vr) = v.render_sample(dt, smooth_ke);
            bells_l += vl;
            bells_r += vr;
        }

        // 2. Render Bandlimited PolyBLEP Detuned Analog Pad + Sub-Bass sculpted by 4-Pole TPT SVF
        let chord = &chord_pads[chord_idx];
        let mut pad_raw_l = 0.0;
        let mut pad_raw_r = 0.0;
        for n_idx in 0..4 {
            let base_f = midi_to_hz(chord[n_idx]);
            for d_idx in 0..3 {
                let f = base_f * detunes[d_idx];
                let p_inc = f * dt;
                let ph = pad_phases[n_idx][d_idx];
                let saw = 2.0 * ph - 1.0 - poly_blep(ph, p_inc);
                pad_phases[n_idx][d_idx] = (ph + p_inc).fract();
                if d_idx == 0 {
                    pad_raw_l += saw * 0.095;
                    pad_raw_r += saw * 0.035;
                } else if d_idx == 2 {
                    pad_raw_l += saw * 0.035;
                    pad_raw_r += saw * 0.095;
                } else {
                    pad_raw_l += saw * 0.065;
                    pad_raw_r += saw * 0.065;
                }
            }
        }

        // Warm fundamental sine sub-bass anchored to chord root
        let sub_f = midi_to_hz(chord[0]);
        let sub_sig = (TAU * sub_phase).sin() * (0.19 + 0.08 * smooth_ke);
        sub_phase = (sub_phase + sub_f * dt).fract();

        // 4-Pole TPT SVF cutoff & resonance driven directly by Navier-Stokes kinetic energy & vorticity
        let lfo_breath = 0.5 + 0.5 * (TAU * 0.18 * t_sec).sin();
        let cutoff_hz = 185.0 * 2.0_f64.powf(3.9 * smooth_ke + 0.55 * lfo_breath);
        let resonance = (0.24 + 0.48 * smooth_vort).clamp(0.15, 0.78);

        let pad_l = pad_svf_l.process_lowpass(pad_raw_l, cutoff_hz, resonance, fs) + sub_sig;
        let pad_r = pad_svf_r.process_lowpass(pad_raw_r, cutoff_hz * 1.03, resonance, fs) + sub_sig;

        // 3. Turbulent Vortex Shimmer / Atmospheric Air Layer (4-Pole TPT Bandpass SVF)
        let white_l = 2.0 * rng.next_f64() - 1.0;
        let white_r = 2.0 * rng.next_f64() - 1.0;
        let air_fc = 620.0 + 2400.0 * com_y + 1200.0 * smooth_vort;
        let air_amp = 0.022 * smooth_vort;
        let air_l = breath_svf_l.process_bandpass(white_l, air_fc, 0.68, fs) * air_amp;
        let air_r = breath_svf_r.process_bandpass(white_r, air_fc * 1.06, 0.68, fs) * air_amp;

        // 4. Apply Binaural ITD (Interaural Time Difference) spatialization to fluid-coupled layers
        let azimuth = (com_x * 2.5).clamp(-1.0, 1.0);
        let delay_l = if azimuth > 0.0 { azimuth * 24.0 } else { 0.0 };
        let delay_r = if azimuth < 0.0 { -azimuth * 24.0 } else { 0.0 };

        let dry_l = itd_left.process(bells_l + air_l, delay_l) + pad_l * 0.78;
        let dry_r = itd_right.process(bells_r + air_r, delay_r) + pad_r * 0.78;

        // 5. Route into 8-Line Orthogonal Hadamard FDN Cathedral Reverb
        let rev_send_l = bells_l * 0.72 + pad_l * 0.42 + air_l * 0.85;
        let rev_send_r = bells_r * 0.72 + pad_r * 0.42 + air_r * 0.85;
        let (wet_l, wet_r) = fdn_reverb.process_stereo(rev_send_l, rev_send_r);

        // Master fade-in (80ms) and smooth cinematic fade-out (last 650ms) + soft-knee mastering limiter
        let fade_in = (t_sec / 0.08).clamp(0.0, 1.0);
        let fade_out = if t_sec > 14.35 {
            let u = ((15.0 - t_sec) / 0.65).clamp(0.0, 1.0);
            0.5 - 0.5 * (PI * u).cos()
        } else {
            1.0
        };
        let master_env = fade_in * fade_out;

        let mix_l = ((dry_l * 0.74 + wet_l * 0.52) * master_env * 0.92).tanh() * 0.91;
        let mix_r = ((dry_r * 0.74 + wet_r * 0.52) * master_env * 0.92).tanh() * 0.91;
        stereo_out.push((mix_l, mix_r));
    }

    stereo_out
}

// ============================================================================
// 24-Bit Signed Little-Endian PCM RIFF/WAVE Encoder & Binary Verifier
// ============================================================================

pub fn encode_wav_24bit_stereo(sample_rate: u32, frames: &[(f64, f64)]) -> Vec<u8> {
    let bytes_per_sample = (BITS_PER_SAMPLE / 8) as u32; // 3 bytes
    let block_align = (NUM_CHANNELS as u32) * bytes_per_sample; // 6 bytes/frame
    let byte_rate = sample_rate * block_align; // 288,000 bytes/sec
    let data_bytes = (frames.len() as u32) * block_align;
    let riff_chunk_size = 36 + data_bytes;

    let mut out = Vec::with_capacity((44 + data_bytes) as usize);
    // RIFF WAVE Header (44 bytes)
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // `fmt ` subchunk (16 bytes payload, PCM format = 1)
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // WAVE_FORMAT_PCM = 1
    out.extend_from_slice(&NUM_CHANNELS.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&(block_align as u16).to_le_bytes());
    out.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());

    // `data` subchunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());

    // Quantize [-1.0, 1.0] to 24-bit signed PCM [-8388608, 8388607] with TPDF triangular dithering
    let mut dither_rng = Rng::from_seed(0x24B1_7DA7_A202);
    const FULL_SCALE_24: f64 = 8_388_607.0;

    for &(l, r) in frames {
        for s in [l, r] {
            let tpdf = (dither_rng.next_f64() - dither_rng.next_f64()) * 0.5;
            let scaled = (s.clamp(-1.0, 1.0) * FULL_SCALE_24 + tpdf)
                .round()
                .clamp(-8_388_608.0, 8_388_607.0) as i32;
            let le = scaled.to_le_bytes();
            out.push(le[0]);
            out.push(le[1]);
            out.push(le[2]);
        }
    }
    out
}

pub fn save_wav_24bit_stereo(
    path: &Path,
    sample_rate: u32,
    frames: &[(f64, f64)],
) -> io::Result<usize> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let wav = encode_wav_24bit_stereo(sample_rate, frames);
    fs::write(path, &wav)?;
    Ok(wav.len())
}

#[derive(Clone, Copy, Debug)]
pub struct WavInspection {
    pub sample_rate: u32,
    pub channels: u16,
    pub bits_per_sample: u16,
    pub frame_count: usize,
    pub rms_left: f64,
    pub rms_right: f64,
    pub peak_abs: f64,
    pub stereo_diff_rms: f64,
}

/// Binary RIFF/WAVE 24-bit PCM validator for the automated test & verification suite.
pub fn decode_and_verify_wav(wav_bytes: &[u8]) -> Result<WavInspection, String> {
    if wav_bytes.len() < 44 {
        return Err("WAV file shorter than 44-byte RIFF header".to_string());
    }
    if &wav_bytes[0..4] != b"RIFF" || &wav_bytes[8..12] != b"WAVE" {
        return Err("Invalid RIFF/WAVE magic signature".to_string());
    }
    let riff_size = u32::from_le_bytes([
        wav_bytes[4],
        wav_bytes[5],
        wav_bytes[6],
        wav_bytes[7],
    ]) as usize;
    if riff_size + 8 != wav_bytes.len() {
        return Err(format!(
            "RIFF chunk size mismatch: {} + 8 != {}",
            riff_size,
            wav_bytes.len()
        ));
    }
    if &wav_bytes[12..16] != b"fmt " {
        return Err("Missing `fmt ` subchunk".to_string());
    }
    let fmt_size = u32::from_le_bytes([
        wav_bytes[16],
        wav_bytes[17],
        wav_bytes[18],
        wav_bytes[19],
    ]);
    let audio_format = u16::from_le_bytes([wav_bytes[20], wav_bytes[21]]);
    let channels = u16::from_le_bytes([wav_bytes[22], wav_bytes[23]]);
    let sample_rate = u32::from_le_bytes([
        wav_bytes[24],
        wav_bytes[25],
        wav_bytes[26],
        wav_bytes[27],
    ]);
    let byte_rate = u32::from_le_bytes([
        wav_bytes[28],
        wav_bytes[29],
        wav_bytes[30],
        wav_bytes[31],
    ]);
    let block_align = u16::from_le_bytes([wav_bytes[32], wav_bytes[33]]);
    let bits_per_sample = u16::from_le_bytes([wav_bytes[34], wav_bytes[35]]);

    if fmt_size != 16 || audio_format != 1 {
        return Err(format!(
            "Unexpected fmt chunk: size={} format={}",
            fmt_size, audio_format
        ));
    }
    if channels != 2 || bits_per_sample != 24 || block_align != 6 || byte_rate != sample_rate * 6 {
        return Err(format!(
            "Invalid 24-bit stereo PCM parameters: ch={} bits={} align={} rate={}",
            channels, bits_per_sample, block_align, byte_rate
        ));
    }
    if &wav_bytes[36..40] != b"data" {
        return Err("Missing `data` subchunk".to_string());
    }
    let data_size = u32::from_le_bytes([
        wav_bytes[40],
        wav_bytes[41],
        wav_bytes[42],
        wav_bytes[43],
    ]) as usize;
    if 44 + data_size != wav_bytes.len() || data_size % 6 != 0 {
        return Err("Invalid 24-bit stereo data payload length".to_string());
    }

    let frame_count = data_size / 6;
    let mut sum_l2 = 0.0;
    let mut sum_r2 = 0.0;
    let mut sum_diff2 = 0.0;
    let mut peak_abs = 0.0_f64;

    for frame in wav_bytes[44..].chunks_exact(6) {
        // Sign-extend 24-bit little-endian integer into i32
        let il = ((frame[0] as i32) | ((frame[1] as i32) << 8) | ((frame[2] as i32) << 16)) << 8 >> 8;
        let ir = ((frame[3] as i32) | ((frame[4] as i32) << 8) | ((frame[5] as i32) << 16)) << 8 >> 8;
        let fl = (il as f64) / 8_388_608.0;
        let fr = (ir as f64) / 8_388_608.0;
        sum_l2 += fl * fl;
        sum_r2 += fr * fr;
        sum_diff2 += (fl - fr) * (fl - fr);
        peak_abs = peak_abs.max(fl.abs()).max(fr.abs());
    }

    let inv_n = 1.0 / (frame_count.max(1) as f64);
    Ok(WavInspection {
        sample_rate,
        channels,
        bits_per_sample,
        frame_count,
        rms_left: (sum_l2 * inv_n).sqrt(),
        rms_right: (sum_r2 * inv_n).sqrt(),
        peak_abs,
        stereo_diff_rms: (sum_diff2 * inv_n).sqrt(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hadamard8_orthogonality() {
        let orig = [0.5, -1.2, 0.3, 2.1, -0.7, 0.9, -0.4, 1.5];
        let mut x = orig;
        hadamard8_normalized(&mut x);
        let norm_orig: f64 = orig.iter().map(|v| v * v).sum();
        let norm_h: f64 = x.iter().map(|v| v * v).sum();
        assert!((norm_orig - norm_h).abs() < 1e-12);

        // Symmetric involutory property: H * (H * x) == x!
        hadamard8_normalized(&mut x);
        for i in 0..8 {
            assert!((x[i] - orig[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn test_tpt_svf_4pole_stability_and_wav_24bit_header() {
        let mut svf = TptSvf4Pole::default();
        let mut frames = Vec::with_capacity(2400);
        for i in 0..2400 {
            let t = (i as f64) / 48_000.0;
            let raw = (TAU * 220.0 * t).sin() + 0.5 * (TAU * 3520.0 * t).sin();
            let fc = 300.0 + 2500.0 * (TAU * 4.0 * t).sin().abs();
            let y = svf.process_lowpass(raw, fc, 0.72, 48_000.0);
            assert!(y.is_finite() && y.abs() < 5.0);
            frames.push((y * 0.4, -y * 0.35));
        }

        let wav = encode_wav_24bit_stereo(48_000, &frames);
        let info = decode_and_verify_wav(&wav).expect("WAV header & payload must be valid");
        assert_eq!(info.sample_rate, 48_000);
        assert_eq!(info.channels, 2);
        assert_eq!(info.bits_per_sample, 24);
        assert_eq!(info.frame_count, 2400);
        assert!(info.rms_left > 0.05 && info.stereo_diff_rms > 0.05);
    }
}
