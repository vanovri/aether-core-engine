use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

/// Deterministic SplitMix64 mixer used to seed Xoshiro256++ state words.
#[inline]
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// High-quality Xoshiro256++ pseudo-random number generator (256-bit state, zero dependencies).
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    #[inline]
    pub fn from_seed(seed: u64) -> Self {
        let mut sm = seed ^ 0xA5A5_5A5A_F0F0_0F0F;
        let mut s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        if s == [0; 4] {
            s[0] = 1;
        }
        Self { s }
    }

    /// Deterministic per-pixel, per-sample seed independent of thread scheduling order.
    #[inline]
    pub fn from_pixel_sample(x: u32, y: u32, sample: u32, salt: u64) -> Self {
        let mut h = salt ^ 0xD1B5_4A32_D192_ED03;
        h ^= (x as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h = h.rotate_left(23).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= (y as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
        h = h.rotate_left(29).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        h ^= (sample as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
        Self::from_seed(h)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[0]
            .wrapping_add(self.s[3])
            .rotate_left(23)
            .wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }

    /// Generates an IEEE-754 double uniformly distributed in [0.0, 1.0).
    #[inline]
    pub fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        const SCALE: f32 = 1.0 / ((1u32 << 24) as f32);
        ((self.next_u32() >> 8) as f32) * SCALE
    }

    #[inline]
    pub fn range_f64(&mut self, min: f64, max: f64) -> f64 {
        min + (max - min) * self.next_f64()
    }

    #[inline]
    pub fn next_usize(&mut self, bound: usize) -> usize {
        if bound <= 1 {
            return 0;
        }
        (self.next_u64() % (bound as u64)) as usize
    }

    /// Shirley-Chiu concentric mapping from unit square [0,1)^2 to unit disk.
    #[inline]
    pub fn concentric_disk(&mut self) -> (f64, f64) {
        let u_x = 2.0 * self.next_f64() - 1.0;
        let u_y = 2.0 * self.next_f64() - 1.0;
        if u_x == 0.0 && u_y == 0.0 {
            return (0.0, 0.0);
        }
        let (r, theta) = if u_x.abs() > u_y.abs() {
            (u_x, FRAC_PI_4 * (u_y / u_x))
        } else {
            (u_y, FRAC_PI_2 - FRAC_PI_4 * (u_x / u_y))
        };
        (r * theta.cos(), r * theta.sin())
    }

    /// Cosine-weighted hemisphere sample in local +Z up space (PDF = z / PI).
    #[inline]
    pub fn cosine_hemisphere(&mut self) -> (f64, f64, f64) {
        let (dx, dy) = self.concentric_disk();
        let z = (1.0 - dx * dx - dy * dy).max(0.0).sqrt();
        (dx, dy, z)
    }

    /// Uniform hemisphere sample in local +Z up space (PDF = 1 / (2 * PI)).
    #[inline]
    pub fn uniform_hemisphere(&mut self) -> (f64, f64, f64) {
        let z = self.next_f64();
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = TAU * self.next_f64();
        (r * phi.cos(), r * phi.sin(), z)
    }

    /// Uniform unit sphere direction sample (PDF = 1 / (4 * PI)).
    #[inline]
    pub fn uniform_sphere(&mut self) -> (f64, f64, f64) {
        let z = 1.0 - 2.0 * self.next_f64();
        let r = (1.0 - z * z).max(0.0).sqrt();
        let phi = TAU * self.next_f64();
        (r * phi.cos(), r * phi.sin(), z)
    }

    /// Cosine hemisphere PDF helper.
    #[inline]
    pub fn cosine_hemisphere_pdf(cos_theta: f64) -> f64 {
        if cos_theta > 0.0 {
            cos_theta / PI
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_bounds_and_mean() {
        let mut rng = Rng::from_seed(42);
        let n = 50_000;
        let mut sum = 0.0;
        for _ in 0..n {
            let v = rng.next_f64();
            assert!((0.0..1.0).contains(&v));
            sum += v;
        }
        let mean = sum / (n as f64);
        assert!((mean - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_sampling_unit_norms() {
        let mut rng = Rng::from_pixel_sample(17, 29, 3, 999);
        for _ in 0..1000 {
            let (x, y, z) = rng.cosine_hemisphere();
            let norm = (x * x + y * y + z * z).sqrt();
            assert!((norm - 1.0).abs() < 1e-12);
            assert!(z >= 0.0);

            let (sx, sy, sz) = rng.uniform_sphere();
            let snorm = (sx * sx + sy * sy + sz * sz).sqrt();
            assert!((snorm - 1.0).abs() < 1e-12);
        }
    }
}
