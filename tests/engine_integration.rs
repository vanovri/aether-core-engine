use aether_core_engine::audio::{
    decode_and_verify_wav, encode_wav_24bit_stereo, synthesize_fluid_soundtrack, SAMPLE_RATE,
    TOTAL_FRAMES,
};
use aether_core_engine::fluid::{FluidMedium, FluidSolver3D};
use aether_core_engine::math::{Aabb, Vec3};
use aether_core_engine::png::{
    decode_and_verify_png, encode_bmp_rgb8, encode_png_rgb8, encode_ppm_rgb8, zlib_compress_stored,
    zlib_decompress,
};
use aether_core_engine::renderer::{render_scene_work_stealing, RenderConfig};
use aether_core_engine::scene::Scene;
use aether_core_engine::verify::run_core_verification_suite;

#[test]
fn integration_full_coupled_pipeline_and_binary_codecs() {
    // 1. Run 3D Navier-Stokes fluid steps
    let mut solver = FluidSolver3D::new(20, 28, 20);
    let mut telemetry = Vec::with_capacity(20);
    for i in 0..20 {
        telemetry.push(solver.step(0.06, (i as f64) * 0.75));
    }

    let medium = FluidMedium::from_solver(
        &solver,
        Aabb::new(Vec3::new(-2.0, 0.2, -2.0), Vec3::new(2.0, 4.8, 1.3)),
        0.85,
        0.32,
    );
    let scene = Scene::build_cyberpunk_scene(16.0 / 9.0, Some(medium));

    // 2. Verify all mathematical, BVH, BSDF, Subsurface, and Fluid invariants
    let checks = run_core_verification_suite(&scene).expect("All core invariants must pass");
    assert_eq!(checks.len(), 4);
    assert!(checks.iter().all(|c| c.passed));

    // 3. Render small multithreaded frame & verify deterministic non-black HDR + PNG/BMP/PPM codecs
    let cfg = RenderConfig {
        width: 128,
        height: 72,
        spp: 8,
        max_bounces: 6,
        tile_size: 16,
        exposure: 1.2,
        indirect_clamp: 18.0,
        num_threads: 4,
    };
    let rep = render_scene_work_stealing(&scene, &cfg);
    assert!(rep.total_rays > (128 * 72 * 8));
    assert!(rep.aggregate_mrays_per_sec > 0.0);
    let mean_luma: f64 = rep
        .rgb8_pixels
        .chunks_exact(3)
        .map(|p| 0.2126 * (p[0] as f64) + 0.7152 * (p[1] as f64) + 0.0722 * (p[2] as f64))
        .sum::<f64>()
        / ((cfg.width * cfg.height) as f64);
    assert!(mean_luma > 8.0 && mean_luma < 245.0, "Mean luma = {}", mean_luma);

    // Verify stored zlib + LZ77 PNG + PPM + BMP
    let stored_z = zlib_compress_stored(&rep.rgb8_pixels[..512]);
    assert_eq!(
        zlib_decompress(&stored_z).expect("Stored zlib must decompress"),
        &rep.rgb8_pixels[..512]
    );

    let png_bytes = encode_png_rgb8(cfg.width, cfg.height, &rep.rgb8_pixels);
    let png_dec = decode_and_verify_png(&png_bytes).expect("PNG must verify");
    assert_eq!(png_dec.rgb_pixels, rep.rgb8_pixels);

    let ppm_bytes = encode_ppm_rgb8(cfg.width, cfg.height, &rep.rgb8_pixels);
    assert!(ppm_bytes.starts_with(b"P6\n128 72\n255\n"));
    let bmp_bytes = encode_bmp_rgb8(cfg.width, cfg.height, &rep.rgb8_pixels);
    assert_eq!(&bmp_bytes[0..2], b"BM");

    // 4. Synthesize full 15s 48kHz 24-bit WAV soundtrack & verify RIFF/WAVE binary format
    let audio_frames = synthesize_fluid_soundtrack(&telemetry);
    assert_eq!(audio_frames.len(), TOTAL_FRAMES);
    let wav_bytes = encode_wav_24bit_stereo(SAMPLE_RATE, &audio_frames);
    let wav_info = decode_and_verify_wav(&wav_bytes).expect("24-bit WAV must verify");
    assert_eq!(wav_info.frame_count, 720_000);
    assert!(wav_info.rms_left > 0.01 && wav_info.peak_abs < 1.0);
}
