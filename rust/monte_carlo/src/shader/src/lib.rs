#![no_std]
#![cfg_attr(target_arch = "spirv", feature(lang_items))]

use glam::UVec3;
use rand_xoshiro::Xoshiro128Plus;
use spirv_std::glam;
use spirv_std::spirv;

#[spirv(compute(threads(256)))]
pub fn main(
    #[spirv(global_invocation_id)] id: UVec3,
    #[spirv(push_constant)] num_workgroups: &u32,
    #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] result: &mut u32,
) {
    let gid = id.x;
    let total_threads = *num_workgroups * 256;
    let points_per_thread = 100000;
    let mut rng = Xoshiro128Plus::seed_from_u64((gid as u64).wrapping_mul(0x12345678));
    let mut count = 0u32;

    for _ in 0..points_per_thread {
        let x = rng.next_u32() as f32 / u32::MAX as f32;
        let y = rng.next_u32() as f32 / u32::MAX as f32;
        if x * x + y * y <= 1.0 {
            count += 1;
        }
    }

    spirv_std::sync::atomic_add_u32(result, count);
}

#[cfg(target_arch = "spirv")]
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}
