use std::env;

fn main() {
    println!("cargo:rerun-if-changed=cpp/");

    let oneapi_root = env::var("ONEAPI_ROOT").unwrap_or_else(|_| "/opt/intel/oneapi".to_string());

    // Add oneAPI library search paths
    println!("cargo:rustc-link-search=native=/opt/intel/oneapi/compiler/2025.2/lib");
    println!("cargo:rustc-link-search=native=/opt/intel/oneapi/lib");

    // Link SYCL library
    println!("cargo:rustc-link-lib=sycl");

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .std("c++17")
        .file("cpp/binomial_kernels.cpp")
        .include("/opt/intel/oneapi/compiler/2025.2/include")
        .flag("-Wall")
        .flag("-Wextra")
        .flag("-fsycl")
        .flag("-fsycl-targets=spir64")
        .flag("-fno-sycl-instrument-device-code")
        .flag("-D_SYCL_DISABLE_DEPRECATION_WARNINGS");

    // Use icpx compiler if available
    if let Ok(cxx_compiler) = env::var("CXX") {
        build.compiler(cxx_compiler);
    }

    build.compile("binomial_kernels");
}