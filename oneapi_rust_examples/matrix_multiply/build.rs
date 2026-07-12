use std::env;

fn main() {
    // Tell Cargo to rebuild if any C++ files change
    println!("cargo:rerun-if-changed=cpp/");

    // Check if SYCL/oneAPI is available
    let oneapi_root = env::var("ONEAPI_ROOT").unwrap_or_else(|_| "/opt/intel/oneapi".to_string());

    // Use the latest compiler version available
    let compiler_versions = ["2025.2", "2025.1", "latest"];
    let mut compiler_path = None;
    let mut lib_path = None;

    for version in &compiler_versions {
        let test_path = format!("{}/compiler/{}", oneapi_root, version);
        if std::path::Path::new(&test_path).exists() {
            compiler_path = Some(format!("{}/include", test_path));
            lib_path = Some(format!("{}/lib", test_path));
            break;
        }
    }

    let compiler_include = compiler_path.expect("Could not find oneAPI compiler installation");
    let compiler_lib = lib_path.expect("Could not find oneAPI compiler lib directory");

    // Configure C++ compilation with oneAPI/SYCL
    let mut build = cc::Build::new();

    build
        .cpp(true)
        .std("c++17")
        .file("cpp/matrix_kernels.cpp")
        .include(&compiler_include)
        .flag("-fsycl")
        .flag("-fsycl-targets=spir64")
        .flag("-fno-sycl-instrument-device-code")
        .flag("-D_SYCL_DISABLE_DEPRECATION_WARNINGS");

    // Add oneAPI library paths - try both the specific compiler lib and the general lib
    println!("cargo:rustc-link-search=native={}", compiler_lib);
    println!("cargo:rustc-link-search=native={}/lib", oneapi_root);

    // Use icpx (the new recommended compiler) instead of dpcpp
    if std::process::Command::new("icpx").arg("--version").output().is_ok() {
        build.compiler("icpx");
    } else if std::process::Command::new("dpcpp").arg("--version").output().is_ok() {
        build.compiler("dpcpp");
    }

    build.compile("matrix_kernels");

    // Link the required libraries
    println!("cargo:rustc-link-lib=sycl");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=OpenCL");

    // Link Intel C++ runtime libraries
    println!("cargo:rustc-link-lib=intlc");
    println!("cargo:rustc-link-lib=svml");
    println!("cargo:rustc-link-lib=irng");

    // Add SYCL-specific runtime path
    println!("cargo:rustc-link-search=native={}/lib/intel64", oneapi_root);
    println!("cargo:rustc-link-search=native={}/lib64", oneapi_root);

    // Add SYCL runtime library directory
    println!("cargo:rustc-link-search=native=/opt/intel/oneapi/compiler/2025.2/lib/intel64");
}