use std::env;

fn main() {
    println!("cargo:rerun-if-changed=cpp/");

    let dpcpp_path = env::var("DPCPP_HOME").unwrap_or_else(|_| "/opt/intel/oneapi".to_string());

    let mut build = cc::Build::new();

    build
        .cpp(true)
        .std("c++17")
        .file("cpp/monte_carlo_kernels.cpp")
        .include(&format!("{}/dpcpp/latest/include", dpcpp_path))
        .flag("-fsycl")
        .flag("-fsycl-targets=spir64");

    println!("cargo:rustc-link-search=native={}/dpcpp/latest/lib", dpcpp_path);
    println!("cargo:rustc-link-lib=sycl");

    if let Ok(dpcpp_compiler) = env::var("DPC++") {
        build.compiler(dpcpp_compiler);
    } else if std::process::Command::new("dpcpp").arg("--version").output().is_ok() {
        build.compiler("dpcpp");
    }

    build.compile("monte_carlo_kernels");
}