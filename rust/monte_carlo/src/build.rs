fn main() {
    // Tell cargo to rerun this script if the shader changes
    println!("cargo:rerun-if-changed=shader/src/lib.rs");
}
