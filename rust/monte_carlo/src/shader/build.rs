use spirv_builder::{MetadataPrintout, SpirvBuilder};

fn main() {
    SpirvBuilder::new(".", "spirv-unknown-vulkan1.1")
        .print_metadata(MetadataPrintout::None)
        .build()
        .unwrap();
}
