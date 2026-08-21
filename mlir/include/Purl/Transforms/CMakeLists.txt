set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Purl)
add_public_tablegen_target(MLIRPurlPassIncGen)
add_mlir_doc(Passes PurlPasses ./ -gen-pass-doc)
