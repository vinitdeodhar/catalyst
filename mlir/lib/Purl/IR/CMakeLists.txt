add_mlir_library(MLIRPurl
    PurlDialect.cpp
    PurlOps.cpp

    ADDITIONAL_HEADER_DIRS
    ${PROJECT_SOURCE_DIR}/include/Purl

    DEPENDS
    MLIRPurlOpsIncGen
    MLIRPurlEnumsIncGen
    MLIRQuantumOpsIncGen

    LINK_LIBS PUBLIC
    MLIRQuantum
)
