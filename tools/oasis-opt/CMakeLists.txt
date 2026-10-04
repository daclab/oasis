add_llvm_executable(oasis-opt oasis-opt.cpp)
llvm_update_compile_flags(oasis-opt)
target_link_libraries(oasis-opt
  PRIVATE
  OASISTransforms
  MLIRAffineDialect
  MLIRAffineToStandard
  MLIRArithDialect
  MLIRArithTransforms
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIROptLib
  MLIRSCFDialect
  MLIRTransforms
  )
# The LLVM static libraries carry debug info (~400 MB); strip it from the binary.
target_link_options(oasis-opt PRIVATE -Wl,--strip-debug)
mlir_check_all_link_libraries(oasis-opt)
