add_mlir_library(OASISTransforms
  FlattenMemref.cpp

  DEPENDS
  OASISTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTransformUtils
  )
