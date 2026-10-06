add_mlir_library(OASISTransforms
  FlattenMemref.cpp
  LowerMemrefCopy.cpp

  DEPENDS
  OASISTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTransformUtils
  )
