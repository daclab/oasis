# lit configuration for OASIS C++ pass tests (FileCheck).
import os

import lit.formats
from lit.llvm import llvm_config

config.name = "OASIS"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.oasis_obj_root, "test")

llvm_config.use_default_substitutions()
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)
llvm_config.add_tool_substitutions(["oasis-opt"], [config.oasis_tools_dir])
