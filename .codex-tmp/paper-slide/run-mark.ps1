$env:RUNTIME_NODE = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$env:RUNTIME_NODE_MODULES = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
$env:RUNTIME_BIN_DIR = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\override'

& $env:RUNTIME_NODE 'C:\Users\Admin\.codex\plugins\cache\openai-primary-runtime\presentations\26.819.11345\skills\presentations\container_tools\mark_artifact_operation_started.mjs' `
  --operation-kind edit `
  --expected-output-count 1 `
  --output-format pptx

