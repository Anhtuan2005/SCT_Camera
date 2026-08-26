$env:RUNTIME_NODE = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$env:RUNTIME_NODE_MODULES = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
$env:RUNTIME_BIN_DIR = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\override'
$env:PATH = 'E:\SCT_Camera\.codex-tmp\paper-slide\unzip-bin;' + $env:PATH

& $env:RUNTIME_NODE 'C:\Users\Admin\.codex\plugins\cache\openai-primary-runtime\presentations\26.819.11345\skills\presentations\template_following_scripts\check_template_fidelity.mjs' `
  --workspace 'E:\SCT_Camera\.codex-tmp\paper-slide-v2' `
  --starter-pptx 'E:\SCT_Camera\.codex-tmp\paper-slide-v2\template-starter.pptx' `
  --final-pptx 'E:\Slide_Cong_bo_khoa_hoc_FDSE_v2.pptx' `
  --map 'E:\SCT_Camera\.codex-tmp\paper-slide-v2\template-frame-map.json' `
  --starter-layout-dir 'E:\SCT_Camera\.codex-tmp\paper-slide-v2\template-starter-layout' `
  --final-layout-dir 'E:\SCT_Camera\.codex-tmp\paper-slide-v2\final-layout\final' `
  --edit-dir 'E:\SCT_Camera\.codex-tmp\paper-slide-v2'

