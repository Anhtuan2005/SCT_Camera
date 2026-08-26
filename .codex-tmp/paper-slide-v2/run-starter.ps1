$env:RUNTIME_NODE = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$env:RUNTIME_NODE_MODULES = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
$env:RUNTIME_BIN_DIR = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\override'
$env:PATH = 'E:\SCT_Camera\.codex-tmp\paper-slide\unzip-bin;' + $env:PATH

& $env:RUNTIME_NODE 'C:\Users\Admin\.codex\plugins\cache\openai-primary-runtime\presentations\26.819.11345\skills\presentations\template_following_scripts\prepare_template_starter_deck.mjs' `
  --workspace 'E:\SCT_Camera\.codex-tmp\paper-slide-v2' `
  --pptx 'E:\Slide_Cong_bo_khoa_hoc_FDSE.pptx' `
  --map 'E:\SCT_Camera\.codex-tmp\paper-slide-v2\template-frame-map.json' `
  --out 'E:\SCT_Camera\.codex-tmp\paper-slide-v2\template-starter.pptx' `
  --preview-dir 'E:\SCT_Camera\.codex-tmp\paper-slide-v2\template-starter-preview' `
  --layout-dir 'E:\SCT_Camera\.codex-tmp\paper-slide-v2\template-starter-layout'

