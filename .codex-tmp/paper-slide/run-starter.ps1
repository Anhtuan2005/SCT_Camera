$env:RUNTIME_NODE = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe'
$env:RUNTIME_NODE_MODULES = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules'
$env:RUNTIME_BIN_DIR = 'C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\bin\override'
$env:PATH = 'E:\SCT_Camera\.codex-tmp\paper-slide\unzip-bin;' + $env:PATH

& $env:RUNTIME_NODE 'C:\Users\Admin\.codex\plugins\cache\openai-primary-runtime\presentations\26.819.11345\skills\presentations\template_following_scripts\prepare_template_starter_deck.mjs' `
  --workspace 'E:\SCT_Camera\.codex-tmp\paper-slide' `
  --pptx 'E:\BaoCao_ThuyetTrinh_BAN_CUOI_v3.pptx' `
  --map 'E:\SCT_Camera\.codex-tmp\paper-slide\template-frame-map.json' `
  --out 'E:\SCT_Camera\.codex-tmp\paper-slide\template-starter.pptx' `
  --preview-dir 'E:\SCT_Camera\.codex-tmp\paper-slide\template-starter-preview' `
  --layout-dir 'E:\SCT_Camera\.codex-tmp\paper-slide\template-starter-layout' `
  --contact-sheet 'E:\SCT_Camera\.codex-tmp\paper-slide\template-starter-contact-sheet.png'

