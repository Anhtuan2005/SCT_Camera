$ErrorActionPreference = 'Stop'

$inputDocx = 'E:\SCT_Camera\NOI_DUNG_BO_SUNG_BAO_CAO.docx'
$qaDir = 'E:\SCT_Camera\.codex_addendum_baocao_20260821_01\qa-01'
$outputPdf = Join-Path $qaDir 'NOI_DUNG_BO_SUNG_BAO_CAO.pdf'

New-Item -ItemType Directory -Path $qaDir -Force | Out-Null

$word = $null
$document = $null
try {
    Write-Output 'Starting Word'
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $word.ScreenUpdating = $false
    $word.AutomationSecurity = 3
    Write-Output 'Opening document'
    $document = $word.Documents.Open($inputDocx, $false, $true)
    Write-Output 'Saving as PDF'
    $document.SaveAs2($outputPdf, 17)
    Write-Output 'PDF complete'
}
finally {
    if ($null -ne $document) {
        $document.Close($false)
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($document)
    }
    if ($null -ne $word) {
        $word.Quit()
        [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Output $outputPdf
