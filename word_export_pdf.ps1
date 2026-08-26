param(
  [Parameter(Mandatory = $true)][string]$InputDocx,
  [Parameter(Mandatory = $true)][string]$OutputPdf
)

$word = $null
$doc = $null
try {
  $word = New-Object -ComObject Word.Application
  $word.Visible = $false
  $word.DisplayAlerts = 0
  $doc = $word.Documents.Open((Resolve-Path -LiteralPath $InputDocx).Path, $false, $true)
  $doc.Repaginate()
  $pages = $doc.ComputeStatistics(2)
  $doc.ExportAsFixedFormat((New-Object System.IO.FileInfo($OutputPdf)).FullName, 17)
  Write-Output "word_pages=$pages"
}
finally {
  if ($null -ne $doc) { $doc.Close($false) }
  if ($null -ne $word) { $word.Quit() }
  [System.GC]::Collect()
  [System.GC]::WaitForPendingFinalizers()
}
