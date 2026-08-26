param(
    [Parameter(Mandatory=$true)][string]$InputDocx,
    [Parameter(Mandatory=$true)][string]$OutputTsv
)

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($InputDocx, $false, $true)
    try {
        $lines = New-Object System.Collections.Generic.List[string]
        $lines.Add("index`tpage`tstyle`ttext")
        for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
            $paragraph = $doc.Paragraphs.Item($i)
            $page = $paragraph.Range.Information(3)
            $style = [string]$paragraph.Range.Style
            $text = $paragraph.Range.Text.Replace("`r", "").Replace("`n", " ").Replace("`t", " ")
            $lines.Add("$i`t$page`t$style`t$text")
        }
        [System.IO.File]::WriteAllLines($OutputTsv, $lines, [System.Text.UTF8Encoding]::new($true))
        Write-Output "paragraphs=$($doc.Paragraphs.Count)"
        Write-Output "pages=$($doc.ComputeStatistics(2))"
    }
    finally {
        $doc.Close(0)
    }
}
finally {
    $word.Quit()
}
