[CmdletBinding()]
param(
    [string]$OutputRoot
)

$ErrorActionPreference = "Stop"

$repositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $repositoryRoot "dist"
}
$stamp = Get-Date -Format "yyyyMMdd"
$packageName = "Speech_Communication_submission_$stamp"
$stagingDirectory = Join-Path $OutputRoot $packageName
$zipPath = Join-Path $OutputRoot "$packageName.zip"

if (Test-Path -LiteralPath $stagingDirectory) {
    throw "Refusing to overwrite existing staging directory: $stagingDirectory"
}

if (Test-Path -LiteralPath $zipPath) {
    throw "Refusing to overwrite existing ZIP: $zipPath"
}

New-Item -ItemType Directory -Path $stagingDirectory -Force | Out-Null

Copy-Item (Join-Path $repositoryRoot "paper\main.tex") $stagingDirectory
Copy-Item (Join-Path $repositoryRoot "paper\refs.bib") $stagingDirectory
Copy-Item (Join-Path $repositoryRoot "paper\highlights.txt") $stagingDirectory
Copy-Item (Join-Path $repositoryRoot "paper\OVERLEAF.md") $stagingDirectory
Copy-Item (Join-Path $repositoryRoot "paper\vendor\cas\cas-dc.cls") $stagingDirectory
Copy-Item (Join-Path $repositoryRoot "paper\vendor\cas\cas-common.sty") $stagingDirectory
Copy-Item (Join-Path $repositoryRoot "paper\vendor\cas\cas-model2-names.bst") $stagingDirectory
Copy-Item (Join-Path $repositoryRoot "paper\vendor\cas\thumbnails") $stagingDirectory -Recurse
Copy-Item (Join-Path $repositoryRoot "paper\sections\*.tex") $stagingDirectory
Copy-Item (Join-Path $repositoryRoot "tables\*.tex") $stagingDirectory
Copy-Item (Join-Path $repositoryRoot "figures\*.pdf") $stagingDirectory

# The official CAS class resolves the corresponding-author email icon from
# thumbnails/cas-email.jpeg. All manuscript sources and figures remain at root.
$mainPath = Join-Path $stagingDirectory "main.tex"
$mainText = Get-Content -LiteralPath $mainPath -Raw
$mainText = $mainText.Replace("\graphicspath{{../figures/}}", "\graphicspath{{./}}")
$mainText = $mainText.Replace("\input{sections/", "\input{")
Set-Content -LiteralPath $mainPath -Value $mainText -NoNewline -Encoding UTF8

Get-ChildItem $stagingDirectory -Filter "*.tex" -File |
    ForEach-Object {
        $text = Get-Content -LiteralPath $_.FullName -Raw
        $text = $text.Replace("\input{../tables/", "\input{")
        Set-Content -LiteralPath $_.FullName -Value $text -NoNewline -Encoding UTF8
    }

Compress-Archive -Path (Join-Path $stagingDirectory "*") -DestinationPath $zipPath

Write-Output "Created staging directory: $stagingDirectory"
Write-Output "Created upload-ready ZIP: $zipPath"
