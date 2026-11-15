param([ValidateSet('check','verify','train','resume','report-path')][string]$Mode)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (!(Test-Path -LiteralPath $python)) {
    throw 'Run task 01 to set up the project environment.'
}
if ($Mode -eq 'report-path') {
    $report = & $python -B -X utf8 (Join-Path $PSScriptRoot 'reproduce.py') $Mode
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & code --reuse-window "$report"
    exit $LASTEXITCODE
}
& $python -B -X utf8 (Join-Path $PSScriptRoot 'reproduce.py') $Mode
exit $LASTEXITCODE
