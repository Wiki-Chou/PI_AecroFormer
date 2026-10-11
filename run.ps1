param([Parameter(ValueFromRemainingArguments=$true)][string[]]$Arguments)
$runtimePython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $runtimePython)) { throw 'Create .venv and install requirements-cpu.lock.txt as described in README.' }
& $runtimePython -m pi_aecroformer @Arguments
exit $LASTEXITCODE
