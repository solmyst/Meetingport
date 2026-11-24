# Windows installer: ask for the access password, create a Python 3.10-3.12 venv, install, and launch.
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

function Test-Version([string]$Exe) {
    & $Exe -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)"
    return $LASTEXITCODE -eq 0
}

function Find-Python {
    if ($env:PYTHON) {
        if (Test-Version $env:PYTHON) { return $env:PYTHON }
        Write-Host "PYTHON is set but is not Python 3.10, 3.11, or 3.12."
        exit 1
    }
    $pyLauncher = Get-Command py -ErrorAction SilentlyContinue
    if ($pyLauncher) {
        foreach ($ver in @("-3.12", "-3.11", "-3.10")) {
            $exe = & py $ver -c "import sys; print(sys.executable)" 2>$null
            if ($LASTEXITCODE -eq 0 -and $exe) {
                $exe = "$exe".Trim()
                if (Test-Version $exe) { return $exe }
            }
        }
    }
    foreach ($name in @("python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        if (Test-Version $cmd.Source) { return $cmd.Source }
    }
    Write-Host @"
Meetingport needs Python 3.10, 3.11, or 3.12.
Install Python 3.12 from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
Then run run.bat again.
You can also point at an interpreter by setting the PYTHON environment variable.
"@
    exit 1
}

function Convert-Secure([Security.SecureString]$Secure) {
    $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
    try { return [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr) }
    finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr) }
}

$Python = Find-Python
$plain = $null
$accepted = $false
foreach ($attempt in 1..3) {
    $secure = Read-Host "Meetingport access password" -AsSecureString
    $plain = Convert-Secure $secure
    $env:MEETINGPORT_PASSWORD = $plain
    & $Python -m mport.access --check
    if ($LASTEXITCODE -eq 0) {
        $accepted = $true
        break
    }
    Write-Host "Wrong password."
    $plain = $null
    Remove-Item Env:MEETINGPORT_PASSWORD -ErrorAction SilentlyContinue
}
if (-not $accepted) {
    Write-Host "Access denied. Ask the person who shared Meetingport for the password."
    exit 1
}

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPy)) {
    Write-Host "Creating a virtual environment"
    & $Python -m venv (Join-Path $PSScriptRoot ".venv")
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if (-not (Test-Version $venvPy)) {
    Write-Host "The existing .venv is not Python 3.10, 3.11, or 3.12. Delete the .venv folder and run run.bat again."
    exit 1
}

$stamp = Join-Path $PSScriptRoot ".venv\.deps-ok"
$req = Get-Item (Join-Path $PSScriptRoot "requirements.txt")
$needInstall = -not (Test-Path $stamp)
if (-not $needInstall) {
    $needInstall = (Get-Item $stamp).LastWriteTime -lt $req.LastWriteTime
}
if ($needInstall) {
    Write-Host "Installing dependencies"
    & $venvPy -m pip install --upgrade pip
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $venvPy -m pip install -r $req.FullName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    New-Item -ItemType File -Path $stamp -Force | Out-Null
}

$cfg = Join-Path $PSScriptRoot "config.json"
if (-not (Test-Path $cfg)) {
    $example = Join-Path $PSScriptRoot "config.example.json"
    if (Test-Path $example) {
        Copy-Item $example $cfg
        Write-Host "Created config.json from config.example.json"
    } else {
        Write-Host "config.json is missing."
        exit 1
    }
}

Write-Host "Starting Meetingport"
$env:MEETINGPORT_PASSWORD = $plain
& $venvPy (Join-Path $PSScriptRoot "meetingport.py") --config $cfg @args
exit $LASTEXITCODE
