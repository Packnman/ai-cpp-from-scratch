$ErrorActionPreference = "Stop"

$vcxsrv = "C:\Program Files\VcXsrv\vcxsrv.exe"
if (-not (Test-Path -LiteralPath $vcxsrv)) {
    throw "VcXsrv was not found at: $vcxsrv"
}

if (Get-Process -Name vcxsrv -ErrorAction SilentlyContinue) {
    Write-Host "VcXsrv is already running. Stop it first if it was not started on display :0 with TCP enabled."
    exit 0
}

# -ac is needed because the Docker container cannot use the Windows user's
# local Xauthority cookie. Only allow VcXsrv through Windows Firewall on the
# Private network profile.
$arguments = @(
    ":0",
    "-multiwindow",
    "-clipboard",
    "-wgl",
    "-ac",
    "-listen", "tcp"
)

Start-Process -FilePath $vcxsrv -ArgumentList $arguments
Write-Host "VcXsrv started on display :0 with TCP enabled."
Write-Host "Now run in the container: ./scripts/simulation/run.sh humanoid-viewer"
