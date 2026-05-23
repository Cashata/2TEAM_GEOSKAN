$ErrorActionPreference = "Stop"

$LOCAL_REPO = $PSScriptRoot
$DRONE = "pioneermini@10.42.0.1"
$REMOTE_DIR = "2TEAM_GEOSKAN"
$ARCHIVE = Join-Path $env:TEMP "2TEAM_GEOSKAN_sync.tgz"

Set-Location -LiteralPath $LOCAL_REPO

if (-not (Test-Path "fly_orb_ransac.py")) {
    throw "fly_orb_ransac.py not found. Run this script from the project folder."
}

if (Test-Path $ARCHIVE) {
    Remove-Item $ARCHIVE -Force
}

Write-Host "[1/4] Packing project..."

tar `
  --exclude=".git" `
  --exclude=".idea" `
  --exclude="__pycache__" `
  --exclude="*/__pycache__" `
  --exclude="repomix-output*.md" `
  --exclude="*.tif" `
  --exclude="*.tiff" `
  -czf "$ARCHIVE" `
  -C "$LOCAL_REPO" .

Write-Host "[2/4] Uploading archive to drone..."
scp "$ARCHIVE" "${DRONE}:2TEAM_GEOSKAN_sync.tgz"

Write-Host "[3/4] Unpacking on drone..."
$REMOTE_CMD = "rm -rf ~/$REMOTE_DIR; mkdir -p ~/$REMOTE_DIR; tar -xzf ~/2TEAM_GEOSKAN_sync.tgz -C ~/$REMOTE_DIR; rm ~/2TEAM_GEOSKAN_sync.tgz"
ssh $DRONE $REMOTE_CMD

Write-Host "[4/4] Checking files on drone..."
$CHECK_CMD = "cd ~/$REMOTE_DIR; pwd; ls -la | head -40; find . -maxdepth 2 -name 'calibration.py'"
ssh $DRONE $CHECK_CMD

Write-Host "Done."
