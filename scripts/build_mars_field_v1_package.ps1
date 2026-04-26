param(
    [string]$PackageRoot = "",
    [string]$VersionTag = "mars_field_engineering_approximation_v1_0"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$workspaceRoot = Split-Path -Parent $repoRoot
if ([string]::IsNullOrWhiteSpace($PackageRoot)) {
    $PackageRoot = Join-Path $workspaceRoot "release_packages"
}
$stagingRoot = Join-Path $PackageRoot $VersionTag
$zipPath = Join-Path $PackageRoot ($VersionTag + ".zip")

if (Test-Path -LiteralPath $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}

New-Item -ItemType Directory -Path $stagingRoot -Force | Out-Null

$copyItems = @(
    "README.md",
    ".gitignore",
    "requirements.txt",
    "environment.linux-gpu.yml",
    "configs",
    "docs",
    "marsstack",
    "scripts",
    "datasets",
    "inputs",
    "outputs\paper_bundle_v1",
    "vendors\ProteinMPNN",
    "vendors\esm-main"
)

foreach ($item in $copyItems) {
    $src = Join-Path $repoRoot $item
    $dst = Join-Path $stagingRoot $item
    $dstParent = Split-Path -Parent $dst
    if (-not (Test-Path -LiteralPath $src)) {
        Write-Host "SKIP missing: $src"
        continue
    }
    if (-not (Test-Path -LiteralPath $dstParent)) {
        New-Item -ItemType Directory -Path $dstParent -Force | Out-Null
    }
    if ((Get-Item -LiteralPath $src) -is [System.IO.DirectoryInfo]) {
        Copy-Item -LiteralPath $src -Destination $dst -Recurse -Force
    } else {
        Copy-Item -LiteralPath $src -Destination $dst -Force
    }
}

$removeItems = @(
    "outputs\paper_bundle_v1\structure_panels\*\figure_session.pse",
    "vendors\ProteinMPNN_tmp_extract"
)

foreach ($pattern in $removeItems) {
    $targetPattern = Join-Path $stagingRoot $pattern
    Get-ChildItem -Path $targetPattern -Force -ErrorAction SilentlyContinue | ForEach-Object {
        if ($_.PSIsContainer) {
            Remove-Item -LiteralPath $_.FullName -Recurse -Force -ErrorAction SilentlyContinue
        } else {
            Remove-Item -LiteralPath $_.FullName -Force -ErrorAction SilentlyContinue
        }
    }
}

$manifestText = @"
# Package Summary

- version tag: $VersionTag
- source repo: $repoRoot
- staging path: $stagingRoot
- zip path: $zipPath
- included:
  - README.md
  - .gitignore
  - configs/
  - docs/
  - marsstack/
  - scripts/
  - datasets/README.md
  - inputs/
  - outputs/paper_bundle_v1/
  - vendors/ProteinMPNN/
  - vendors/esm-main/
"@

$manifestText | Set-Content -LiteralPath (Join-Path $stagingRoot "PACKAGE_SUMMARY.md") -Encoding UTF8

Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $zipPath -Force

Write-Host "Created staging package: $stagingRoot"
Write-Host "Created zip package: $zipPath"
