# PowerShell script to verify server deployment via SSH
# Usage: .\scripts\verify_deployment_ssh.ps1

param(
    [string]$ServerIP = "52.0.207.242",
    [string]$ServerUser = "ubuntu",
    [string]$SSHKey = "$env:USERPROFILE\.ssh\server_saver_key"
)

$ErrorActionPreference = "Continue"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "DiffRhythm Server Deployment Verification" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Server: ${ServerUser}@${ServerIP}" -ForegroundColor Yellow
Write-Host ""

$ProjectDir = "/opt/diffrhythm"
$ContainerName = "diffrhythm-api"
$ImageName = "diffrhythm:prod"

function Invoke-SSHCommand {
    param([string]$Command)
    $fullCommand = "ssh -i `"$SSHKey`" -o StrictHostKeyChecking=no ${ServerUser}@${ServerIP} `"$Command`""
    $result = Invoke-Expression $fullCommand 2>&1
    return $result
}

function Write-Success {
    param([string]$Message)
    Write-Host "✓ $Message" -ForegroundColor Green
}

function Write-Error {
    param([string]$Message)
    Write-Host "✗ $Message" -ForegroundColor Red
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠ $Message" -ForegroundColor Yellow
}

function Write-Section {
    param([string]$Title)
    Write-Host ""
    Write-Host "=== $Title ===" -ForegroundColor Blue
    Write-Host ""
}

$overallStatus = 0

# Phase 1: Server Connectivity
Write-Section "Phase 1: Server Connectivity"
try {
    $result = Invoke-SSHCommand "echo Connected"
    if ($LASTEXITCODE -eq 0) {
        Write-Success "SSH connection successful"
    } else {
        Write-Error "SSH connection failed"
        exit 1
    }
} catch {
    Write-Error "SSH connection failed: $_"
    Write-Host "Please check:" -ForegroundColor Yellow
    Write-Host "  - SSH key: $SSHKey"
    Write-Host "  - Server IP: $ServerIP"
    Write-Host "  - Server user: $ServerUser"
    exit 1
}

# Phase 2: Docker Installation
Write-Section "Phase 2: Docker Installation"
try {
    $dockerVersion = Invoke-SSHCommand "docker --version"
    if ($dockerVersion -and $LASTEXITCODE -eq 0) {
        Write-Success "Docker installed: $dockerVersion"
    } else {
        Write-Error "Docker not installed"
        $overallStatus = 1
    }
} catch {
    Write-Error "Docker check failed"
    $overallStatus = 1
}

# Phase 3: Project Directory
Write-Section "Phase 3: Project Directory"
try {
    $dirCheck = Invoke-SSHCommand "test -d $ProjectDir"
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Project directory exists: $ProjectDir"
        $dirSize = Invoke-SSHCommand "du -sh $ProjectDir | cut -f1"
        Write-Host "  Directory size: $dirSize"
    } else {
        Write-Error "Project directory not found: $ProjectDir"
        $overallStatus = 1
    }
} catch {
    Write-Error "Directory check failed"
    $overallStatus = 1
}

# Phase 4: Docker Image
Write-Section "Phase 4: Docker Image"
try {
    $imageExists = Invoke-SSHCommand "docker images $ImageName --format '{{.Repository}}:{{.Tag}}'"
    if ($imageExists -and $imageExists -notmatch "error") {
        Write-Success "Docker image exists: $ImageName"
        $imageSize = Invoke-SSHCommand "docker images $ImageName --format '{{.Size}}'"
        Write-Host "  Image size: $imageSize"
    } else {
        Write-Error "Docker image not found: $ImageName"
        Write-Host "  Run: docker build -f Dockerfile.prod -t $ImageName ." -ForegroundColor Yellow
        $overallStatus = 1
    }
} catch {
    Write-Error "Image check failed"
    $overallStatus = 1
}

# Phase 5: Docker Container
Write-Section "Phase 5: Docker Container"
$isRunning = $false
try {
    $containerStatus = Invoke-SSHCommand "docker ps -a --filter name=$ContainerName --format '{{.Status}}'"
    if ($containerStatus -and $containerStatus -notmatch "error") {
        Write-Success "Container exists: $ContainerName"
        Write-Host "  Status: $containerStatus"
        
        $runningCheck = Invoke-SSHCommand "docker ps --filter name=$ContainerName --format '{{.Names}}'"
        if ($runningCheck -and $runningCheck -notmatch "error" -and $runningCheck.Trim()) {
            Write-Success "Container is running"
            $isRunning = $true
        } else {
            Write-Error "Container is not running"
            Write-Host "  Start with: docker-compose -f docker-compose.prod.yml up -d" -ForegroundColor Yellow
            $overallStatus = 1
        }
    } else {
        Write-Error "Container not found: $ContainerName"
        $overallStatus = 1
    }
} catch {
    Write-Error "Container check failed"
    $overallStatus = 1
}

# Phase 6: Container Health
Write-Section "Phase 6: Container Health"
if ($isRunning) {
    try {
        $healthStatus = Invoke-SSHCommand "docker inspect $ContainerName --format '{{.State.Health.Status}}'"
        if ($healthStatus -eq "healthy") {
            Write-Success "Container health: healthy"
        } elseif ($healthStatus -eq "starting") {
            Write-Warning "Container health: starting (may need more time)"
        } else {
            Write-Error "Container health: $healthStatus"
            $overallStatus = 1
        }
        
        Write-Host ""
        Write-Host "Recent container logs:" -ForegroundColor Cyan
        $logs = Invoke-SSHCommand "docker logs $ContainerName --tail 20"
        Write-Host $logs
    } catch {
        Write-Error "Health check failed"
        $overallStatus = 1
    }
}

# Phase 7: API Health Check
Write-Section "Phase 7: API Health Check"
if ($isRunning) {
    try {
        $healthResponse = Invoke-SSHCommand "curl -s -f http://localhost:8000/api/v1/health"
        if ($healthResponse -and $LASTEXITCODE -eq 0) {
            Write-Success "API health endpoint responding"
            Write-Host "  Response:" -ForegroundColor Cyan
            try {
                $json = $healthResponse | ConvertFrom-Json
                $json | ConvertTo-Json -Depth 10
            } catch {
                Write-Host $healthResponse
            }
            
            if ($healthResponse -match '"models_loaded":\s*true') {
                Write-Success "Models loaded successfully"
            } else {
                Write-Warning "Models not loaded yet (may still be loading)"
            }
        } else {
            Write-Error "API health endpoint not responding"
            $overallStatus = 1
        }
    } catch {
        Write-Error "API health check failed"
        $overallStatus = 1
    }
} else {
    Write-Warning "Skipping API check (container not running)"
    $overallStatus = 1
}

# Phase 8: Disk Space
Write-Section "Phase 8: Disk Space"
try {
    $diskInfo = Invoke-SSHCommand "df -h / | tail -1"
    if ($diskInfo) {
        $parts = $diskInfo -split '\s+'
        $diskPercent = $parts[4] -replace '%', ''
        $diskFree = $parts[3]
        
        if ([int]$diskPercent -lt 80) {
            Write-Success "Disk usage: ${diskPercent}% (OK)"
        } elseif ([int]$diskPercent -lt 90) {
            Write-Warning "Disk usage: ${diskPercent}% (Warning)"
        } else {
            Write-Error "Disk usage: ${diskPercent}% (Critical)"
            $overallStatus = 1
        }
        Write-Host "  Free space: $diskFree"
    }
} catch {
    Write-Error "Could not check disk usage"
}

# Summary
Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Deployment Verification Summary" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if ($overallStatus -eq 0) {
    Write-Host "✓ Deployment appears successful!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Test from frontend"
    Write-Host "  2. Monitor logs: ssh -i $SSHKey $ServerUser@$ServerIP 'docker logs -f $ContainerName'"
    Write-Host "  3. Check metrics: curl http://${ServerIP}:8000/api/v1/metrics"
} else {
    Write-Host "✗ Some issues detected" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please review the errors above and:" -ForegroundColor Yellow
    Write-Host "  1. Check container logs: ssh -i $SSHKey $ServerUser@$ServerIP 'docker logs $ContainerName'"
    Write-Host "  2. Verify Docker image: ssh -i $SSHKey $ServerUser@$ServerIP 'docker images $ImageName'"
    Write-Host "  3. Check disk space: ssh -i $SSHKey $ServerUser@$ServerIP 'df -h /'"
    Write-Host "  4. Restart services: ssh -i $SSHKey $ServerUser@$ServerIP 'cd $ProjectDir && docker-compose -f docker-compose.prod.yml restart'"
}

exit $overallStatus
