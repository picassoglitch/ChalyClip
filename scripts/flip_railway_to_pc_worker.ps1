# Point Railway prod at the PC worker (jobs + LLM) - the "no paid APIs" flip.
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\flip_railway_to_pc_worker.ps1 -FunnelUrl https://<host>.ts.net
#
# Sets on the ChalyClip service (triggers a redeploy):
#   CHALYBCLIP_JOB_DISPATCHER=modal          (the dispatcher is protocol-generic)
#   CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL   -> the worker's tunnel URL
#   CHALYBCLIP_OPENLLM_BASE_URL              -> <tunnel>/v1 (worker's Ollama proxy)
#   OPENLLM_API_KEY                        -> same shared secret as CHALYBCLIP_MODAL_TOKEN
param(
    [Parameter(Mandatory = $true)][string]$FunnelUrl
)

$ErrorActionPreference = "Stop"
$base = $FunnelUrl.TrimEnd("/")

$app = railway variables --service ChalyClip --json | ConvertFrom-Json
$token = $app.CHALYBCLIP_MODAL_TOKEN
if (-not $token) { Write-Error "CHALYBCLIP_MODAL_TOKEN missing on the ChalyClip service" }

railway variables --service ChalyClip `
    --set "CHALYBCLIP_JOB_DISPATCHER=modal" `
    --set "CHALYBCLIP_MODAL_PIPELINE_ENDPOINT_URL=$base/" `
    --set "CHALYBCLIP_OPENLLM_BASE_URL=$base/v1" `
    --set "OPENLLM_API_KEY=$token" `
    --skip-deploys

Write-Output "Variables set. Trigger the deploy when ready: railway redeploy --service ChalyClip"
