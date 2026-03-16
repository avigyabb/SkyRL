#!/usr/bin/env bash
#
# Auto-retry wrapper for training scripts with Ray cluster restart
#
# Automatically relaunches training when it exits (crash, OOM, etc.)
# The script uses resume_mode=latest, so it auto-resumes from checkpoints.
# On each retry, Ray head is restarted to ensure a clean cluster state.
#
# Usage:
#   ./run_biomni_qwen30ba3b_gspo_tis_autoretry.sh [OPTIONS] [max_retries]
#
# Options:
#   --script PATH    Override the training script (default: run_biomni_qwen30ba3b_gspo_tis.sh)
#   --no-ray-restart Skip Ray restart on retries (for debugging)
#
# Arguments:
#   max_retries: Maximum number of restart attempts (default: unlimited, use -1)
#
# Examples:
#   ./run_biomni_qwen30ba3b_gspo_tis_autoretry.sh              # Unlimited retries
#   ./run_biomni_qwen30ba3b_gspo_tis_autoretry.sh 5            # Max 5 retries
#   ./run_biomni_qwen30ba3b_gspo_tis_autoretry.sh --script ./mock_crash_training.sh 3  # Test mode
#

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../../logs"
WRAPPER_LOG="$LOG_DIR/autoretry_wrapper.log"

# Defaults
TRAINING_SCRIPT="$SCRIPT_DIR/run_biomni_qwen30ba3b_gspo_tis.sh"
RAY_RESTART_ENABLED=true
MAX_RETRIES=-1  # -1 means unlimited

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --script)
            TRAINING_SCRIPT="$2"
            shift 2
            ;;
        --no-ray-restart)
            RAY_RESTART_ENABLED=false
            shift
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            MAX_RETRIES="$1"
            shift
            ;;
    esac
done

# Configuration
RETRY_DELAY_SECONDS=30
MIN_RUN_SECONDS=120      # Detect rapid crash loops
RAY_STABILIZE_SECONDS=60  # Wait for workers to reconnect after Ray restart

# Counters
ATTEMPT=0
TOTAL_RESTARTS=0

log() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $*" | tee -a "$WRAPPER_LOG"
}

cleanup_gpu_processes() {
    log "Killing all stale GPU processes..."
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    if [[ -n "$gpu_pids" ]]; then
        local count
        count=$(echo "$gpu_pids" | wc -l)
        log "Found $count processes holding GPU memory — killing all"
        echo "$gpu_pids" | xargs -r kill -9 2>/dev/null
        sleep 5
        local remaining
        remaining=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if [[ "$remaining" -gt 0 ]]; then
            log "WARNING: $remaining GPU processes survived kill -9 (likely zombies). GPU memory may still be leaked."
            log "If OOMs persist, restart the Docker container to reclaim zombie CUDA contexts."
        else
            log "All GPU memory reclaimed"
        fi
    else
        log "No GPU processes found"
    fi
}

restart_ray_head() {
    if [[ "$RAY_RESTART_ENABLED" != "true" ]]; then
        log "Ray restart disabled (--no-ray-restart), skipping..."
        return 0
    fi
    
    log "Restarting Ray head node..."
    
    # Source the Ray head start script (includes `ray stop -f`)
    if [[ -f "$SCRIPT_DIR/start_ray_head.sh" ]]; then
        bash "$SCRIPT_DIR/start_ray_head.sh"
        local ray_exit_code=$?
        
        if [[ $ray_exit_code -ne 0 ]]; then
            log "WARNING: Ray head start returned exit code $ray_exit_code"
        else
            log "Ray head started successfully"
        fi
        
        log "Waiting ${RAY_STABILIZE_SECONDS}s for Ray cluster to stabilize..."
        sleep $RAY_STABILIZE_SECONDS
    else
        log "WARNING: start_ray_head.sh not found at $SCRIPT_DIR/start_ray_head.sh"
        log "Skipping Ray restart"
    fi
}

# Validate training script exists
if [[ ! -f "$TRAINING_SCRIPT" ]]; then
    log "ERROR: Training script not found: $TRAINING_SCRIPT"
    exit 1
fi

mkdir -p "$LOG_DIR"

log "=============================================="
log "Auto-retry wrapper started"
log "Training script: $TRAINING_SCRIPT"
log "Max retries: $MAX_RETRIES (-1 = unlimited)"
log "Ray restart enabled: $RAY_RESTART_ENABLED"
log "=============================================="

while true; do
    ATTEMPT=$((ATTEMPT + 1))
    
    log "----------------------------------------------"
    log "Attempt #$ATTEMPT (Total restarts: $TOTAL_RESTARTS)"
    log "----------------------------------------------"
    
    # Clean up stale GPU processes and restart Ray before retries (not on first attempt)
    if [[ $TOTAL_RESTARTS -gt 0 ]]; then
        cleanup_gpu_processes
        restart_ray_head
    fi
    
    start_time=$(date +%s)
    
    # Run training (script uses resume_mode=latest)
    bash "$TRAINING_SCRIPT"
    exit_code=$?
    
    end_time=$(date +%s)
    run_duration=$((end_time - start_time))
    
    log "Training exited with code $exit_code after ${run_duration}s"
    
    # Success - exit cleanly
    if [[ $exit_code -eq 0 ]]; then
        log "Training completed successfully!"
        exit 0
    fi
    
    # Check max retries
    if [[ $MAX_RETRIES -ne -1 ]] && [[ $TOTAL_RESTARTS -ge $MAX_RETRIES ]]; then
        log "ERROR: Max retries ($MAX_RETRIES) reached. Giving up."
        exit 1
    fi
    
    # Rapid crash detection
    if [[ $run_duration -lt $MIN_RUN_SECONDS ]]; then
        log "WARNING: Crashed after only ${run_duration}s - possible config issue"
        # sleep $((RETRY_DELAY_SECONDS * 3))
    fi
    
    TOTAL_RESTARTS=$((TOTAL_RESTARTS + 1))
    
    log "Waiting ${RETRY_DELAY_SECONDS}s before retry..."
    sleep $RETRY_DELAY_SECONDS
done
