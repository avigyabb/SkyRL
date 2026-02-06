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
GPU_BUSY_DELAY_SECONDS=600  # 10 minutes - wait when GPUs are occupied by others

# Counters
ATTEMPT=0
TOTAL_RESTARTS=0

log() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $*" | tee -a "$WRAPPER_LOG"
}

restart_ray_head() {
    if [[ "$RAY_RESTART_ENABLED" != "true" ]]; then
        log "Ray restart disabled (--no-ray-restart), skipping..."
        return 0
    fi
    
    log "Restarting Ray head node..."
    
    # Source the Ray head start script
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

is_gpu_busy_error() {
    # Check if the most recent training log indicates GPUs are occupied by other processes
    # This specific error from vLLM means someone else is using the GPUs
    
    # Find the most recently modified log file
    local latest_log
    latest_log=$(ls -t "$LOG_DIR"/*.log 2>/dev/null | head -1)
    
    if [[ -z "$latest_log" || ! -f "$latest_log" ]]; then
        log "No log file found to check for GPU errors"
        return 1  # No log file, can't determine
    fi
    
    # Check the last 500 lines of the log for the GPU busy error pattern
    # Pattern: vLLM error when GPU memory is insufficient due to other processes
    if tail -500 "$latest_log" | grep -qiE "Free memory on device.*is less than desired GPU memory utilization|reduce GPU memory used by other processes"; then
        log "GPU busy error found in: $latest_log"
        return 0  # True - GPU busy error detected
    fi
    
    return 1  # False - not a GPU busy error
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
    
    # Restart Ray before retries (not on first attempt)
    if [[ $TOTAL_RESTARTS -gt 0 ]]; then
        restart_ray_head
    fi
    
    start_time=$(date +%s)
    
    # Run training
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
    
    # Check if GPUs are occupied by other processes (vLLM init failure)
    if is_gpu_busy_error; then
        log "GPU BUSY: Detected that GPUs are occupied by other processes"
        log "Waiting ${GPU_BUSY_DELAY_SECONDS}s (10 minutes) for resources to free up..."
        sleep $GPU_BUSY_DELAY_SECONDS
    else
        # Rapid crash detection
        if [[ $run_duration -lt $MIN_RUN_SECONDS ]]; then
            log "WARNING: Crashed after only ${run_duration}s - possible config issue"
        fi
        
        log "Waiting ${RETRY_DELAY_SECONDS}s before retry..."
        sleep $RETRY_DELAY_SECONDS
    fi
    
    TOTAL_RESTARTS=$((TOTAL_RESTARTS + 1))
done
