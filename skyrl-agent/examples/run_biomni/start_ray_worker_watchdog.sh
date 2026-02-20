#!/usr/bin/env bash
#
# Ray Worker Watchdog
#
# Monitors Ray worker connectivity and automatically reconnects when
# the head node's session is destroyed (e.g., after head restart).
#
# Run this in a tmux session on each worker node.
#
# Usage:
#   ./start_ray_worker_watchdog.sh [OPTIONS]
#
# Options:
#   --poll-interval SECONDS  How often to check connectivity (default: 10)
#   --head-address ADDRESS   Ray head address (default: from start_ray.sh or 10.138.0.3:6379)
#

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/../../logs"
WATCHDOG_LOG="$LOG_DIR/worker_watchdog.log"

# Defaults
POLL_INTERVAL=10
HEAD_ADDRESS="10.138.0.3:6379"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --poll-interval)
            POLL_INTERVAL="$2"
            shift 2
            ;;
        --head-address)
            HEAD_ADDRESS="$2"
            shift 2
            ;;
        -*)
            echo "Unknown option: $1"
            exit 1
            ;;
        *)
            shift
            ;;
    esac
done

# Use ray directly from the venv
VENV_BIN=/mnt/biomni_filestore/venvs/skyrl-agent/bin

mkdir -p "$LOG_DIR"

log() {
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    echo "[$timestamp] $*" | tee -a "$WATCHDOG_LOG"
}

is_ray_connected() {
    # Check if this worker is properly connected to the head node
    # Simply checking 'ray status' exit code is not enough - local Ray processes
    # may still be running even after the head restarts.
    #
    # Instead, we check if the cluster has at least 2 nodes (head + this worker).
    # If we only see 1 node, we're likely orphaned from the head.
    
    local status_output
    status_output=$($VENV_BIN/ray status 2>&1)
    local exit_code=$?
    
    # If ray status fails completely, not connected
    if [[ $exit_code -ne 0 ]]; then
        log "ray status failed with exit code $exit_code"
        return 1
    fi
    
    # Count active nodes - look for lines matching "1 node_" pattern
    local node_count
    node_count=$(echo "$status_output" | grep -c "node_")
    
    if [[ $node_count -lt 2 ]]; then
        log "Only $node_count node(s) in cluster (expected >= 2). Likely disconnected from head."
        return 1
    fi
    
    return 0
}

restart_ray_worker() {
    log "Restarting Ray worker..."
    
    if [[ -f "$SCRIPT_DIR/start_ray.sh" ]]; then
        bash "$SCRIPT_DIR/start_ray.sh"
        local exit_code=$?
        
        if [[ $exit_code -ne 0 ]]; then
            log "WARNING: Ray worker start returned exit code $exit_code"
            return 1
        else
            log "Ray worker started successfully"
            return 0
        fi
    else
        log "ERROR: start_ray.sh not found at $SCRIPT_DIR/start_ray.sh"
        return 1
    fi
}

log_cluster_status() {
    # Log the current cluster status (node count, resources)
    local status_output
    status_output=$($VENV_BIN/ray status 2>&1)
    local node_count
    node_count=$(echo "$status_output" | grep -c "node_")
    local cpu_info
    cpu_info=$(echo "$status_output" | grep "CPU" | head -1)
    local gpu_info
    gpu_info=$(echo "$status_output" | grep "GPU" | head -1)
    
    log "Cluster status: $node_count node(s), $cpu_info, $gpu_info"
}

log "=============================================="
log "Ray Worker Watchdog started"
log "Head address: $HEAD_ADDRESS"
log "Poll interval: ${POLL_INTERVAL}s"
log "=============================================="

# Track if we've logged the first successful connection
first_connection_logged=false

# Initial connection attempt
if ! is_ray_connected; then
    log "Not connected to Ray cluster, starting worker..."
    restart_ray_worker
fi

# Log status after initial connection
if is_ray_connected; then
    log "Initial connection successful"
    log_cluster_status
    first_connection_logged=true
fi

# Main watchdog loop
consecutive_failures=0
MAX_CONSECUTIVE_FAILURES=3

while true; do
    sleep $POLL_INTERVAL
    
    if is_ray_connected; then
        # Connected - reset failure counter
        if [[ $consecutive_failures -gt 0 ]]; then
            log "Connection restored after $consecutive_failures failures"
            log_cluster_status
            consecutive_failures=0
        fi
        
        # Log first connection if we haven't yet (e.g., if initial check failed but later succeeded)
        if [[ "$first_connection_logged" == "false" ]]; then
            log "First successful connection to cluster"
            log_cluster_status
            first_connection_logged=true
        fi
    else
        consecutive_failures=$((consecutive_failures + 1))
        log "Connection check failed (consecutive failures: $consecutive_failures)"
        
        if [[ $consecutive_failures -ge $MAX_CONSECUTIVE_FAILURES ]]; then
            log "Lost connection to Ray head (failed $consecutive_failures times), restarting worker..."
            
            if restart_ray_worker; then
                consecutive_failures=0
                # Wait a bit longer after restart before checking again
                log "Waiting 30s after restart before next check..."
                sleep 30
            else
                log "Worker restart failed, will retry in next poll cycle"
            fi
        fi
    fi
done
