while true; do
  # Wakes up the agent, points it to your log, and runs your custom skill
  # The --dangerously-skip-permissions (or -y) flag ensures it doesn't wait for you.
  agent "/monitor-training-oneshot" -y
  
  echo "Check-in at $(date) complete. Next check in 60 minutes..."
  sleep 3600
done