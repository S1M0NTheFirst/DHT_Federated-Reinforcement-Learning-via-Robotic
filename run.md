##conda activate swiftbot
 sudo tee /etc/sudoers.d/swiftbot-chown <<EOF
 simon ALL=(root) NOPASSWD: /usr/bin/chown

docker stop $(docker ps -aq) 2>/dev/null;
 docker rm -f $(docker ps -aq) 2>/dev/null; 
 redis-cli flushall;
  rm -rf /tmp/swiftbot_checkpoints/* 
  sudo rm -rf /tmp/swiftbot_checkpoints/*
Step 2 — Build Docker images

  cd ~/swiftbot_rl

  # DHT+FRL image (Condition A)
  docker build -f dht_frl/Dockerfile -t swiftbot-robot:latest dht_frl/
  
  sudo -E env "PATH=$CONDA_PREFIX/bin:$PATH" SIMULATE_CRIU=1 python3 dht_frl/dht_frl_runner.py


    " This is a known limitation of the CRIU cuda plugin: it can't lock the CUDA context when the target process is in certain states (most often, when other CUDA processes on
  the same GPU are competing for it, or the target hasn't reached an idle CUDA state). The plugin's path is brittle on consumer GPUs."
                                                                                                                                                   
    
  # docker build -f swiftbot_rl/dht_frl/Dockerfile -t swiftbot-robot:latest swiftbot_rl/dht_frl/

  # Baseline image (Conditions B and C — built from swiftbot_rl/ root for multi-dir COPY)
  docker build -f criu_cold/Dockerfile -t swiftbot-baseline:latest .

  Step 3 — Run Condition A (DHT+FRL) — ~90 min

  # Terminal 1 — Flower server
  cd ~/swiftbot_rl
  python3 dht_frl/flower_server.py

  # Wait for: "Waiting for 8 robot clients to connect..."

  # Terminal 2 — DHT runner
  cd ~/swiftbot_rl
  python3 dht_frl/dht_frl_runner.py

  # Monitor progress
  watch -n 10 "wc -l ~/swiftbot_rl/dht_frl/results/migration_events.csv"

  Done when Terminal 1 shows FedAvg complete. Results saved. Then:
  docker stop $(docker ps -q)
  redis-cli flushall

  Step 4 — Run Condition B (CRIU Cold) — ~30 min

  cd ~/swiftbot_rl
  python3 criu_cold/criu_cold_runner.py
  # Completes automatically when all 8 robots finish

  redis-cli flushall

  Step 5 — Run Condition C (CRIU Warm) — ~30 min

  cd ~/swiftbot_rl
  python3 criu_warm/criu_warm_runner.py

  redis-cli flushall

  Step 6 — Generate paper figures

  cd ~/swiftbot_rl
  python3 evaluation/compare_all.py
  # Output: evaluation/figures/fig1–fig4.png + summary_table.csv + .tex

  Quick sanity check after all 3 conditions

  python3 - <<'EOF'
  import pandas as pd
  dht  = pd.read_csv('dht_frl/results/migration_events.csv')
  cold = pd.read_csv('criu_cold/results/migration_events.csv')
  warm = pd.read_csv('criu_warm/results/migration_events.csv')
  print(f"DHT+FRL:   regression={dht['regression_pct'].mean():.1f}%  policy_load={dht['policy_load_ms'].mean():.1f}ms")
  print(f"CRIU cold: regression={cold['regression_pct'].mean():.1f}%  policy_load={cold['policy_load_ms'].mean():.1f}ms")
  print(f"CRIU warm: regression={warm['regression_pct'].mean():.1f}%  policy_load={warm['policy_load_ms'].mean():.1f}ms")
  EOF
  # Expected: DHT+FRL regression << cold/warm, DHT+FRL policy_load_ms > 0
  ##