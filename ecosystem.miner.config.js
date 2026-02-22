module.exports = {
  apps: [
    {
      name: "natix_miner",
      script: "start_miner.sh",
      interpreter: "bash",
      cwd: __dirname,
      autorestart: true,
      watch: false,
      max_restarts: 10,
      min_uptime: "10s",
    }
  ]
}