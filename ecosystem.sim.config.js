module.exports = {
  apps: [
    {
      name: 'sim',
      script: '/home/administrator/workspace/sn-79/sim-launcher.sh',
      autorestart: true,
      restart_delay: 5000,
      watch: false,
      max_restarts: 10,
    },
  ],
};
