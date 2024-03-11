ray up -y aws_autoscaler_config.yaml
export RAY_ADDRESS="ray://$(cat autoscaler_up.log | grep "ray start --address" | awk -F "'" '{split($2, a, ":"); print a[1]}'):10001"
