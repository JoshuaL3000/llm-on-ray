ray up -y aws_autoscaler_config.yaml | tee autoscaler_up.log
export RAY_ADDRESS_HEAD="ray://$(cat autoscaler_up.log | grep "ray start --address" | awk -F "'" '{split($2, a, ":"); print a[1]}'):10001"
