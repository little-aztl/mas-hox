
docker run -it --rm \
    -v "/home/exouser/Codes/marl-mini:/workspaces/my_awesome_rl" \
    -w "/workspaces/my_awesome_rl" \
    tencentailab/marl-mini:20240607 \
    bash