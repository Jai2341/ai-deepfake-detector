#!/usr/bin/env bash

echo "Installing Git LFS..."
apt-get update
apt-get install -y git-lfs

echo "Pulling LFS files..."
git lfs install
git lfs pull

echo "Build completed."