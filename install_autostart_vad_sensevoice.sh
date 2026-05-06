#!/usr/bin/env bash
set -euo pipefail

# 一键安装开机自启动服务（systemd）
# 默认端口固定为 6020，可在命令前覆盖：
#   PORT=6021 PROVIDER=cpu ./install_autostart_vad_sensevoice.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="vad-sensevoice.service"
SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"

PORT="${PORT:-6020}"
PROVIDER="${PROVIDER:-cpu}"
LANGUAGE="${LANGUAGE:-auto}"
CONDA_ENV="${CONDA_ENV:-sherpa}"
RUN_USER="${RUN_USER:-$(id -un)}"

echo "[1/4] 写入 systemd 服务: ${SERVICE_PATH}"
sudo tee "${SERVICE_PATH}" >/dev/null <<EOF
[Unit]
Description=VAD SenseVoice WebUI Service
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=${RUN_USER}
WorkingDirectory=${SCRIPT_DIR}
Environment=PORT=${PORT}
Environment=PROVIDER=${PROVIDER}
Environment=LANGUAGE=${LANGUAGE}
Environment=CONDA_ENV=${CONDA_ENV}
Environment=FREE_PORT=1
ExecStart=/bin/bash ${SCRIPT_DIR}/start_vad_sensevoice.sh
Restart=always
RestartSec=3
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

echo "[2/4] 重新加载 systemd"
sudo systemctl daemon-reload

echo "[3/4] 启用并立即启动服务"
sudo systemctl enable --now "${SERVICE_NAME}"

echo "[4/4] 服务状态"
sudo systemctl --no-pager --full status "${SERVICE_NAME}" || true

echo ""
echo "安装完成。"
echo "访问地址: https://<设备IP>:${PORT}"
echo "常用命令:"
echo "  sudo systemctl restart ${SERVICE_NAME}"
echo "  sudo systemctl stop ${SERVICE_NAME}"
echo "  sudo journalctl -u ${SERVICE_NAME} -f"
