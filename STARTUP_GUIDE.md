# 启动脚本使用说明

本文档记录当前项目常用启动命令与开机自启动管理方式。

## 1) 手动启动（默认 6020 端口）

```bash
cd /home/tsingwin/apps/sound/sherpa-onnx
./start_vad_sensevoice.sh
```

- 默认端口：`6020`
- 默认语言：`auto`
- 推理后端：自动检测（可手动指定）

## 2) 常用启动参数

### 指定端口

```bash
PORT=6021 ./start_vad_sensevoice.sh
```

### 强制释放端口后启动

```bash
FREE_PORT=1 PORT=6020 ./start_vad_sensevoice.sh
```

### 指定语言

```bash
LANGUAGE=zh ./start_vad_sensevoice.sh
LANGUAGE=en ./start_vad_sensevoice.sh
```

### 指定推理后端

```bash
PROVIDER=cpu ./start_vad_sensevoice.sh
PROVIDER=cuda ./start_vad_sensevoice.sh
```

### 指定 conda 环境

```bash
CONDA_ENV=sherpa ./start_vad_sensevoice.sh
```

## 3) 安装开机自启动（systemd）

项目已提供安装脚本：`install_autostart_vad_sensevoice.sh`

```bash
cd /home/tsingwin/apps/sound/sherpa-onnx
./install_autostart_vad_sensevoice.sh
```

默认安装参数：

- 端口：`6020`
- 后端：`cpu`
- 语言：`auto`
- 服务名：`vad-sensevoice.service`

### 安装时覆盖参数

```bash
PORT=6020 PROVIDER=cuda LANGUAGE=auto ./install_autostart_vad_sensevoice.sh
```

## 4) systemd 服务管理命令

### 查看状态

```bash
sudo systemctl status vad-sensevoice.service
```

### 启动 / 重启 / 停止

```bash
sudo systemctl start vad-sensevoice.service
sudo systemctl restart vad-sensevoice.service
sudo systemctl stop vad-sensevoice.service
```

### 开机启用 / 取消开机启用

```bash
sudo systemctl enable vad-sensevoice.service
sudo systemctl disable vad-sensevoice.service
```

### 查看实时日志

```bash
sudo journalctl -u vad-sensevoice.service -f
```

## 5) 卸载开机自启动服务

```bash
sudo systemctl disable --now vad-sensevoice.service
sudo rm -f /etc/systemd/system/vad-sensevoice.service
sudo systemctl daemon-reload
```

## 6) 访问地址

服务启动后可在浏览器访问：

```text
https://<设备IP>:6020
```

如果本机访问：

```text
https://localhost:6020
```

