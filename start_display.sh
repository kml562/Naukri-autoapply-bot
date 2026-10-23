#!/bin/bash
# Starts virtual display + VNC + noVNC so you can watch the bot in your browser
# Access at: http://YOUR_EC2_IP:6080/vnc.html

set -e

# Kill any existing instances
pkill Xvfb   2>/dev/null || true
pkill x11vnc  2>/dev/null || true
pkill websockify 2>/dev/null || true
sleep 1

# Start virtual display
Xvfb :99 -screen 0 1280x800x24 &
echo "Xvfb started on :99"
sleep 2

# Start VNC server on the virtual display (no password)
x11vnc -display :99 -nopw -listen localhost -xkb -forever -bg -logfile /tmp/x11vnc.log
echo "x11vnc started"
sleep 1

# Start noVNC web interface on port 6080
websockify --web /usr/share/novnc/ --daemon 6080 localhost:5900
echo "noVNC started on port 6080"

echo ""
echo "✅ Virtual display ready!"
echo "   Open in browser: http://$(curl -s ifconfig.me 2>/dev/null || echo YOUR_EC2_IP):6080/vnc.html"
echo "   Click Connect to watch the bot"
