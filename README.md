# 
debian11 x64 python3
name:telegram bulk msg
pip install telethon
sudo apt-get -y install screen
后台运行screen -dmS tgbulkmsg python3 tgbulkmsg.py 
screen -ls   列出正在后台运行的程序

screen -X -S 2091275 quit   退出后台运行的脚本

2091275是screen -ls列出来的后台运行的程序ID

