from telethon.sync import TelegramClient, events
from telethon.tl.types import InputMessagesFilterPhotos
import configparser
import time
import random
import logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO,filename='bulkmsg.log')
logger = logging.getLogger(__name__)
config = configparser.ConfigParser()
config.read('config.ini')
#配置文件导入的所有值  都是字符串，需要根据需要转换list int ......
api_id=config['API']['api_id']
api_hash = config['API']['api_hash']
phone= config['API']['phone']
bulkmsg= config['bulkmsg']['bulkmsg']
groups= config['groups']['username']
groups = groups.split(",")
interval = config['interval']['interval']#间隔时间
interval = interval.split(",")#random 随机时间
# 登录
client = TelegramClient(phone, api_id, api_hash)
client.start()

# 获取已加入的超级群组
# dialogs = client.get_dialogs()
# megagroups = [d.entity.username for d in dialogs if d.is_group and d.entity.megagroup]
# print(megagroups)#username  导出群名和username对应的列表
# 发送消息
while True:
    for group in groups:
        try:
            # 找到聊天实体
            entity = client.get_entity(group)
            # photos = client.get_messages(entity, 0, filter=InputMessagesFilterPhotos)
            # print(photos.total)
            # for message in client.iter_messages(entity):#10 requests per 30s
            #     print(message.id, message.text)
            # # 发送消息
            client.send_message(entity, bulkmsg)

            logger.info(f'{bulkmsg} sent to {group}')
        except Exception as e:
            logger.info(f'Error sending message to {group}: {e}')
        time.sleep(random.randint(int(interval[0]),int(interval[1])))

    # 退出
client.disconnect()

