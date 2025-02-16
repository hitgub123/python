import pandas as pd
import numpy as np
from PIL import Image      # 读取图片的包
from wordcloud import WordCloud,ImageColorGenerator   # 做词云图
import matplotlib.pyplot as plt   # 作图

df = pd.read_excel('数据.xlsx',sheet_name=0,engine='openpyxl')   # 读取excel数据信息
data=pd.DataFrame(index=df['关键词'])                            # 新生成一个DF文件，index为df的index
data['权重']=0     # data生成一个值均为0的列，主要定义这一列为int格式，为下面赋值做准备
# 将df的数据复制到 data中
for i in range(0,len(df)):
    data.iloc[i,0]=df.iloc[i,1]
#
data = data['权重'].sort_values(ascending = False)    # 排序
data = dict(data)     # 生成dict格式数据

font_path='F:/教学类材料//SourceHanSansCN-Regular.ttf'

# 读取背景图片
background_Image = np.array(Image.open("枫叶2.jpg"))
# 提取背景图片颜色
img_colors = ImageColorGenerator(background_Image)

#创建画板
plt.figure(figsize=(10,8),dpi=1000)    # 创建画板 ,定义图形大小及分辨率
mask = plt.imread(r"枫叶.jpg")          #自定义背景图片
# 设置词云图相关参数
wc=WordCloud(mask=mask,
             font_path=font_path,
             width=800,height=500,
             scale=2,mode="RGBA",
             background_color='white') 
wc=wc.generate_from_frequencies(data)  # 利用生成的dict文件制作词云图
#根据图片色设置背景色
wc.recolor(color_func=img_colors)

#存储图像
wc.to_file('词云图1.png')

#显示图片
plt.imshow(wc,interpolation="bilinear")
plt.axis("off")
plt.savefig("词云图2.png")