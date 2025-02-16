import pandas as pd
import numpy as np

# 1．准备格式化数据。
d=pd.read_excel('5.xlsx')
print(d)

# 2．通过案例掌握 Pandas 进行数据处理的基本操作方法。
print(d.loc[0])

# 3．掌握 Linux 有关文件和目录操作的常用命令。 
#ls
#cd

# 4．学会用系统调用和库函数进行编程，实现对文件的创建、打开、关闭、读和写。
#创建、
f = open("5.txt",'w')
#写
f.write("hello world!")
#、关闭
f.close()
# 打开
f = open("5.txt")
# 读
print(f.read())