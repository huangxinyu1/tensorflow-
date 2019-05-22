# -*- coding: utf-8 -*-
"""
Created on Wed May 22 18:26:35 2019

@author: Administrator
"""

import tensorflow as tf

#创建一个常量
ml = tf.constant([[3,3]])
#创建一个常量
m2= tf.constant([[2],[3]])
#创建一个矩阵乘法
product = tf.matmul(ml,m2)
print(product)


#定义一个会话启动图
sess = tf.Session()
#调用sess的run方法来执行矩阵乘法
#run(product)触发了图中的3个OP
result = sess.run(product)
print(result)

sess.close()


with tf.Session() as sess:
    #调用sess的run方法来执行矩阵乘法
    #run(product)触发了图中的3个OP
    result = sess.run(product)
    print(result)




#**************************************************************************

#变量的使用

x = tf.Variable([1,2])
a = tf.constant([3,3])
#增加一个减法的op
sub = tf.subtract(x,a)
#增加一个加法的op
add = tf.add(x,sub)

#初始化所有的变量
init = tf.global_variables_initializer()


with tf.Session() as sess:
    #变量初始化操作
    sess.run(init)
    print(sess.run(sub))
    print(sess.run(add))


#########################创建一个自增的数###################
#创建一个变量初始化为0
state = tf.Variable(0,name="counter")
#创建一个op，作用是使state加1
new_value = tf.add(state,1)
#赋值op
update = tf.assign(state,new_value)
#变量初始化
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
        sess.run(update)
        print(sess.run(state))



#########################Fetch  和  Feed #######
#Fetch
input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)

add = tf.add(input2,input3)
#乘法
mul = tf.multiply(input1,add)

with tf.Session() as sess:
    result = sess.run([add,mul])
    print(result)


#Feed
#创建占位符
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2)

with tf.Session() as sess:
    #feed的数据以字典的形式传入
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))


####################简单示例##########################
import numpy as np

#使用numpy生成100个随机点
x_data = np.random.rand(100)
y_data = x_data*0.1+0.2

#构造一个线性模型
b = tf.Variable(0.)
k = tf.Variable(0.)

y = k*x_data +b

#二次代价函数
loss = tf.reduce_mean(tf.square(y_data-y))
#定义一个梯度下降法来进行训练的优化器
optimizer = tf.train.GradientDescentOptimizer(0.2)
#定义最小化代价函数
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for step in range(201):
        sess.run(train)
        if step%20==0:
            print(step,sess.run([k,b]))



































