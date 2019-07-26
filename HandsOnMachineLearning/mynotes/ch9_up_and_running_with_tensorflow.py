import tensorflow as tf


x = tf.Variable(3, name="x")
y = tf.Variable(4,name="y")

f = x*x*y + y + 2
# with tf.Session() as sess:
#     x.initializer.run()
#     y.initializer.run()
#     result = f.eval()
#
# sess.close()

init = tf.global_variables_initializer() # prepare an init node
with tf.Session() as sess:
    init.run() # actually initialize all the variables
    result = f.eval()

print(result)
sess.close()
