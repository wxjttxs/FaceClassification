import tensorflow as tf 

train_whole_sample_size=6
test_whole_sample_size=2
face_class=2
train_batch_size=1
test_batch_size=1
nums_threads=1

image_size=28
train_path="output.tfrecords"
graph_path="mygraph/mygra"
cnn_model_save_path="cnn_model/cnn_model.ckpt"
print("======face train demo=======")

sess=tf.InteractiveSession()
x=tf.placeholder(tf.float32,[None,2352]) #28*28*3
y_=tf.placeholder("float",[None,2])
def weight_variable(shape):
	initial=tf.truncated_normal(shape,stddev=0.1)
	return tf.Variable(initial)
def bias_variable(shape):
	initial=tf.constant(0.1,shape=shape)
	return tf.Variable(initial)

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding="SAME")

def max_pool_2_2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

def read_and_decode(filename):
	filename_queue=tf.train.string_input_producer([filename])
	reader=tf.TFRecordReader()
	_,serialized_example=reader.read(filename_queue)
	features=tf.parse_single_example(
		serialized_example,
		features={
		'label':tf.FixedLenFeature([],tf.int64),
		'img_raw':tf.FixedLenFeature([],tf.string)
		})
	img=tf.decode_raw(features['img_raw'],tf.uint8)
	img=tf.reshape(img,[image_size,image_size,3])
	img=tf.cast(img,tf.float32)*(1./255.)-0.5
	label=tf.cast(features['label'],tf.int64)
	return img,label
print("step 1 : initial done")
#训练集
img_train,labels_train=read_and_decode(train_path)
img_train_batch,labels_train_batch=tf.train.shuffle_batch([img_train,labels_train],
															batch_size=train_batch_size,
															capacity=train_whole_sample_size,
															min_after_dequeue=1,
															num_threads=nums_threads)
train_labels=tf.one_hot(labels_train_batch,face_class,1,0)
# 测试集
img_test,labels_test=read_and_decode(train_path)
img_test_batch,labels_test_batch=tf.train.shuffle_batch([img_test,labels_test],
															batch_size=test_batch_size,
															capacity=test_whole_sample_size,
															min_after_dequeue=1,
															num_threads=nums_threads)
test_labels=tf.one_hot(labels_test_batch,face_class,1,0)
print("step 2 : trans to batch done")
print("cnn train start:")
def cnn():
	W_conv1=weight_variable([5,5,3,32])
	b_conv1=bias_variable([32])
	x_image=tf.reshape(x,[-1,28,28,3])
	h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
	h_pool1=max_pool_2_2(h_conv1)

	W_conv2=weight_variable([5,5,32,64])
	b_conv2=bias_variable([64])
	h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
	h_pool2=max_pool_2_2(h_conv2)

	# 全连接层
	W_fc1=weight_variable([7*7*64,1024])
	b_fc1=bias_variable([1024])
	h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
	h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

	# 减少过拟合，在输出层之前加上dropout
	keep_prob=tf.placeholder("float")
	h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

	# 输出层，预测值
	W_fc2=weight_variable([1024,2])
	b_fc2=bias_variable([2])
	y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

	# 训练加模型评估
	cross_entropy=-tf.reduce_sum(y_*tf.log(y_conv))
	train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction=tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
	accuracy=tf.reduce_mean(tf.cast(correct_prediction,"float"))
	sess.run(tf.global_variables_initializer())

	for i in range(2):
		img_xs,label_xs=sess.run([img_train_batch,train_labels])
		if i%1==0:
			train_accuracy=sess.run(accuracy,feed_dict={x:img_xs,y_:label_xs,keep_prob:1.0})
			print("step %d,training accuracy %g" %(i,train_accuracy))
		sess.run(train_step,feed_dict={x:img_xs,y_:label_xs,keep_prob:0.5})

	img_test_xs,label_test_xs=sess.run([img_test_batch,test_labels])
	print("test accuracy %g" %sess.run(accuracy,feed_dict={x:img_test_xs,y_:label_test_xs,keep_prob:1.0}))

cnn()
sess.close()