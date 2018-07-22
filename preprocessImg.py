import tensorflow as tf
import numpy as np
import scipy.misc
import os
import os.path
# import matplotlib
# from PIL import Image
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write2TfRecord(path,tfname):
	with tf.Session() as sess:
		writer=tf.python_io.TFRecordWriter(tfname)
		for filename in os.listdir(path):
			file=os.path.join(path,filename)
			image_raw=tf.gfile.FastGFile(file,'rb').read()
			img_data=tf.image.decode_jpeg(image_raw)
			example=tf.train.Example(features=tf.train.Features(feature={
					'img_raw':_bytes_feature(img_data.eval().tostring()),
					'label':_int64_feature(1)
					}))
			writer.write(example.SerializeToString())
		writer.close()

def distort_color(image,color_ordering=0):
	if color_ordering==0:
		image=tf.image.random_brightness(image,max_delta=32./255.)
		image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
		image=tf.image.random_hue(image,max_delta=0.2)
		image=tf.image.random_contrast(image,lower=0.5,upper=1.5)
	elif color_ordering==1:
		image=tf.image.random_contrast(image,lower=0.5,upper=1.5)
		image=tf.image.random_brightness(image,max_delta=32./255.)
		image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
		image=tf.image.random_hue(image,max_delta=0.2)	
	elif color_ordering==2:
		image=tf.image.random_hue(image,max_delta=0.2)
		image=tf.image.random_contrast(image,lower=0.5,upper=1.5)
		image=tf.image.random_brightness(image,max_delta=32./255.)
		image=tf.image.random_saturation(image,lower=0.5,upper=1.5)
	return tf.clip_by_value(image,0.0,1.0)

def preprocess_for_train(image,height,width):
	distorted_img=tf.image.central_crop(image,0.5)
	distorted_img=tf.image.resize_images(distorted_img,size=[height,width],method=np.random.randint(4))
	# distorted_img=distort_color(distorted_img,np.random.randint(2))
	return distorted_img

def changeImg(filepath):
	if not os.path.exists("test"):
		os.makedirs("test")
	
	with tf.Session() as sess:		
		for i,file in enumerate(os.listdir(filepath)):
			filename=os.path.join(filepath,file)
			img_raw=tf.gfile.FastGFile(filename,"rb").read()			
			img_data=tf.image.decode_jpeg(img_raw)
			img_data = tf.cast(img_data, tf.float32)			
			result=preprocess_for_train(img_data,28,28)		
			scipy.misc.imsave("test/test{}.jpg".format(i),result.eval())
			print(result.eval().shape)

if __name__=="__main__":
	# changeImg("inpath")
	outname="output.tfrecords"
	write2TfRecord("test",outname)