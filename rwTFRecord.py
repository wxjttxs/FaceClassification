import tensorflow as tf 
import os
import os.path

def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def write2TfRecord(path,tfname):
	with tf.Session() as sess:
		writer=tf.python_io.TFRecordWriter(tfname)
		for filename in os.listdir(path):
			file=os.path.join(path,filename)
			image_raw=tf.gfile.FastGFile(file,'rb'.read())
			img_data=tf.image.decode_jpeg(image_raw)
			example=tf.train.Example(features=tf.train.Features(feature={
					'img_raw':_bytes_feature(img_data.eval().tostring()),
					'label':_int64_feature(1)
					}))
			writer.write(example.SerializeToSting())
		writer.close()
