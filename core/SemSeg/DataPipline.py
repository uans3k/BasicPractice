import  Augmentor
import glob
import shutil
import os
import tensorflow as TF
import  cv2 as CV
import numpy as NP

def augment(sourcePath,labelPath,outputPath,size):
	p = Augmentor.Pipeline(
		source_directory=sourcePath
		,output_directory=outputPath
	)
	p.ground_truth(labelPath)
	p.rotate(probability=0.2, max_left_rotation=2, max_right_rotation=2)
	p.zoom(probability=0.2, min_factor=1.1, max_factor=1.2)
	p.skew(probability=0.2)
	p.random_distortion(probability=0.2, grid_width=100, grid_height=100, magnitude=1)
	p.shear(probability=0.2, max_shear_left=2, max_shear_right=2)
	p.crop_random(probability=0.2, percentage_area=0.8)
	p.flip_random(probability=0.2)
	p.sample(n=size)

def augmentSplit(sourcePath,trainAugPath,labelAugPath):
	if not os.path.lexists(trainAugPath):
		os.mkdir(trainAugPath)
	if not os.path.lexists(labelAugPath):
		os.mkdir(labelAugPath)

	label=glob.glob(sourcePath+"/_groundtruth*")
	label.sort()
	for i, val in enumerate(label):
		shutil.move(val,labelAugPath+"/"+str(i)+".tif")

	train = glob.glob(sourcePath + "/train*")
	train.sort()
	for i, val in enumerate(train):
		shutil.move(val, trainAugPath + "/" + str(i) + ".tif")
	pass

def convert2TFRecord(trainPath, labelPath, tfRecordPath,w,h):

	writer = TF.python_io.TFRecordWriter(tfRecordPath)

	trainImg = glob.glob(trainPath + "/*.tif")
	trainImg.sort()

	trainLabel = glob.glob(labelPath + "/*.tif")
	trainLabel.sort()

	size=len(trainImg)
	# train_set
	for index in range(size):
		img = CV.imread(trainImg[index])
		img=NP.asarray(a=img[:, :, 0], dtype=NP.uint8)
		img=CV.resize(src=img, dsize=(w,h))
		label = CV.imread(trainLabel[index])
		label = NP.asarray(a=label[:, :, 0], dtype=NP.uint8)
		label = CV.resize(src=label, dsize=(w, h))
		label[label <= 100] = 0
		label[label > 100] = 1
		example = TF.train.Example(features=TF.train.Features(feature={
			'label': TF.train.Feature(bytes_list=TF.train.BytesList(value=[label.tobytes()])),
			'image_raw': TF.train.Feature(bytes_list=TF.train.BytesList(value=[img.tobytes()]))
		}))
		writer.write(example.SerializeToString())
		if index % 50 == 0:
			print('Has Done writing %.2f%%' % (index / size * 100))
	writer.close()
	print("Done!")

def makeTFRecord(sourcePath
				 ,labelPath
				 ,outAugPath
				 ,trainAugPath
				 ,labelAugPath
				 ,tfRecodPath
				 ,size
				 ,w,h):
	augment(sourcePath, labelPath, outAugPath, size)
	augmentSplit(outAugPath,trainAugPath,labelAugPath)
	convert2TFRecord(trainAugPath,labelAugPath,tfRecodPath,w,h)


# def checkTFRecord(tfRecordPath,w,h,channel):
# 	queue = TF.train.string_input_producer(
# 		string_tensor=TF.train.match_filenames_once(tfRecordPath), num_epochs=1, shuffle=True)
# 	reader = TF.TFRecordReader()
# 	_, serialized = reader.read(queue)
# 	features = TF.parse_single_example(
# 		serialized,
# 		features={
# 			'label': TF.FixedLenFeature([], TF.string),
# 			'image_raw': TF.FixedLenFeature([], TF.string)
# 		})
#
# 	image = TF.decode_raw(features['image_raw'], TF.uint8)
# 	image = TF.reshape(image, [w,h,channel])
# 	label = TF.decode_raw(features['label'], TF.uint8)
# 	label = TF.reshape(label, [w,h,channel])
# 	# imageBatch, labelBatch = TF.train.batch([image,label], batch_size=1)
# 	with TF.Session() as sess:
# 		sess.run(TF.global_variables_initializer())
# 		sess.run(TF.local_variables_initializer())
# 		coord = TF.train.Coordinator()
# 		threads = TF.train.start_queue_runners(sess=sess,coord=coord)
# 		rImg, rLabel = sess.run([image, label])
# 		CV.imshow('image', rImg)
# 		CV.imshow('lael', rLabel * 255)
# 		CV.waitKey(0)
# 		coord.request_stop()
# 		coord.join(threads)
# 	print("Done")
# 	queue.close()
def checkTFRecord(tfRecordPath,w,h,channel):
	dataset = TF.data.TFRecordDataset([tfRecordPath])


	def parser(record):
		features = {
			'label': TF.FixedLenFeature([], TF.string),
			'image_raw': TF.FixedLenFeature([], TF.string)
		}
		example=TF.parse_single_example(record, features)

		image = TF.decode_raw(example['image_raw'], TF.uint8)
		image = TF.reshape(image, [w, h, channel])

		label = TF.decode_raw(example['label'], TF.uint8)
		label = TF.reshape(label, [w, h, channel])

		return  image,label

	dataset = dataset.map(parser)
	dataset = dataset.batch(2)
	dataset = dataset.repeat(1)
	dataset = dataset.shuffle(20)
	nextOP=dataset.make_one_shot_iterator().get_next()
	with TF.Session() as sess:
		sess.run(TF.global_variables_initializer())
		sess.run(TF.local_variables_initializer())

		rImgs, rLabels = sess.run(nextOP)
		CV.imshow('image', rImgs[0])
		CV.imshow('lael', rLabels[0] * 255)
		CV.waitKey(0)
		CV.imshow('image', rImgs[1])
		CV.imshow('lael', rLabels[1] * 255)
		CV.waitKey(0)

