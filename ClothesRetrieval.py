from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras import backend as K
import numpy as np
import pickle 


def processImage(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = preprocess_input(x)
  return x

def getFeature(model, image):
  get_feauter_layer_output = K.function([model.layers[0].input],
                                  [model.layers[21].output])
  feature = get_feauter_layer_output([image])[0][0]
  return feature

def find_k_largest(k, arr):
  indices = sorted(range(len(arr)), key = lambda sub: arr[sub])[-k:]
  return indices

def get_paths(dir):
  paths = []
  f = open(dir, "r")
  for x in f:
    paths.append(x.replace("\n","").replace("/content/drive/My Drive/",""))
  f.close
  return paths

def result(paths, indices):
  res = []
  for i in indices:
    res.append(paths[i])
  return res

def cal_dis(features, query):
  dist = []
  for i in features:
    x = np.linalg.norm(i-query)
    dist.append(x)
  return dist

def load_data(dir):
  file = open(dir,'rb')
  data = pickle.load(file)
  return data

# model: model VGG 16
# image_path: Duong dan cua anh can tim kiem
# data_path: Đường dẫn đến file feauture đã được extract, do mới chỉ có 1 loại là áo nên để như dưới hàm main
# lable_path: Đường dẫn đến file chứa đường dẫn các ảnh.
def retrieval(model,image_path,data_path,label_path):
  img_input = processImage(image_path)
  query = getFeature(model,img_input)
  data = load_data(data_path)
  dist = cal_dis(data,query)
  paths = get_paths(label_path)
  dist, paths = zip(*sorted(zip(dist, paths)))
  return paths[0:20]


if __name__ == "__main__":

  model = VGG16(weights='imagenet', include_top=True)
  result = retrieval(model,'dataset/shirt/8.MF100P-5-1.jpg','shirt_db_v2','shirt_path.txt')
  print(result)

  

