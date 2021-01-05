import numpy as np
from PIL import Image
import requests
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
from scipy.optimize import fmin_l_bfgs_b
from matplotlib import pyplot as plt
iterations = 10#迭代次数
CHANNELS = 3
image_size = 300 #图片大小
image_width = image_size
image_height = image_size
imagenet_mean_rgb_values = [123.68, 116.779, 103.939]
content_weght = 0.02
style_weight = 4.5
total_variation_weght = 0.995 #TV正则项权重
total_variation_loss_factor = 1.25#TV正则项损失因子

input_image_path = "cartoon_images/Cartoon_FCM.jpg"#需要风格转移的图片路径
style_image_path = "cartoon_images/style8.jpg"#风格图片路径
output_image_path = "cartoon_images/output_style8.png"#风格迁移后图片输出的路径
combined_image_path = "cartoon_images/combined_style8.png"#组合对比图片路径

input_image = Image.open(input_image_path)
input_image = input_image.resize((image_width, image_height))
input_image.save(input_image_path)

style_image = Image.open(style_image_path)
style_image = style_image.resize((image_width, image_height))
style_image.save(style_image_path)


#选择一张输入图，减去通道颜色均值后，得到风格图片在vgg16各个层的输出值
input_image_array = np.asarray(input_image, dtype="float32")
input_image_array = np.expand_dims(input_image_array, axis=0)
input_image_array[:, :, :, 0] -= imagenet_mean_rgb_values[2]
input_image_array[:, :, :, 1] -= imagenet_mean_rgb_values[1]
input_image_array[:, :, :, 2] -= imagenet_mean_rgb_values[0]
input_image_array = input_image_array[:, :, :, ::-1] # bgr ->rgb


#选择一张风格图，减去通道颜色均值后，得到风格图片在vgg16各个层的输出值
style_image_array = np.asarray(style_image, dtype="float32")
style_image_array = np.expand_dims(style_image_array, axis=0)
style_image_array[:, :, :, 0] -= imagenet_mean_rgb_values[2]
style_image_array[:, :, :, 1] -= imagenet_mean_rgb_values[1]
style_image_array[:, :, :, 2] -= imagenet_mean_rgb_values[0]
style_image_array = style_image_array[:, :, :, ::-1] # bgr ->rgb

input_image = backend.variable(input_image_array)
style_image = backend.variable(style_image_array)
combination_image = backend.placeholder((1, image_height, image_size, 3))

input_tensor = backend.concatenate([input_image,style_image,combination_image], axis=0)
model = VGG16(input_tensor=input_tensor, include_top=False)

def content_loss(content, combination):# 内容损失函数loss值
    return backend.sum(backend.square(combination - content))

layers = dict([(layer.name, layer.output) for layer in model.layers])

content_layer = "block2_conv2"
# 内容特征对应的vgg16各层名称

layer_features = layers[content_layer]
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss = backend.variable(0.)
loss = loss + content_weght * content_loss(content_image_features,
                                      combination_features)
#计算 loss值 loss等于 内容特征的权重 * 输入图片内容损失函数的loss值

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def compute_style_loss(style, combination): # 计算风格图片的loss值  即风格损失函数
    style = gram_matrix(style)
    combination = gram_matrix(combination)
    size = image_height * image_width
    return backend.sum(backend.square(style - combination)) / (4. * (CHANNELS ** 2) * (size ** 2))

style_layers = ["block1_conv2", "block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
# 风格特征对应的vgg16各层名称  

for layer_name in style_layers: #计算风格特征对应的各层的损失值相加
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    style_loss = compute_style_loss(style_features, combination_features)
    loss += (style_weight / len(style_layers)) * style_loss 
# loss值 等于风格特征的权重除以层数 * 对应层的风格损失值 最后再求和


def total_variation_loss(x):# 定义组合图片的损失函数total variation loss
    a = backend.square(x[:, :image_height-1, :image_width-1, :] - x[:, 1:, :image_width-1, :])
    b = backend.square(x[:, :image_height-1, :image_width-1, :] - x[:, :image_height-1, 1:, :])
    return backend.sum(backend.pow(a + b, total_variation_loss_factor))#TV正则项损失因子

loss += total_variation_weght * total_variation_loss(combination_image)
# TV正则项权重*损失函数total variation loss

outputs = [loss]
outputs += backend.gradients(loss, combination_image)

def evaluate_loss_and_gradients(x):
    #评估 loss值 和梯度
    x = x.reshape((1, image_height, image_width, CHANNELS))
    outs = backend.function([combination_image], outputs)([x])
    loss = outs[0]
    gradients = outs[1].flatten().astype("float64")
    return loss, gradients

class Evaluator:
    def loss(self, x):
        loss, gradients = evaluate_loss_and_gradients(x)
        self._gradients = gradients
        return loss

    def gradients(self, x):
        return self._gradients

evaluator = Evaluator()


x = np.random.uniform(0, 255, (1, image_height, image_width, 3)) - 128.
# 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high.

for i in range(iterations):
    x, loss, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.gradients, maxfun=20)
    #evaluator.loss loss函数
    # 返回值 x 估计最小值的位置，即loss最小时对应的x
    # loss最小的Func值,即loss值。
    # x.flatten() 最初的猜测，即待更新参数初始值
    # fprime  梯度函数
    # maxfun 功能评估的最大数量
    
    print("Iteration %d loss: %d" % (i, loss))
    
x = x.reshape((image_height, image_width, CHANNELS))
x = x[:, :, ::-1] # bgr ->rgb
# 将之前减去通道颜色均值加回来
x[:, :, 0] += imagenet_mean_rgb_values[2]
x[:, :, 1] += imagenet_mean_rgb_values[1]
x[:, :, 2] += imagenet_mean_rgb_values[0]


x = np.clip(x, 0, 255).astype("uint8") # 防止越界 限制在0-255之间
output_image = Image.fromarray(x)
output_image.save(output_image_path)
plt.imshow(output_image)
# 可视化合并结果
combined = Image.new("RGB", (image_width*3, image_height))
x_offset = 0
for image in map(Image.open, [input_image_path, style_image_path, output_image_path]):
    combined.paste(image, (x_offset, 0))
    x_offset += image_width
combined.save(combined_image_path)
plt.imshow(combined)
plt.show()