from PIL import Image
import resnet50_model as RN50


epoch = 50
train_path = "/home/JHlin/seed_class/plant-seedlings-classification/train"
test_path = "/home/JHlin/seed_class/plant-seedlings-classification/test"
class_idx = RN50.run(train_path, epoch)
RN50.test(test_path, class_idx)

