from models.deeplabv3 import DeepLabV3Student
from models.mobile_unet import MobileUNetStudent
from models.fssnet import FSSNetStudent


def build_student(student_name):
    student_classes = {"DeepLabV3_MobileNetV2": DeepLabV3Student, "FssNet": FSSNetStudent, "Mobile_UNet": MobileUNetStudent, }
    return student_classes[student_name]()
