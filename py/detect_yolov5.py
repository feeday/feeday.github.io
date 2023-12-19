# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""

Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Install - config:
    https://www.anaconda.com/download-success
    https://github.com/ultralytics/yolov5
    https://codeload.github.com/ultralytics/yolov5/zip/refs/heads/master
    https://docs.ultralytics.com/zh/
    (base) PS F:\yolov5> pip install yolov5
    (base) PS F:\yolov5> python detect.py

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     screen                          # screenshot
                                                     path/                           # directory
                                                     list.txt                        # list of images
                                                     list.streams                    # list of streams
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/LNwODJXcvt4'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import csv
import os
import platform
import sys
from pathlib import Path

import torch

# 获取当前脚本的路径和根目录
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 根目录
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # 将根目录添加到 PATH 中
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # 相对路径

# 导入所需的模块和函数
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
    increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh
)
from utils.torch_utils import select_device, smart_inference_mode

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # 模型路径或triton URL
        source=ROOT / 'data/images',  # 文件/目录/URL/glob/screen/0(摄像头)
        data=ROOT / 'data/coco128.yaml',  # 数据集.yaml路径（可选）
        imgsz=(640, 640),  # 推理尺寸（高度，宽度）
        conf_thres=0.25,  # 置信度阈值
        iou_thres=0.45,  # NMS IOU 阈值
        max_det=1000,  # 每张图像的最大检测数
        device='',  # CUDA 设备，例如 0 或 0,1,2,3 或 CPU
        view_img=False,  # 显示结果
        save_txt=False,  # 保存结果为 *.txt
        save_csv=False,  # 以 CSV 格式保存结果
        save_conf=False,  # 保存置信度在 --save-txt 标签中
        save_crop=False,  # 保存裁剪的预测框
        nosave=False,  # 不保存图像/视频
        classes=None,  # 按类别筛选：--classes 0 或 --classes 0 2 3
        agnostic_nms=False,  # 类别无关的 NMS
        augment=False,  # 增强推理
        visualize=False,  # 可视化特征
        update=False,  # 更新所有模型
        project=ROOT / 'runs/detect',  # 保存结果的项目/名称
        name='exp',  # 保存结果的项目/名称
        exist_ok=False,  # 存在的项目/名称是否可行，不递增
        line_thickness=3,  # 边界框厚度（像素）
        hide_labels=False,  # 隐藏标签
        hide_conf=False,  # 隐藏置信度
        half=False,  # 使用 FP16 半精度推理
        dnn=False,  # 使用 OpenCV DNN 进行 ONNX 推理
        vid_stride=1,  # 视频帧速率步长
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # 保存推理图像
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.streams') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # 下载

    # 目录
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # 递增运行
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # 创建目录

    # 加载模型
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # 检查图像大小

    # 数据加载器
    bs = 1  # 批量大小
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # 运行推理
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # 预热
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 转为 fp16/32
            im /= 255  # 0 - 255 转为 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # 扩展为批次维度

        # 推理
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS（非极大值抑制）
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # 第二阶段分类器（可选）
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # 定义 CSV 文件的路径
        csv_path = save_dir / 'predictions.csv'

        # 创建或追加到 CSV 文件
        def write_to_csv(image_name, prediction, confidence):
            data = {'Image Name': image_name, 'Prediction': prediction, 'Confidence': confidence}
            with open(csv_path, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not csv_path.is_file():
                    writer.writeheader()
                writer.writerow(data)

        # 处理预测
        for i, det in enumerate(pred):  # 每张图像
            seen += 1
            if webcam:  # 批量大小 >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # 转为 Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # 打印字符串
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # 归一化增益 whwh
            imc = im0.copy() if save_crop else im0  # 用于保存裁剪
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # 将边界框从 img_size 缩放到 im0 大小
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # 打印结果
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # 每类的检测数
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # 添加到字符串

                # 写入结果
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # 整数类别
                    label = names[c] if hide_conf else f'{names[c]}'
                    confidence = float(conf)
                    confidence_str = f'{confidence:.2f}'

                    if save_csv:
                        write_to_csv(p.name, label, confidence_str)

                    if save_txt:  # 写入文件
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # 归一化 xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # 标签格式
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or save_crop or view_img:  # 添加边界框到图像
                        c = int(cls)  # 整数类别
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

            # 显示结果
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # 允许窗口调整大小（Linux）
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 毫秒

            # 保存结果（带有检测的图像）
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' 或 'stream'
                    if vid_path[i] != save_path:  # 新视频
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # 释放以前的视频写入器
                        if vid_cap:  # 视频
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # 流
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # 强制保存为 *.mp4 后缀的结果视频
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # 打印时间（仅推理）
        LOGGER.info(f"{s}{'' if len(det) else '(没有检测到)，'}{dt[1].dt * 1E3:.1f}ms")

    # 打印结果
    t = tuple(x.t / seen * 1E3 for x in dt)  # 每张图像的速度
    LOGGER.info(f'Speed: %.1fms 预处理, %.1fms 推理, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # 更新模型（修复SourceChangeWarning）


def parse_opt():
    # 创建参数解析器
    parser = argparse.ArgumentParser()
    
    # 模型权重路径或Triton URL
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='模型路径或Triton URL')
    
    # 输入源：文件/目录/URL/通配符/屏幕/0（摄像头）
    parser.add_argument('--source', type=str, default=ROOT / '0', help='文件/目录/URL/通配符/屏幕/0（摄像头）')
    
    # 数据集配置文件路径（可选）
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='数据集配置文件路径（可选）')
    
    # 推理图像尺寸（高度，宽度）
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='推理图像尺寸 高度,宽度')
    
    # 置信度阈值
    parser.add_argument('--conf-thres', type=float, default=0.25, help='置信度阈值')
    
    # NMS IoU 阈值
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU 阈值')
    
    # 每张图像的最大检测数
    parser.add_argument('--max-det', type=int, default=1000, help='每张图像的最大检测数')
    
    # CUDA 设备（例如：0或0,1,2,3或cpu）
    parser.add_argument('--device', default='', help='CUDA 设备 例如：0或0,1,2,3或cpu')
    
    # 显示结果
    parser.add_argument('--view-img', action='store_true', help='显示结果')
    
    # 保存结果到文本文件
    parser.add_argument('--save-txt', action='store_true', help='保存结果到*.txt文件')
    
    # 以CSV格式保存结果
    parser.add_argument('--save-csv', action='store_true', help='以CSV格式保存结果')
    
    # 在--save-txt标签中保存置信度
    parser.add_argument('--save-conf', action='store_true', help='在--save-txt标签中保存置信度')
    
    # 保存裁剪的预测框
    parser.add_argument('--save-crop', action='store_true', help='保存裁剪的预测框')
    
    # 不保存图像/视频
    parser.add_argument('--nosave', action='store_true', help='不保存图像/视频')
    
    # 按类别筛选：--classes 0，或--classes 0 2 3
    parser.add_argument('--classes', nargs='+', type=int, help='按类别筛选：--classes 0，或--classes 0 2 3')
    
    # 类别无关的非最大抑制
    parser.add_argument('--agnostic-nms', action='store_true', help='类别无关的非最大抑制')
    
    # 数据增强
    parser.add_argument('--augment', action='store_true', help='数据增强')
    
    # 可视化特征
    parser.add_argument('--visualize', action='store_true', help='可视化特征')
    
    # 更新所有模型
    parser.add_argument('--update', action='store_true', help='更新所有模型')
    
    # 保存结果的项目目录
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='保存结果的项目目录')
    
    # 结果名称
    parser.add_argument('--name', default='exp', help='结果名称')
    
    # 存在时是否覆盖
    parser.add_argument('--exist-ok', action='store_true', help='存在时是否覆盖')
    
    # 边界框线条厚度（像素）
    parser.add_argument('--line-thickness', default=3, type=int, help='边界框线条厚度（像素）')
    
    # 隐藏标签
    parser.add_argument('--hide-labels', default=False, action='store_true', help='隐藏标签')
    
    # 隐藏置信度
    parser.add_argument('--hide-conf', default=False, action='store_true', help='隐藏置信度')
    
    # 使用FP16半精度推理
    parser.add_argument('--half', action='store_true', help='使用FP16半精度推理')
    
    # 使用OpenCV DNN进行ONNX推理
    parser.add_argument('--dnn', action='store_true', help='使用OpenCV DNN进行ONNX推理')
    
    # 视频帧率跨度
    parser.add_argument('--vid-stride', type=int, default=1, help='视频帧率跨度')
    
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # 扩展图像尺寸
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
