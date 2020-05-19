import argparse
from sys import platform

from cellyolomodels import *  # set ONNX_EXPORT in cellyolomodels.py
from cellyolo.utils import *


def segmentation(opt):
    img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        _ = load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        torch.onnx.export(model, img, 'weights/export.onnx', verbose=False, opset_version=10)

        # Validate exported model
        import onnx
        model = onnx.load('weights/export.onnx')  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=img_size, half=half)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if opt.half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        print(len(pred))
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i]
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string

            if det is not None and len(det):
                print(len(det))
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                # for mutil detections
                for i in range(len(det)):
                    if i <= len(det) - 2:
                        # print(det[i][0:4])
                        det[i][0:4] = GBCIOU(det[i][0:4], det[i + 1][0:4], x1y1x2y2=True)[0]
                        det[i + 1][0:4] = GBCIOU(det[i][0:4], det[i + 1][0:4], x1y1x2y2=True)[1]
                    else:
                        pass

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        savepath = str(Path(out) / Path(p).name).split('.')[0]
                        with open(savepath + '.txt', 'a') as file:

                            box_x_min = int(xyxy[0])  # 左上角横坐标
                            box_y_min = int(xyxy[1])  # 左上角纵坐标
                            box_x_max = int(xyxy[2])  # 右下角横坐标
                            box_y_max = int(xyxy[3])  # 右下角纵坐标
                            # 转成相对位置和宽高#转换成yolov3的标签格式，需要归一化到（0-1）的范围内
                            x_center = round((box_x_min + box_x_max) / (2 * im0.shape[1]), 4)
                            y_center = round((box_y_min + box_y_max) / (2 * im0.shape[0]), 4)
                            width = round((box_x_max - box_x_min) / im0.shape[1], 4)
                            height = round((box_y_max - box_y_min) / im0.shape[0], 4)

                            file.write(('%g ' * 5 + '\n') % (0, x_center, y_center, width, height))

                    if save_img or view_img:  # Add bbox to image
                        label = '%s %.2f' % (names[int(cls)], conf)

                        # to view of the box detection
                        #plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        # to see of the confidence score
                        #im0=plot_one_cycle(xyxy, im0, label=label, color=colors[int(cls)])
                        # to save only circle segment after cellyolo
                        im0 = plot_one_cycle(xyxy, im0, color=colors[int(cls)])


            # Print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + out + ' ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='cellyolo/cfg/yolov3-tiny-train.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='cellyolo/data/coco.names', help='*.names path')
    parser.add_argument('--weights', type=str, default='cellyolo/weights/best.pt', help='path to weights file')
    parser.add_argument('--source', type=str, default='cellyolo/5000imagesforretraincellyolo', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='cellyolo/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', default=0, help='display results')
    parser.add_argument('--save-txt', action='store_true', default=0, help='display results')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
