import json
import os
import pathlib

import cv2
import numpy as np

cls2index = {"blue_cone": 1, "yellow_cone": 2, "orange_cone": 3, "large_orange_cone": 4}

def coco2yolo(fscoco_file_dir: os.path, output_ds_path: os.path):
    #constants
    target_shape = (1920, 1200)  # (breite, höhe)  # TODO maybe oberste ?200? pixel ausschneiden und runterskalieren
    border_widht = 140  # according to https://www.fsoco-dataset.com/tools/
    #paths
    img_dir_name = "images"
    images_dir = fscoco_file_dir / img_dir_name
    bounding_boxes_dir = fscoco_file_dir / "bounding_boxes"
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    boundingbox_files = [f for f in os.listdir(bounding_boxes_dir) if os.path.isfile(os.path.join(bounding_boxes_dir, f))]
    #make shure they match up in correct order (probably unneccesary
    #image_files.sort(key=lambda x: str(x)+".json")
    #boundingbox_files.sort(key=lambda x: str(x))

    output_image_path = output_ds_path/"images"
    output_label_path = output_ds_path/"labels"
    output_image_path.mkdir(parents=True, exist_ok=True)
    output_label_path.mkdir(parents=True, exist_ok=True)

    # creeate fscoco.yaml
    yaml = f"# fscoco from https://www.fsoco-dataset.com/\n\
    \n\
    path: {output_ds_path}  # dataset root dir\n\
    train: {img_dir_name}  # train images (relative to 'path')\n\
    val: {img_dir_name}  # val images (relative to 'path') \n\
    test:  # test images (optional)\n\
    \n\
    # Classes (the numbers are set without presedence.)\n\
    names:\n\
      1: blue_cone\n\
      2: yellow_cone\n\
      3: orange_cone\n\
      4: large_orange_cone".replace("\t", "")
    with open(output_ds_path/"fscoco.yaml", 'w+') as f:
        f.writelines(yaml)

    #iterate over all data
    for img_f, bbf in zip(image_files, boundingbox_files):
        assert str(img_f)+".json" == str(bbf)
        img = cv2.imread(os.path.join(images_dir, img_f))

        # remove black outline and scale to 1280x720
        height, width, _ = img.shape
        img = img[border_widht:height-border_widht, border_widht:width-border_widht]
        height -= 2*border_widht
        width -= 2*border_widht
        img = cv2.resize(img, target_shape)


        # scale and translate annotation
        # target annotation:
        # for each object in image: new line (classindex, center_x, center_y, height, width), with positions relative to image size (in range(0, 1))
        labels = []  # [(class, p0, p1, p2, p3)
        with open(bounding_boxes_dir / bbf, 'r') as f:
            tmp = json.load(f)
            # change from JSON format with additional information to python tupel with only class and position
            labels = [(cls2index[label["classTitle"]], int(label["points"]["exterior"][0][0]), int(label["points"]["exterior"][0][1]), int(label["points"]["exterior"][1][0]), int(label["points"]["exterior"][1][1])) for label in tmp["objects"]]
            # change position from pixel position to relative position in changed image
            labels = [(classIndex, (p0-border_widht)/width, (p1-border_widht)/height, (p2-border_widht)/width, (p3-border_widht)/height) for (classIndex, p0, p1, p2, p3) in labels]
            # change position from corners of cone to center of cone + size
            labels = [(classIndex, 0.5*p0+0.5*p2, 0.5*p1+0.5*p3, abs(p2-p0), abs(p3-p1)) for (classIndex, p0, p1, p2, p3) in labels]
        # show image with labels
        #print(f"labels of file {bbf} = {labels}")
        #for label in labels:
        #    cv2.rectangle(img, (int((label[1]-0.5*label[3])*target_shape[0]), int((label[2]-0.5*label[4])*target_shape[1])), (int((label[1]+0.5*label[3])*target_shape[0]), int((label[2]+0.5*label[4])*target_shape[1])), colors[label[0]], 2)  # start_point = koordinates of corner
        #cv2.imshow("first_image", img)
        #cv2.waitKey(0)
        # save image to output_ds_path/images/train2017/str(img_f).jpg and annotation to output_ds_path/labels/train2017/str(img_f).txt
        cv2.imwrite(output_image_path / f"{str(img_f)}.jpg", img)
        with open(output_label_path / f"{str(img_f)}.txt", 'w+') as f:
            f.writelines([' '.join([str(t) for t in label])+"\n" for label in labels])

def mp4tojpgs(video_path:os.path, output_frames_path:os.path):
    vidcap = cv2.VideoCapture(str(video_path))  # os.path: no way cv2 can read this. string that looks like a path: now were going.
    success, image = vidcap.read()
    count = 0
    output_frames_path.mkdir(parents=True, exist_ok=True)
    while success:
        cv2.imwrite(output_frames_path/f"frame{count}.jpg", image)     # save frame as JPEG file
        success, image = vidcap.read()
        print('Read a new frame: ', success)
        count += 1


def cutout_cones_fscoco(fscoco_file_dir:os.path, output_dir:os.path):
    img_dir_name = "images"
    images_dir = fscoco_file_dir / img_dir_name
    bounding_boxes_dir = fscoco_file_dir / "labels"
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    boundingbox_files = [f for f in os.listdir(bounding_boxes_dir) if os.path.isfile(os.path.join(bounding_boxes_dir, f))]
    try:
        index = max([int(str(f).replace(".jpg", "").split("_")[-1]) for f in os.listdir(output_dir) if str(f).endswith(".jpg") and os.path.isfile(os.path.join(output_dir, f))])+1
    except:
        index = 0
    sizes = []  # [(height, width)]
    for img_f, bbf in zip(image_files, boundingbox_files):
        img = cv2.imread(os.path.join(images_dir, img_f))

        height, width, _ = img.shape
        labels = []  # [(class, p0, p1, p2, p3)
        with open(bounding_boxes_dir / bbf, 'r') as f:
            tmp = json.load(f)
            # change from JSON format with additional information to python tupel with only class and position
            labels = [(cls2index[label["classTitle"]], int(label["points"]["exterior"][0][0]), int(label["points"]["exterior"][0][1]), int(label["points"]["exterior"][1][0]), int(label["points"]["exterior"][1][1])) for label in tmp["objects"]]
        for (classIndex, p0, p1, p2, p3) in labels:
            #labels = [(classIndex, (p0-border_widht)/width, (p1-border_widht)/height, (p2-border_widht)/width, (p3-border_widht)/height) for (classIndex, p0, p1, p2, p3) in labels]
            sizes.append((abs(p1-p3), abs(p0-p2)))
            img_cone = img[p1:p3, p0:p2]
            img_cone = cv2.resize(img_cone, (72, 55))
            cv2.imwrite(os.path.join(output_dir, f"cone_{index}.jpg"), img_cone)
            index += 1
    print(f"average bounding box size: ({np.sum([s[0] for s in sizes])/len(sizes)}, {np.sum([s[1] for s in sizes])/len(sizes)})")


def cutout_cones_detect(images_dir:os.path, output_dir:os.path, model):
    image_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f))]
    index = max([int(str(f).replace(".jpg", "").split("_")[-1]) for f in os.listdir(output_dir) if os.path.isfile(os.path.join(output_dir, f))])+1
    for img_path in image_files:
        img_path = images_dir/img_path
        print("read image ", img_path)
        results = model(img_path)
        labels, cord_thres = results.xyxyn[0][:, -1].numpy(), results.xyxyn[0][:, :-1].numpy()
        img_img = cv2.imread(img_path)
        h, w, _ = img_img.shape
        for i in range(len(labels)):
            bounding_box = [int(cord_thres[i][0] * w), int(cord_thres[i][1] * h), int(cord_thres[i][2] * w), int(cord_thres[i][3] * h)]
            img_cone = img_img[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2]]
            cv2.imwrite(output_dir/f"cone_{index}.jpg", img_cone)
            index += 1


def cutout_cones_yoloformat(images_dir: os.path, labels_dir: os.path, output_dir: os.path):
    for label_file in [f for f in os.listdir(labels_dir) if str(f) == "frame_2032.txt"]:
        if len(str(label_file).split(".")) == 3:
            continue
        img_name = label_file.replace(".txt", ".jpg")
        print("read file ", images_dir/img_name)
        img = cv2.imread(str(images_dir/img_name))
        h, w, _ = img.shape
        with open(labels_dir/label_file) as f:
            i = 0
            for line in f.readlines():
                #0 0.4513020833333333 0.51625 0.07135416666666666 0.15916666666666668
                classlabel, cx, cy, sx, sy = line.split(" ")
                classlabel, cx, cy, sx, sy = int(classlabel), float(cx), float(cy), float(sx), float(sy)
                #cutout from img
                print(f"classlabel {classlabel}, cx {cx}, cy {cy}, sx {sx}, sy {sy}")
                cone_img = img[int(h*(cy-0.5*sy)):int(h*(cy+0.5*sy)), int(w*(cx-0.5*sx)):int(w*(cx+0.5*sx))]
                #save
                cv2.imwrite(str(output_dir/f"camL3_{img_name}_cone_{i}.jpg"), cone_img)
                i += 1


def draw_numberd_bounding_boxes(img_file: os.path, label_file: os.path):
    img = cv2.imread(str(img_file))
    h, w, _ = img.shape
    with open(label_file) as f:
        for i, line in enumerate(f.readlines()):
            classid, py, px, sy, sx = line.split(" ")
            classid, px, py, sx, sy = int(classid), float(px)*h, float(py)*w, float(sx)*h, float(sy)*w
            color = (255, 0, 0)  # ?blue
            if classid == 1:
                color = (0, 255, 255)  # ?yelllow
            cv2.rectangle(img, (int(py-0.5*sy), int(px-0.5*sx)), (int(py+0.5*sy), int(px+0.5*sx)), color, 3)
            cv2.putText(img,f"{i}cone_{i})", (int(py), int(px)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, 2)
    cv2.imshow(str(img_file), img)
    cv2.waitKey(0)

if __name__ == "__main__":
    draw_numberd_bounding_boxes(img_file=pathlib.Path("C:/Users/Idefix/PycharmProjects/tmpProject/cam_footage/left_cam_14_46_00/frame_2032.jpg"), label_file=pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fscoco_sample_translated/labels/frame_2032.txt"))
    #cutout_cones_yoloformat(images_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fscoco_sample_translated/images"),
    #                        labels_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fscoco_sample_translated/labels"),
    #                        output_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/keypoints/images"))
    #cutout_cones_fscoco(pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fscoco_sample_translated"), pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/keypoints"))
    #cutout_cones_detect(images_dir=our_images_path, output_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/keypoints"), model=model)
    #coco2yolo(pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fsoco_sample"), pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fscoco_sample_translated"))
    #mp4tojpgs(pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/Basler Cam recordings (Accel)/right_cam.mp4"), pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/Basler Cam recordings (Accel)/right_cam_frames"))
