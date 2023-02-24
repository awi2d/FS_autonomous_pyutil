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


def cutout_cones_yoloformat(images_dir: os.path, labels_dir: os.path, output_dir: os.path):
    sizes = []
    for label_file in [f for f in os.listdir(labels_dir)]:
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
                sizes.append((sx, sy))
                #cutout from img
                print(f"classlabel {classlabel}, cx {cx}, cy {cy}, sx {sx}, sy {sy}")
                cone_img = img[int(h*(cy-0.5*sy)):int(h*(cy+0.5*sy)), int(w*(cx-0.5*sx)):int(w*(cx+0.5*sx))]
                #save
                cv2.imwrite(str(output_dir/f"camL3_{img_name}_cone_{i}.jpg"), cone_img)
                i += 1
    print(np.average([x for (x, y) in sizes]), np.average([y for (x, y) in sizes]))


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


def draw_cone_keypoints(img_file: os.path, label_file: os.path):
    img = cv2.imread(str(img_file))
    img_h, img_w, _ = img.shape
    print("(h, w) = ", (img_h, img_w))
    with open(label_file) as f:
        for line in f.readlines():
            #C:\Users\Idefix\PycharmProjects\datasets\keypoints\images\camL3_frame_1572.jpg_cone_1.jpg,0.7283333333333334#0.07,0.46944444444444444#0.31777777777777777,
            if line.split(",")[0].split("\\")[-1] == str(img_file).split("\\")[-1]:
                for keypoint in line.split(",")[1:]:
                    pos_width, pos_hight = keypoint.split("#")
                    pos_width, pos_hight = float(pos_width)*img_w, float(pos_hight)*img_h
                    color = (255, 0, 0)  # blue
                    cv2.circle(img, (int(pos_width),int(pos_hight)), 5, color, -1)
                break
    #cv2.rectangle(img, (ecke1.wpos, ecke1.hpos), (ecke2.wpos, ecke2.hpos), (255, 0, 255), 3)  # width = ecke2.wpos-ecke1.wpos
    cv2.imshow(str(img_file), img)
    cv2.waitKey(0)


def draw_numberd_bounding_boxes(img_file: os.path, label_file: os.path, keypoint_file: os.path=None):
    img = cv2.imread(str(img_file))
    img_h, img_w, _ = img.shape
    print("(h, w) = ", (img_h, img_w))
    keypoints = {}
    if keypoint_file is not None:
        with open(keypoint_file) as f:
            for line in f.readlines()[1:]:
                imgnr_conenr = line.split(",")[0].split("/")[-1]  # line = /cones/camL3_frame_1572.jpg_cone_1.jpg,float#float,...
                if not imgnr_conenr.startswith("cone_"):
                    tmp = imgnr_conenr.split("_")
                    cam = tmp[0]
                    frnr = int(tmp[2].replace(".jpg", ""))
                    conenr = int(tmp[4].replace(".jpg", ""))
                    keypoints[(cam, frnr, conenr)] = [(float(kp.split("#")[0]), float(kp.split("#")[1])) for kp in line.split(",")[1:]]

    framenumber = int(str(img_file).split("\\")[-1].split("_")[-1].replace(".jpg", ""))
    with open(label_file) as f:
        for i, line in enumerate(f.readlines()):
            classid, pos_width, pos_hight, width, height = line.split(" ")
            classid, pos_width, pos_hight, width, height = int(classid), float(pos_width)*img_w, float(pos_hight)*img_h, float(width)*img_w, float(height)*img_h
            color = (255, 0, 0)  # blue
            if classid == 1:
                color = (0, 255, 255)  # yelllow
            cv2.rectangle(img, (int(pos_width-0.5*width), int(pos_hight-0.5*height)), (int(pos_width+0.5*width), int(pos_hight+0.5*height)), color, 3)
            if (framenumber, i) in keypoints.keys():
                print(f"add keypoints to cone {i}")
                for keypoint in keypoints[("camL3", framenumber, i)]:
                    cv2.circle(img, (int(pos_width-0.5*width+width*keypoint[0]), int(pos_hight-0.5*height+height*keypoint[1])), 5, color, -1)
            cv2.putText(img, f"{i}cone_{i})", (int(pos_width), int(pos_hight)), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 1, 2)
    #cv2.rectangle(img, (ecke1.wpos, ecke1.hpos), (ecke2.wpos, ecke2.hpos), (255, 0, 255), 3)  # width = ecke2.wpos-ecke1.wpos
    cv2.imshow(str(img_file), cv2.resize(img, (1800, 1000)))
    cv2.waitKey(0)

def bb_overlap(bb_a, bb_b):
    # gven two bounding boxes in the (class, center_width, center_height, size_width, size_height) format, computes the fraction that the intersection of these boxes cover of the larger of the two boxes.
    (cls_a, pw_a, ph_a, sw_a, sh_a) = bb_a
    (cls_b, pw_b, ph_b, sw_b, sh_b) = bb_b

    sw = min(pw_a+0.5*sw_a, pw_b+0.5*sw_b)-max(pw_a-0.5*sw_a, pw_b-0.5*sw_b)
    sh = min(ph_a+0.5*sh_a, ph_b+0.5*sh_b)-max(ph_a-0.5*sh_a, ph_b-0.5*sh_b)
    if sw < 0 or sh < 0:
        return 0  # boxes dont overlap
    return sw*sh/max(sw_a*sh_a, sw_b*sh_b)


def poii_bb_files2std():
    poiibb_dir = pathlib.Path("C:/Users/Idefix/PycharmProjects/OpenLabeling/main/output/YOLO_darknet/")
    vp_labels_dir = pathlib.Path("C:/Users/Idefix/PycharmProjects/tmpProject/vp_labels/")  # cone-keypoint annotations for all
    bbi_poii_dir = pathlib.Path("C:/Users/Idefix/PycharmProjects/tmpProject/vp_labels/bbi_poii")
    # line in file in input_dir: poii, bb
    # line in output_idr/camL3_bb: cls, bb
    # math bbs, write poii of bb in (same row as bb in output_dir/camL3_bb) in output_dir/bbi_poii/same_name_as_filre_in_output_dir/camL3_bb
    poiibb_files = [file for file in os.listdir(poiibb_dir)]
    clsbb_files = [file for file in os.listdir(vp_labels_dir/"camL3_bb")]
    # should be equal
    fully_annotated_files = set(poiibb_files).intersection(set(clsbb_files))

    fully_annotated_files = list(fully_annotated_files)
    fully_annotated_files.sort()
    for filename in fully_annotated_files:
        poii_bb = []
        with open(poiibb_dir/filename) as f:
            poii_bb = f.readlines()
        #classid, pos_width, pos_hight, width, height = line.split(" ")
        if len(poii_bb) == 0:
            continue
        print("\nfilename =", filename)
        poii_bb = [(int(classid), float(pos_width), float(pos_hight), float(width), float(height)) for (classid, pos_width, pos_hight, width, height) in [line.split(" ") for line in poii_bb]]
        print(f"poii_bb = {poii_bb[:5]}")
        cls_bb = []
        with open(vp_labels_dir/"camL3_bb"/filename) as f:
            cls_bb = f.readlines()
        cls_bb = [(int(classid), float(pos_width), float(pos_hight), float(width), float(height)) for (classid, pos_width, pos_hight, width, height) in [line.split(" ") for line in cls_bb]]
        bbi_poii_file = [-1 for _ in range(len(cls_bb))]
        for cls_bb_i, line in enumerate(cls_bb):
            max_poii_bb_i = -1
            max_overlap = 0
            for poii_bb_i in range(len(poii_bb)):
                if bb_overlap(line, poii_bb[poii_bb_i]) > max_overlap:
                    max_poii_bb_i = poii_bb_i
                    max_overlap = bb_overlap(line, poii_bb[poii_bb_i])
            if max_overlap > 0.6:
                # line and poii_bb[poii_line_i] are the same bb
                poii = poii_bb[max_poii_bb_i][0]+9
                if cls_bb[0] == 0:
                    assert 8 < poii < 47  # only cones with poii in this range are blue
                if cls_bb[1] == 1:
                    assert 46 < poii < 88  # ony cones with poii in this range are yellow
                    assert poii != 68  # 68 is no cone
                bbi_poii_file[cls_bb_i] = poii
        with open(bbi_poii_dir/filename, 'w') as f:
            f.write("\n".join([str(index0) for index0 in bbi_poii_file]))

if __name__ == "__main__":
    poii_bb_files2std()
    datasets_path = pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/")
    #cutout_cones_yoloformat(images_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/OpenLabeling/main/input"), labels_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/OpenLabeling/main/output/YOLO_darknet/"), output_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/keypoints/cones/"))
    #for frnr in range(2360, 4021, 10):
    i = 1280
    draw_numberd_bounding_boxes(img_file=datasets_path/pathlib.Path(f"testrun_2022_12_17/cam_footage/left_cam_14_46_00/camL3_frame_{i}.jpg"), label_file=pathlib.Path(f"C:/Users/Idefix/PycharmProjects/tmpProject/vp_labels/camL3_bb/camL3_frame_{i}.txt"), keypoint_file=datasets_path/pathlib.Path("keypoints/cone_annotations.csv"))
    #draw_cone_keypoints(img_file=datasets_path/pathlib.Path("keypoints/cones/camL3_frame_1562.jpg_cone_0.jpg"), label_file=datasets_path/pathlib.Path("keypoints/cone_annotations.csv"))
    #cutout_cones_yoloformat(images_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fscoco_sample_translated/images"),
    #                        labels_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fscoco_sample_translated/labels"),
    #                        output_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/keypoints/images"))
    #cutout_cones_fscoco(pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fscoco_sample_translated"), pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/keypoints"))
    #cutout_cones_detect(images_dir=our_images_path, output_dir=pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/keypoints"), model=model)
    #coco2yolo(pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fsoco_sample"), pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/fscoco_sample_translated"))
    #mp4tojpgs(pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/Basler Cam recordings (Accel)/right_cam.mp4"), pathlib.Path("C:/Users/Idefix/PycharmProjects/datasets/Basler Cam recordings (Accel)/right_cam_frames"))
