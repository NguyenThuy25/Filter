import hydra
import numpy as np
import cv2
from omegaconf import DictConfig
import pyrootutils
from pytorch_lightning import LightningModule

from src import utils
from src.data.dlib_dataset import TransformDataset
from src.models.dlib_module import DlibLitModule
import albumentations as A
from albumentations import Compose
from albumentations.pytorch.transforms import ToTensorV2


log = utils.get_pylogger(__name__)
pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

def add_transparent_image(background, foreground, x_offset, y_offset):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

    return background

def detect_face(cfg: DictConfig, glasses_path):
    net: LightningModule = hydra.utils.instantiate(cfg.get('net'))
    model = DlibLitModule.load_from_checkpoint(checkpoint_path="/Users/admin/Downloads/Work/filter_project-main/ckpt/last_100.ckpt", net=net)
    glasses_img = cv2.imread(glasses_path, cv2.IMREAD_UNCHANGED)  # IMREAD_UNCHANGED => open image with the alpha channel

    faceCascade = cv2.CascadeClassifier('/Users/admin/Downloads/Work/filter_project-main/src/haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture('/Users/admin/Downloads/Work/filter_project-main/data/Video/test_vid.mov')
    # cap = cv2.VideoCapture(1)
    transform = Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, img = cap.read()
        if not ret:    # ADD hereeeeeeeee
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,     
            scaleFactor=1.2,
            minNeighbors=5,     
            minSize=(20, 20)
        )
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            cropped_img = img[y:y+h, x:x+w]
            cropped_img = np.array(cropped_img)
            cropped_img = transform(image=cropped_img)
            keypoints = model(cropped_img['image'].unsqueeze(0)).detach().numpy()
            for i in range(68):
                keypoints[0][i][0] = (keypoints[0][i][0] + 0.5)*h + x
                keypoints[0][i][1] = (keypoints[0][i][1] + 0.5)*w +y
                tmp_x, tmp_y = keypoints[0][i][0], keypoints[0][i][1]
                cv2.circle(img, (int(tmp_x), int(tmp_y)), 2, (0, 255, 0), -1)
            
            left_eye_x, left_eye_y = int(keypoints[0][36][0]), int(keypoints[0][36][1])
            right_eye_x, right_eye_y = int(keypoints[0][45][0]), int(keypoints[0][45][1])
            if left_eye_x is not None and right_eye_x is not None:
                    glasses_width = int(1.5 * abs(right_eye_x - left_eye_x))
                    glasses_height = int(glasses_width * glasses_img.shape[0] / glasses_img.shape[1])
                    center_x = int((left_eye_x + right_eye_x) / 2)
                    center_y = int((left_eye_y + right_eye_y) / 2)
                    glasses_left = int(center_x - glasses_width / 2)
                    glasses_right = int(center_x + glasses_width / 2)
                    glasses_top = int(center_y - glasses_height / 2)
                    glasses_bottom = int(center_y + glasses_height / 2)

                    
                    cv2.circle(img, (glasses_left, glasses_top), 2, (255, 0, 0), -1)
                    cv2.circle(img, (glasses_left, glasses_bottom), 2, (255, 0, 0), -1)
                    cv2.circle(img, (glasses_right, glasses_top), 2, (255, 0, 0), -1)
                    cv2.circle(img, (glasses_right, glasses_bottom), 2, (255, 0, 0), -1)
            
            glasses_resized = cv2.resize(glasses_img, (glasses_width, glasses_height))
            face_with_glasses = add_transparent_image(img, glasses_resized, glasses_left, glasses_top)
            
            out.write(face_with_glasses) 
            cv2.imshow('video',face_with_glasses)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

@hydra.main(version_base="1.3", config_path="../configs/model", config_name="dlib_resnet.yaml")
def main(cfg: DictConfig) -> None:
    detect_face(cfg, '/Users/admin/Downloads/Work/filter_project-main/data/glasses.png')
if __name__ == "__main__":
    main()

