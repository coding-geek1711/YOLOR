import os
import darknet 
import cv2
import numpy as np 
from argparse import ArgumentParser
from tqdm import tqdm

def parser():
    parser = ArgumentParser(description='Arguments for validation')
    parser.add_argument('--config', help='cfg file for darknet')
    parser.add_argument('--weights', help='weights file for darknet')
    parser.add_argument('--data', help='data file for darknet')
    parser.add_argument('--input_dir', help='input directory for darknet')
    parser.add_argument('--thresh', default=0.25, help='Threshold for detection')
    return parser.parse_args()

def init_darknet(config_file, weight_file, data_file, batch_size=1):
    network, class_names, class_colors = darknet.load_network(config_file, data_file, weight_file, batch_size)
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_img = darknet.make_image(width, height, 3)
    return network, class_names, width, height, darknet_img

def load_image(img_path, width, height):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (width, height))

def prediction(img, network, class_names, width, height, darknet_img, thresh):
    darknet.copy_image_from_bytes(darknet_img, img.tobytes())
    return darknet.detect_image(network, class_names, darknet_img, thresh)    

def convert_bbox(img, bbox):
    height, width, _ = img.shape
    x = bbox[0]/width
    y = bbox[1]/height
    w = bbox[2]/width
    h = bbox[3]/height
    
    return [x, y, w, h]

def main():
    args = parser()
    network, class_names, width, height, darknet_img = init_darknet(args.config, 
                                                                    args.weights, 
                                                                    args.data)

    preds = []
    error_list = []
    
    for image in tqdm(os.listdir(args.input_dir)):
        try:
            img = load_image(f'{args.input_dir}/{image}', width, height)
            pred = prediction(img, network, class_names, width, height, darknet_img, args.thresh)
            if len(pred) < 1:
#                 print("ERROR in " + image)
                error_list.append(image + "\n")
                continue
            class_name = pred[0][0]
            class_score = pred[0][1]
            bbox = pred[0][2]
            bbox = convert_bbox(img, bbox)

            preds.append('-'.join([image, class_name, class_score, ' '.join([str(i) for i in bbox])]) + '\n')

        except:
            print(f"BIG ERROR IN {image}")
    with open('results_new.txt', 'w') as file:
        file.writelines(preds)
        
    with open('no_predictions.txt', 'w') as file:
        file.writelines(error_list)
if __name__ == '__main__':
    main()