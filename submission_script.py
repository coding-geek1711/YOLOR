import json, os, shutil, cv2
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--img_dir', help='directory of submission images')
parser.add_argument('--results_txt', help='results.txt path')
parser.add_argument('--json', help='validation json path')

args = parser.parse_args()

# READ validation.json
valid_json = ''
with open(args.json) as file:
    valid_json = json.load(file)
    
results = ''
with open(args.results_txt) as file:
    results = file.readlines()

results = [i.replace('\n', '') for i in results]
results = [i.split('-') for i in results]

categories = ['empty','fox','skunk','rodent','bird','american crow','american black bear','chicken','virginia opossum','domestic cat','grey fox','rooster','donkey','raven','petrel_chick','goat','pig','shearwater','iguana','cat']

img_dir = args.img_dir

def get_category_id_from_name(name):
    global categories
    
    return categories.index(name)

def get_id_from_img_name(img_name):
    global valid_json
    
    for image in valid_json['images']:
        if image['file_path'].split('/')[-1] == img_name:
            return image['id']
    
    return None

def convert_to_coco(img, bbox):
    height, width, _ = img.shape
    x = bbox[0] * width
    y = bbox[1] * height
    w = bbox[2] * width
    h = bbox[3] * height
    return [int(x - w/2), int(y - h/2), int(x + w/2), int(y + h/2)]

def make_coco_list():
    global results
    global categories
    global valid_json
    global img_dir

    
    lis = []
    for i in tqdm(results):
        dic = {}
        img_name, category_name, score, bbox = i
        
        img_id = get_id_from_img_name(img_name)
        category_id = get_category_id_from_name(category_name)
        score = float(score)

        img_path = f'{img_dir}/{img_name}'
        img = cv2.imread(img_path)
        
        bbox = [float(j) for j in i[-1].split()]
        x1, y1, x2, y2 = convert_to_coco(img, bbox)

        if category_id == 0:
            dic['bbox'] = [0, 0, img.shape[1], img.shape[0]]
        else:
            dic["bbox"] = [x1, y1, x2, y2]

        dic["id"] = img_id
        dic["category_id"] = category_id
        dic["score"] = score

        lis.append(dic)
    
    return lis

def save_json(lis):
    json_object = json.dumps(lis)
    
    with open('submissions_test.json', 'w') as f:
        f.write(json_object)

def main():
    lis = make_coco_list()
    save_json(lis)

    
if __name__ == '__main__':
    main()