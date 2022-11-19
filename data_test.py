import os
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

_BASEPATH = 'C:\\Users\\dazet\\Documents\\CSML\\final_project_data\\archive\\PKLot\\'
_CAMERA_ANGLES = ('PUCPR', 'UFPR04', 'UFPR05')
_WEATHER = ('Cloudy', 'Rainy', 'Sunny')

def run():
    spaces_data = {}
    columns = ('angle', 'weather', 'date', 'image_name', 'spaces', 'spaces_occupied', 'spaces_unlabeled')
    for angle in _CAMERA_ANGLES:
        for weather in _WEATHER:
            path = _BASEPATH + angle + "\\" + weather
            dates = os.listdir(path)
            for date in dates:
                print("starting: " + angle + "_" + weather + "_" + date)
                newpath = path + "\\" + date
                files = os.listdir(newpath)
                jpgs, xmls = separate_jpg_xml(files)

                tot_spaces, tot_occupied, tot_unlabled = 0,0,0
                total_values = np.array([tot_spaces, tot_occupied, tot_unlabled])
                for i in range(len(jpgs)):
                    image_name = jpgs[i]
                    xml_idx = xmls.index(image_name[:-3] + 'xml')
                    xml = xmls[xml_idx]
                    spaces, spaces_occupied, spaces_unlabeled = xml_to_labels(os.path.join(newpath, xml))
                    total_values = total_values + np.array([spaces, spaces_occupied, spaces_unlabeled])
                    row = (angle, weather, date, image_name, spaces, spaces_occupied, spaces_unlabeled)
                    key = angle + "_" + weather + "_" + date + "_" + image_name
                    spaces_data[key] = row

    a = pd.DataFrame.from_dict(spaces_data, orient='index', columns=columns)
    a.to_csv("./data_labeled_stats.csv")

def xml_to_labels(xml):
    tree = ET.parse(xml)
    root = tree.getroot()
    label = -1
    total_occupied = 0
    total_empty = 0
    total_nones = 0
    total = 0
    for space in root.findall('space'):
        id = space.get('id')
        total += 1
        occupied = space.get('occupied')
        if occupied is None: #TODO: WHAT DOES THIS MEAN? THAT IS IT OCCUPIED??
            total_nones += 1
            total_occupied += 1
        else:
            if int(occupied) == 0:
                total_empty +=1
            else:
                total_occupied += int(occupied)

    return total, total_occupied, total_nones

def separate_jpg_xml(files):
    jpgs, xmls = [], []
    for f in files:
        if f[-4:] == '.xml':
            xmls.append(f[:-4])
        elif f[-4:] == '.jpg':
            jpgs.append(f[:-4])
        else:
            print("found non jpg/xml: ", str(f))

    common = list(set(xmls) & set(jpgs))
    jpgs, xmls,  = [], []
    for c in common:
        jpgs.append(c + '.jpg')
        xmls.append(c+'.xml')
    return jpgs, xmls

if __name__ == "__main__":
    df = run()