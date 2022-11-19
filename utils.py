import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import xml.etree.ElementTree as ET

class PKLot(Dataset):
    def __init__(self, paths, classes, none_threshold, bucketing=True):
        self.paths = paths
        self.classes = classes

        self.no_occupied_label_threshold = none_threshold

        self.data = []
        self.bucketing = bucketing
        self.read_lot_data()

    def read_lot_data(self):
        to_tensor = transforms.ToTensor()
        for path in self.paths:
            files = os.listdir(path)
            jpgs, xmls = self._seperate_jpg_xml(files)

            for i in range(len(jpgs)):
                image_name = jpgs[i]
                xml_idx = xmls.index(image_name[:-3] + 'xml')
                xml = xmls[xml_idx]
                gold_label = self.xml_to_labels(os.path.join(path, xml))
                if gold_label != -1:
                    image = Image.open(os.path.join(path, image_name))
                    self.data.append( (to_tensor(image), gold_label, str(os.path.join(path, image_name))) )
                else:
                    print("GOLD_LABEL IS -1 FOR " + str(os.path.join(path, image_name)))
            print("finished path: " + path)
        # SET SELF.DATA

    def xml_to_labels(self, xml):
        bucketing = self.bucketing
        tree = ET.parse(xml)
        root = tree.getroot()
        label = -1
        total_empty = 0
        total_nones = 0
        for space in root.findall('space'):
            id = space.get('id')
            occupied = space.get('occupied')
            if occupied is None: #TODO: WHAT DOES THIS MEAN? THAT IS IT OCCUPIED??
                total_nones += 1
            else:
                if int(occupied)==0:
                    total_empty += 1

        if total_nones > self.no_occupied_label_threshold:
            print('returning -1')
            return -1

        if bucketing:
            for idx, bucket in enumerate(self.classes):
                if total_empty <= bucket:
                    label = idx
                    break
        else:
            label = total_empty
        return label

    def _seperate_jpg_xml(self, files):
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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


