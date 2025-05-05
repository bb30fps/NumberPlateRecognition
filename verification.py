import os 
import xml.etree.ElementTree as ET
from glob import glob
image_dir = "data/images"
anno_dir = "data/annotations"

images = set([os.path.splitext(f)[0] for f in os.listdir(image_dir)])
annos = set([os.path.splitext(f)[0] for f in os.listdir(anno_dir)])

missing_images = annos - images
missing_annos = images - annos

print(f"Missing images for {len(missing_images)} XMLs")
print(f"Missing XMLs for {len(missing_annos)} images")

for xml_file in glob(f"{anno_dir}/*.xml"):
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        assert root.find('filename') is not None
        assert root.find('object/name') is not None
        assert all(root.find(f'object/bndbox/{coord}') is not None 
                  for coord in ['xmin','ymin','xmax','ymax'])
    except Exception as e:
        print(f"Invalid XML {xml_file}: {str(e)}")
