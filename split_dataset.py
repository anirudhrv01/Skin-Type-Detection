import os
import shutil
import random
from sklearn.model_selection import train_test_split

source_dir='data/raw'
output_dir='data/processed'

train_ratio=0.7
val_ratio=0.15
test_ratio=0.15
random.seed(42)

classes=os.listdir(source_dir)
for cls in classes:
    class_path=os.path.join(source_dir,cls)
    
    if not os.path.isdir(class_path):
        continue
    images=os.listdir(class_path)
    
    train_images, temp_images=train_test_split(
        images,
        test_size=1-train_ratio,
        random_state=42
    )
    val_images,test_images=train_test_split(
        temp_images,
        test_size=(test_ratio/(test_ratio+val_ratio)),
        random_state=42
    )
    def copy_files(image_list,split_name):
        for img in image_list:
            src=os.path.join(class_path,img)
            dst=os.path.join(output_dir,split_name,cls)
            os.makedirs(dst,exist_ok=True)
            shutil.copy(src,os.path.join(dst,img))
    copy_files(train_images,'train')
    copy_files(val_images,'val')
    copy_files(test_images,'test')
print("Dataset split completed successfully.")