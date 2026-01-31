def extract_label(file_path):
    parts = str(file_path).split("/")
    return parts[-2]

print(extract_label("/kaggle/input/mnistasjpg/trainingSet/trainingSet/0/img_10043.jpg"))
print(extract_label("/kaggle/input/mnistasjpg/trainingSet/trainingSet/8/img_230920.jpg"))