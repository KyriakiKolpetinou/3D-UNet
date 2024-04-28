def change_img_to_label_path(path):
    parts = list(path.parts)
    parts[parts.index("imagesTr")] = "labelsTr"
    return Path(*parts)


path = Path("/content/imagesTr")
subject_paths = list(path.glob("liver_*"))
subjects = [] #store torch.io subjects

for subject_path in subject_paths:
  label_path = change_img_to_label_path(subject_path)
  subject = tio.Subject({"CT": tio.ScalarImage(subject_path), "Label": tio.LabelMap(label_path)})
  subjects.append(subject)