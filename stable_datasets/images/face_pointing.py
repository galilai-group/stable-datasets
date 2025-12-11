# The url is no longer valid. The code is commented out.


# import tarfile
# import numpy as np
# import datasets
# import re
# from PIL import Image
# import io
#
#
# class FacePointing(datasets.GeneratorBasedBuilder):
#     """Head angle classification dataset."""
#
#     VERSION = datasets.Version("1.0.0")
#
#     def _info(self):
#         return datasets.DatasetInfo(
#             description="The head pose database consists of 15 sets of images. Each set contains 2 series of 93 images "
#                         "of the same person at different poses.",
#             features=datasets.Features({
#                 "image": datasets.Image(),
#                 "person_id": datasets.Value("int32"),
#                 "angles": datasets.Sequence(datasets.Value("int32"))
#             }),
#             supervised_keys=("image", "angles"),
#             citation="""@inproceedings{gourier2004estimating,
#                         title={Estimating face orientation from robust detection of salient facial features},
#                         author={Gourier, Nicolas and Hall, Daniela and Crowley, James L},
#                         booktitle={ICPR International Workshop on Visual Observation of Deictic Gestures},
#                         year={2004},
#                         organization={Citeseer}}
#                      """
#         )
#
#     def _split_generators(self, dl_manager):
#         urls = {
#             f"http://www-prima.inrialpes.fr/perso/Gourier/Faces/Person{i:02d}-{j}.tar.gz": f"Person{i:02d}-{j}.tar.gz"
#             for i in range(1, 16) for j in range(1, 3)
#         }
#
#         archive_paths = dl_manager.download_and_extract(urls)
#
#         return [
#             datasets.SplitGenerator(
#                 name=datasets.Split.TRAIN,
#                 gen_kwargs={"archive_paths": archive_paths}
#             )
#         ]
#
#     def _generate_examples(self, archive_paths):
#         for path in archive_paths.values():
#             with tarfile.open(path, "r:gz") as tar:
#                 for member in tar.getmembers():
#                     if member.isfile():
#                         match = re.search(r"personne(\d{2})_([+-]\d+)_([+-]\d+)", member.name)
#                         if match:
#                             person_id, vert_angle, horiz_angle = map(int, match.groups())
#                             file = tar.extractfile(member)
#                             image = Image.open(io.BytesIO(file.read())).convert("RGB")
#
#                             yield f"{person_id}_{vert_angle}_{horiz_angle}", {
#                                 "image": image,
#                                 "person_id": person_id,
#                                 "angles": [vert_angle, horiz_angle]
#                             }
