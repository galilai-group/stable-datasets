# goal
- complete every empty files related to rsscn7 and resisc45.

## base homepage
* RSSCN7 : https://github.com/palewithout/RSSCN7?tab=readme-ov-file
* RESISC45 : https://github.com/tensorflow/datasets/blob/master/docs/catalog/resisc45.md

# step1
- Fill out .rst files. Refer to `clevrer.rst`, `not_mnist.rst`, and `tiny_imagenet.rst` and follow their format. 

# step2
- Fill out rsscn7.py and resisc45.py. 
- Refer to `cifar10.py` for it. 
- Basic instruction is just use BaseDatasetBuilder.
- Default method to download dataset is CachedSession().get() in utils. But you don't have to care about this if download link is normal.
- If you determined the link cannot be donwloaded in this way, you have to implement `_split_generator` manually under the dataset class, but this is not recommended so you need to discuss with me first. 
- `_generate_examples` is important. If downloaded data is zip file or tar file or other compressed format, determine which format the file is in and convert that into library-friendly format. You can refer to `tiny_imagenet.py` or  `tiny_imagenet_c.py` for _generate_examples part.
- SOURCE and __info__ is also important. Please only fill in the correct information. 
- This is onedrive link for resisc45 : https://onedrive.live.com/?redeem=aHR0cHM6Ly8xZHJ2Lm1zL3UvcyFBbWdLWXpBUkJsNWNhM0hOYUhJbHpwX0lYanM&cid=5C5E061130630A68&id=5C5E061130630A68%21107&parId=5C5E061130630A68%21112&o=OneUp , but this looks just website, not a desired file. Investigate yourself to find the real download link based on the link. 
-  This is repo of RSSCN7, repo itself is a dataset : https://github.com/palewithout/RSSCN7?tab=readme-ov-file


# step3
- Write test_{dataset}.py. Refer to test_clevrer, test_tiny_imagenet, test_not_mnist.


