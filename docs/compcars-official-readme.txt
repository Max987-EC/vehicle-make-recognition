This file provides documentation for the CompCars dataset.
(http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)

If you use the dataset in your work, please cite our CVPR 2015 paper:
Linjie Yang, Ping Luo, Chen Change Loy, Xiaoou Tang. "A Large-Scale Car Dataset for Fine-Grained Categorization and Verification", In Computer Vision and Pattern Recognition (CVPR), 2015.
 
Descriptions of the folders and files are as follows:

-image:
  Stores all full car images in the path format 'make_id/model_id/released_year/image_name.jpg'.
-label:
  Stores all labels to the full car images in the path format 'make_id/model_id/released_year/image_name.txt'. Each label file has three lines. The first line is a number which is the viewpoint annotation (-1 - uncertain, 1 - front, 2 - rear, 3 - side, 4 - front-side, 5 - rear-side). The second line is the number of the bounding boxes, which is all '1' in the current release. The third line is the coordinates of the bounding box in the format 'x1 y1 x2 y2' in pixels, where 1 <= x1 < x2 <= image_width, and 1 <= y1 < y2 <= image_height. 
-misc:
  -attributes.txt:
    Each line is the attribute annotation for one model which is in the format 'model_id maximum_speed displacement door_number seat_number type'. For car types, a number from 1~12 corresponds to a specific type, which is described in 'car_type.mat'. Unavailable attributes are denoted by '0' or '0.0'.
  -make_model_name.mat
    Cell array 'make_names' provides the projections from 'make_id' to make names, and cell array 'model_names' provides the projections from 'model_id' to model names.
-part:
  Stores all part images in the path format 'make_id/model_id/released_year/part_id/image_name.jpg'. The correspondance of 'part_id' and part names are: 1 - headlight, 2 - taillight, 3 - fog light, 4 - air intake, 5 - console, 6 - steering wheel, 7 - dashboard, and 8 - gear lever.
-train_test_split:
  This folder generally provides all the train/test subsets used in the paper.
  -classification
    Stores the train/test lists for the classification task with full car images in the paper.
  -part:
    Stores the train/test lists for the classification task with car part in the paper.
  -verification:
    'verification_train.txt' is the image list for training the verification models which is also for testing attribute prediction. 'verification_pairs_easy.txt', 'verification_pairs_medium.txt', and 'verification_pairs_hard.txt' are the three sets with different difficulties for testing car verification models. Each line of 'verification_pairs_XXX.txt' is in the format of 'path_to_image_1 path_to_image_2 label' where label is '1' for positive pairs and is '0' for negative pairs.