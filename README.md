# Feature-based Visual Odometry for Bronchoscopy: A Dataset and Benchmark [IROS'23]

---
## Installation of PySLAM

Please check this [file](./README_pyslam.md) for more detail instructions.

---
## Bronchoscopy dataset

Here's the link to download our dataset: [link](https://drive.google.com/file/d/1GUTV17sN5S73YM4d7tjjtz0Z1xnRhJvY/view?usp=sharing)


---
## Usage for bronchoscopy dataset

1. Link the data folder to ```./dataset``` in this architecture

    ```
    PYSLAM
    |__ main_vo.py
    |__ config.ini
    |__ ....
    |___ dataset
        |__ bronchoscopy
            |__ real_seq_xxx_part_x_dif_x.mp4
            |__ real_seq_xxx_part_x_dif_x.txt
            |__ ....
            
    ```
2. Change your ```config.ini``` file accordingly. Find the section for [FINAL_DATASET], change the base path to your project folder. For example: ```base_path = ~/PYSLAM/dataset```. Then change the ```tag=bronchoscopy```. Select the file for evaluation by setting the keyword ```fname=file_to_evaluation```. Choose the correcting camera setting file in ```cam_settings=settings/bronchoscopy.yaml```. While kept other keyword unchanged. 
3. Dependencies: mannually install those libraries if you have module not found error.
4. Camera parameters is stored in ```./settings/*.yaml```, make sure you are using the right camera parameter file
5. Figures will be shown when you run the program. Mannualy save those you think are useful. **Press Q on the image window to close those figures**. Evaluation for APE will be shown after closing the window, save them for further evaluation.
6. Further information about evaluation, check the repo ```evo``` on github.
--- 
