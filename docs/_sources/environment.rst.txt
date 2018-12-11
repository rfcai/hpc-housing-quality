Environment and Data Acquisition
================================
To run the software, you need to create an environment using the environment.yml file in the Github repository. To do this, follow these steps:

1. After cloning our repository onto your local machine, in your terminal/command window, `cd` in to the directory where `environment.yml` lives
2. Optional: `cat environment.yml` to view the environment dependencies. Our environment name is called `hpc`.
3. To recreate the environment locally, type `conda env create -f environment.yml`
4. Now you can activate this environment `conda activate hpc` (only for the first time). After this environment exists locally, you can go ahead and use the `source activate hpc` (for Mac) or `activate hpc` (for Windows) command.
5. Open Jupyter after this environment is activated. Now you are able to create a new Jupyter notebook in this environment. i.e. you will see a choice between the  default Python 3 and one in your current environment, e.g. Python [conda env:root].
6. If you want to modify the environment, you can `conda` or `pip` installing new packages in your terminal/command window, while this environment (hpc) is active. Save out the modification to replace the old environment.yml file: `conda env export > environment.yml`
7. To get out of this environment, use `source deactivate hpc` or `deactivate hpc`

You can find the datasets `here <https://drive.google.com/drive/folders/1ZcTc8uSwtxO0G-p1gAOlDrEIvB22kZUf?usp=sharing>`_.