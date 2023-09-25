# Description
this project is an example of using 'cookiecutter', 'DVC', and 'git' in a real project.in this repository, we use stable diffusion to generate defects.

# instruction
* install cookiecutter on the local device
* Create a repository in GitHub and connect with cookiecutter
* DVC gets start
* DVC data versioning
* DVC data pipeline

## Install cookiecutter on the local device
do this steps one by one
install cookiecutter on env

        pip install cookiecutter

install cookiecutter

        cd ~
        cookiecutter cookiecutter-pypackage/

fill the 

* "full_name": your name
* "email": your email address
* "github_user_name": fill user name of your GitHub account
* "project_name": name of the project 
* "project_slug": name of the project if wants to install with 'pip install <project_slug>'
* "project_short_description": describe your project.
* "pypi_username": user name of PyPI
* "version": version of the project
* "use_pytest": [y/n]
* "use_black": [y/n]
* "use_pypi_deployment_with_travis": [y/n]
* "add_pyup_badge": [y/n]
* "Select command_line_interface": [1/2/3]
* "create_author_file": [y/n]
* "Select open_source_license": [1/2/3/4/5/6]

## create a repository in GitHub and connect with cookiecutter
After installing cookiecutter 
go to GitHub and create a new repository with the name 'project_slug'
then follow the steps to connect the cookiecutter project to the GitHub repository.

        cd <project_slug>
        git init
        git add .
        git commit -m "first commit"
        git branch -M main
        git remote add origin https://github.com/<github_username>/<project_slug>.git
        git push -u origin main
## DVC gets start
initialize DVC and commit to GitHub

        dvc init
        git commit -m "Initialize DVC"

**directory structure**

        .
        ├── AUTHORS.rst
        ├── CONTRIBUTING.rst
        ├── docs
        │   ├── authors.rst
        │   ├── conf.py
        │   ├── contributing.rst
        │   ├── history.rst
        │   ├── index.rst
        │   ├── installation.rst
        │   ├── make.bat
        │   ├── Makefile
        │   ├── readme.rst
        │   └── usage.rst
        ├── HISTORY.rst
        ├── LICENSE
        ├── Makefile
        ├── MANIFEST.in
        ├── <project_slug>
        │   ├── cli.py
        │   ├── __init__.py
        │   └── <project_slug>.py
        ├── README.rst
        ├── requirements_dev.txt
        ├── setup.cfg
        ├── setup.py
        ├── tests
        │   ├── __init__.py
        │   └── test_<project_slug>.py
        └── tox.ini

make a new directory with the name of the data to put your data on it

        mkdir data

rename the directory of <project_slug> to model to put your model files in it

        mv /path/to/<project_slug>/<project_slug> /path/to/<project_slug>/models

make new directory results to outputs in this directory

        mkdir results

make newe directory of src to put train.py , inference.py , ...

        mkdir src

**note:** *data*, *models*, and *results* contain files that are being stored and versioned by DVC, and *src* contains scripts for training and evaluating the model as well as tests and scripts for pipelines and APIs.

**note:** You can make new directory *notebooks* to Jupyter Notebooks used for the exploratory analysis, development of models, or data manipulation.

Then do

        git add .
        git commit -m "dvc data structure updated"
        git push
# DVC data versioning
Now it is time to put your data in the data directory.
after that, you should 'dvc add'
        
        dvc add data/images

after running the up command. **images.dvc** and **.gitignore** file will create in data directory.
in gitignore wrote **/images** to ignore uploading this directory to GitHub because it should upload to dvc.

**challeng:** In adding a directory containing images to "dvc add data/images" if dvc got  permission error do this

        chmod +rwx path/to/images

**note:** images directory contain multi images should add like this *dvc add data/images*  not this *dvc add data/images/*

**note:** images.dvc contains information of your images in .dvc/cache directory. Take care of it. if it is removed after pushing dvc . you can not pull your data again. 

It is time to git add

        git add data/images.dvc data/.gitignore
        git commit -m "Add raw data"

Now it is time to add dvc remote to googleDrive but before that you should install its package 

        pip install dvc[gdrive]

so add remote to your Google Drive
let's imagine your googleDrive folder path

**https://drive.google.com/drive/folders/14BN_D1HWKFw0POOJHPf5HqjW1bZayNCC**

put last part of the path in remote add

        dvc remote add -d <remote_name> gdrive://14BN_D1HWKFw0POOJHPf5HqjW1bZayNCC

now dvc push to upload data to googleDrive

        dvc push


now lets imagine your image directory is removed or .dvc/cache is removed.you can just have your data with (but rememmber *data/images.dvc* should not be removed)

now let's imagine your image directory is removed or .dvc/cache is removed. You can just have your data with (but remember *data/images.dvc* should not be removed)


        dvc pull

# DVC data pipeline

DVC allow you to better organize projects and reproduce complete workflows and results.

to defining each *stage* of this workflow you should use:**"dvc stage add"**

**dvc stage parameters:**

dvc stage add
* -n stage name
* -p stage dependency or parameters
* -d stage input files
* -o stage output file or directory


in out example there is 3 stage
* resize
* train
* inference

in **resize** step .we resize raw images to train.it could be any preproccess step.
Lets define this stage

        dvc stage add   -n resize \ 
                        -d  data/scratch_cv \ 
                        -d src/resize.py \ 
                        -p resize.img_type,\
                        resize.input_dir,\
                        resize.output_dir,\
                        resize.size \ 
                        -o data/resized \ 
                        python src/resize.py params.yaml


with running this command you can check root directory there is a file with name **"dvc.yaml"** which contain some information about the stage.in addition you should make a file **params.yaml** in root directory to read stages parameter values from that. for examlpe resize stage 

By running this command you can check the root directory there is a file with the name **"dvc.yaml"** which contains some information about the stage. In addition you should make a file **params.yaml** in the root directory to read stages parameter values from that. for example resize the stage 

        resize:
                input_dir: "data/scratch_cv"
                output_dir: "data/resized"
                size: 128
                img_type: "jpg"

Now lets define train stage
Now let's define train stage

        dvc stage add -n train \ 
                      -d data/resized \ 
                      -d src/train.py \ 
                      -p train.checkpointing_steps,\
                      train.dataset_name,\
                      train.instance_data_dir,\
                      train.instance_prompt,\
                      train.max_train_steps,\
                      train.num_train_epochs,\
                      train.pretrained_model_name_or_path,\
                      train.resolution,\
                      train.sample_batch_size,\
                      train.train_batch_size,\
                      train.trained_model_path \ 
                      -o results/scratch/scratch.ckpt 
                      python src/train.py params.yaml

***dvc.yaml*** will update after running the command.
also you should define train parameters in **param.yaml**.

last stage is inference
also, you should define train parameters in **param.yaml**.

the last stage is the inference

        dvc stage add -n app
                      -d results/scratch/scratch.ckpt
                      -d src/app.py
                      -p  app.input_checkpoint_path,\
                      app.output_checkpoint_path \
                      -o results/scratch/trained_model \
                      python src/app.py params.yaml

like preview step ***dvc.yaml*** will change in this step and inference stage parameters should be define by you in **params.yaml** 

like preview step ***dvc.yaml*** will change in this step and inference stage parameters should be defined by you in **params.yaml** 


Now git add

        git add .gitignore data/.gitignore dvc.yaml
        git commit -m "pipline defined"

its time to reproducing pipeline

        dvc repro

after reproducing.you should commit dvc.lock

        git add dvc.lock && git commit -m "first pipeline repro"

you can watch pipeline by 

        dvc dag

After changing parameters. if it need . you should **dvc repro** again
